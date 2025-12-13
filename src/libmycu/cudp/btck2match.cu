/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/cudp/dpw_complex_com.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "btck2match.cuh"

// #define CUDP_MTCH_ALN_TESTPRINT 0 //-1 //0

// =========================================================================

// NOTE: parameters are passed to the device via constant memory and are 
// limited to 4 KB
// 
// device functions for finalizing MAP dynamic programming;
// NOTE: ANCHORRGN, template parameter, anchor region is in use:
// NOTE: Regions outside the anchor are not explored,
// NOTE: decreasing computational complexity;
// NOTE: BANDED, template parameter, banded alignment;
// NOTE: COMPLEX, template parameter, flag of processing complexes;
// NOTE: MAX_NCHAINS, max #complex chains for this kernel to continue;
// NOTE: thread block is 2D to process one query-reference structure pair;
// NOTE: keep #registers below 32!
// ndbCstrs, number of references in a chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
// stepnumber, step number which also corresponds to the superposition
// variant used;
// complexstepnumber, step number corresponding to memory section for 
// matched coordinates to be written;
// assgmaxnsteps, max number of steps for complex chain assignments;
// assgstepnumber, step number for complex chain assignments;

// -------------------------------------------------------------------------
// BtckToMatched32x: device code for copying with 32x unrolling the 
// coordinates of matched (aligned) positions to destination location;
// NOTE: memory pointers should be aligned!
// btckdata, backtracking information data;
// wrkmemaux, auxiliary working memory;
// tmpdpalnpossbuffer, destination of copied coordinates;
// wrkmem2, working memory for chain scores and assignments;
// 
template<
    bool ANCHORRGN,
    bool BANDED,
    bool COMPLEX,
    int MAX_NCHAINS>
__global__
void BtckToMatched32x(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint stepnumber,
    const char* __restrict__ btckdata,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem2,
    const uint complexstepnumber,
    const uint assgmaxnsteps,
    const uint assgstepnumber)
{
    // blockDim.x defines the number of matched positions read and written in 
    // parallel within a block;
    // blockDim.y (3 or 6) number of coordinates read and written in parallel 
    // within a block;
    // blockIdx.x is the reference serial number;
    // blockIdx.y is the query serial number;
    //cache of the query and reference match positions:
    enum {bmQRYNDX, bmRFNNDX, bmTotal};
    __shared__ int posCache[bmTotal][CUDP_MATCHED_DIM_X+1];
    //cache of the query and reference coordinates:
    __shared__ float crdCache[nTDPAlignedPoss][CUDP_MATCHED_DIM_X+1];
    const uint dbstrndx = blockIdx.x;
    const uint qryndx = blockIdx.y;
    uint dbcpxndx = dbstrndx;//reference complex serial number
    uint qrycpxndx = qryndx;//query complex serial number
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos = 0, rfnpos = 0;
    int sfragndx = 0, sfragpos = 0;
    int fraglen = 0;


    if(COMPLEX || MAX_NCHAINS > 0) {
        if(threadIdx.x == 0 && threadIdx.y == 0) posCache[0][4] = GetQueryStrField<INTYPE,pps2DCpxI>(qryndx);
        if(threadIdx.x == 0 && threadIdx.y == 1) posCache[0][5] = GetDbStrField<INTYPE,pps2DCpxI>(dbstrndx);
        if(MAX_NCHAINS > 0 && threadIdx.x == 0 && threadIdx.y == 0)
            posCache[0][6] = GetQueryStrField<INTYPE,pcx2DN>(posCache[0][4]);
        if(MAX_NCHAINS > 0 && threadIdx.x == 0 && threadIdx.y == 1)
            posCache[0][7] = GetDbStrField<INTYPE,pcx2DN>(posCache[0][5]);
        __syncthreads();
        //NOTE: posCache[0][4-7] cannot be overwritten until the next sync!
        if(MAX_NCHAINS > 0 &&
          (MAX_NCHAINS < posCache[0][6]) && (MAX_NCHAINS < posCache[0][7]))
            return;//block exits
        if(COMPLEX) {
            qrycpxndx = posCache[0][4];
            dbcpxndx = posCache[0][5];
        }
    }

    //check convergence first
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        //NOTE: reuse cache to read convergence flag at both 0 and stepnumber:
        uint mloc0 = ((qrycpxndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        posCache[1][0] = wrkmemaux[mloc0 + dbcpxndx];//float->int
        if(stepnumber == 0) posCache[1][1] = posCache[1][0];
    }

    if(stepnumber != 0 && threadIdx.x == 0 && threadIdx.y == 1) {
        uint mloc = ((qrycpxndx * maxnsteps + stepnumber) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        posCache[1][1] = wrkmemaux[mloc + dbcpxndx];//float->int
    }

    __syncthreads();

    if((posCache[1][0] & (CONVERGED_LOWTMSC_bitval)) ||
       (posCache[1][1] & (CONVERGED_SCOREDP_bitval | CONVERGED_NOTMPRG_bitval | CONVERGED_LOWTMSC_bitval)))
    {   //all threads in the block exit upon appropriate convergence;
        //NOTE: set alignment length (#matched/aligned positions) at pos==0 to 0 so that refinement halts:
        uint mloc0 = ((qrycpxndx * maxnsteps + complexstepnumber/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        //NOTE: for complexes, multiple blocks write to the same location the SAME value!
        //NOTE: no race condition!
        if(threadIdx.x == 0 && threadIdx.y == 0)
            wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs + dbstrndx] = 0.0f;//alnlen;
        return;
    }


    //reuse cache
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        posCache[0][0] = GetDbStrLength(dbstrndx);
        posCache[0][2] = GetQueryLength(qryndx);
    }

    if(threadIdx.x == 0 && threadIdx.y == 1) {
        posCache[0][1] = dbstrdst = GetDbStrDst(dbstrndx);
        posCache[0][3] = qrydst = GetQueryDst(qryndx);
        posCache[0][4] = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst);
        posCache[0][5] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(qrydst);
    }

    if(ANCHORRGN) {
        if((threadIdx.x == tawmvQRYpos || threadIdx.x == tawmvRFNpos ||
        threadIdx.x == tawmvSubFragNdx || threadIdx.x == tawmvSubFragPos) &&
        threadIdx.y == 0)
        {
            //NOTE: reuse cache to contain query and reference positions and other fields
            //structure-specific-formatted data: 4 uncoalesced reads
            //NOTE: backtracking information always corresponds to slot 0, sfragfct:
            uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
            crdCache[0][threadIdx.x] = wrkmemaux[mloc + threadIdx.x * ndbCstrs + dbstrndx];
        }
    }

    __syncthreads();

    // if(GetMoleculeType(posCache[0][4]) != GetMoleculeType(posCache[0][5]))
    //     return;//NOTE: no write to [4,5] until the next sync!

    if(COMPLEX &&
       ChainsWithinAssignment<CUDP_MATCHED_DIM_X/*xdim*/,CUDP_MATCHED_DIM_Y/*ydim*/>(
            qrycpxndx, dbcpxndx,  qryndx, dbstrndx,  ndbCstrs,
            assgmaxnsteps, assgstepnumber,
            wrkmem2, crdCache[1]) == 0)
        //all threads exit if chains are not mutually assigned;
        return;

    if(GetMoleculeType(posCache[0][4]) != GetMoleculeType(posCache[0][5])) {
        uint mloc0 = ((qrycpxndx * maxnsteps + complexstepnumber) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(threadIdx.x == 0 && threadIdx.y == 0)
            wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs + dbstrndx] = 0.0f;//alnlen;
        return;//NOTE: no write to [4,5] until the next sync!
    }

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlen = posCache[0][0]; dbstrdst = posCache[0][1];
    qrylen = posCache[0][2]; qrydst = posCache[0][3];

    if(ANCHORRGN) {
        qrypos = crdCache[0][tawmvQRYpos]; rfnpos = crdCache[0][tawmvRFNpos];
        sfragndx = crdCache[0][tawmvSubFragNdx]; sfragpos = crdCache[0][tawmvSubFragPos];
    }

    __syncthreads();


    if(ANCHORRGN) {
        fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
        qrypos += sfragpos; rfnpos += sfragpos;
    }

    int x, y;

    GetTerminalCellXY<ANCHORRGN,BANDED>(x, y,
        qrylen, dbstrlen, qrypos, rfnpos, fraglen);


#ifdef CUDP_MTCH_ALN_TESTPRINT
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        if((CUDP_MTCH_ALN_TESTPRINT>=0)? dbstrndx==CUDP_MTCH_ALN_TESTPRINT: 1)
            printf(" MTCH: bid= %u tid= %u: len= %d addr= %u "
                "qrypos= %d rfnpos= %d fraglen= %d (y= %d x= %d)\n",
                dbstrndx,threadIdx.x,dbstrlen,dbstrdst,
                qrypos,rfnpos,fraglen,y,x
            );
    }
#endif


    int alnlen = 0;
    char btck = dpbtckDIAG;
    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx:
    //NOTE: write alignment at pos==0 for refinement to follow!
    //NOTE: for complexes, nothing changes: one-to-one chain assignment ensures
    //NOTE: at most only one reference chain aligned to one query chain!
    //NOTE: alignment occupies the same max of space!
    const int yofff = (qrycpxndx * maxnsteps + complexstepnumber/*sfragfct*/) * dblen * nTDPAlignedPoss;
    //const int yofff = (qryndx * maxnsteps + stepnumber/*sfragfct*/) * dblen * nTDPAlignedPoss;


    //backtrace over the alignment
    while(btck != dpbtckSTOP) {
        int ndx = 0;
        if(threadIdx.x == 0 && threadIdx.y == 0) {
            //thread 0 records matched positions
            for(; ndx < CUDP_MATCHED_DIM_X;) {
                if(x < 0 || y < 0) {
                    btck = dpbtckSTOP;
                    break;
                }
                int qpos = (qrydst + y) * dblen + dbstrdst + x;
                btck = btckdata[qpos];//READ
                if(btck == dpbtckSTOP)
                    break;
                if(btck == dpbtckUP) {
                    y--; 
                    continue; 
                }
                else if(btck == dpbtckLEFT) { 
                    x--; 
                    continue; 
                }
                //(btck == dpbtckDIAG)
                posCache[bmQRYNDX][ndx] = y;
                posCache[bmRFNNDX][ndx] = x;
                x--; y--; ndx++;
            }
            //save ndx and btck values in cache:
            posCache[bmQRYNDX][CUDP_MATCHED_DIM_X] = ndx;
            posCache[bmRFNNDX][CUDP_MATCHED_DIM_X] = btck;
        }

        __syncthreads();

        ndx = posCache[bmQRYNDX][CUDP_MATCHED_DIM_X];
        btck = posCache[bmRFNNDX][CUDP_MATCHED_DIM_X];

#ifdef CUDP_MTCH_ALN_TESTPRINT
        if(threadIdx.x == 0 && threadIdx.y == 0) {
            if((CUDP_MTCH_ALN_TESTPRINT>=0)? dbstrndx==CUDP_MTCH_ALN_TESTPRINT: 1){
                for(int ii=0; ii<ndx; ii++) printf(" %5d", posCache[bmQRYNDX][ii]);
                printf("\n");
                for(int ii=0; ii<ndx; ii++) printf(" %5d", posCache[bmRFNNDX][ii]);
                printf("\n\n");
            }
        }
#endif

        //write the coordinates of the matched positions to gmem
        if(threadIdx.x < ndx) {
            //READ coordinates
            int pos = qrydst + posCache[bmQRYNDX][threadIdx.x];
            int crd = threadIdx.y;
            //field section (query, reference):
            int fldsec = ndx_qrs_dc_pm2dvfields_;
#if (CUDP_MATCHED_DIM_Y == 3)
            crdCache[threadIdx.y][threadIdx.x] = GetQueryCoord(crd, pos);
            pos = dbstrdst + posCache[bmRFNNDX][threadIdx.x];
            crdCache[threadIdx.y+pmv2DNoElems][threadIdx.x] = GetDbStrCoord(crd, pos);
#elif (CUDP_MATCHED_DIM_Y == 6)
            if(pmv2DNoElems <= threadIdx.y) {
                pos = dbstrdst + posCache[bmRFNNDX][threadIdx.x];
                crd -= pmv2DNoElems;
                fldsec = ndx_dbs_dc_pm2dvfields_;
            }
            crdCache[threadIdx.y][threadIdx.x] = GetQryRfnCoord(fldsec, crd, pos);
#else
#error "INVALID EXECUTION CONFIGURATION: Assign CUDP_MATCHED_DIM_Y to 3 or 6."
#endif
            //WRITE coordinates in the order reverse to alignment itself;
            //take into account #positions written already (alnlen);
            //threadIdx.y * dblen represents a particular coordinates section in nTDPAlignedPoss:
            pos = yofff + dbstrdst + alnlen  + threadIdx.y * dblen;
            tmpdpalnpossbuffer[pos+threadIdx.x] = crdCache[threadIdx.y][threadIdx.x];
#if (CUDP_MATCHED_DIM_Y == 3)
            pos += pmv2DNoElems * dblen;
            tmpdpalnpossbuffer[pos+threadIdx.x] = crdCache[threadIdx.y+pmv2DNoElems][threadIdx.x];
#endif
        }
        __syncthreads();
        //#aligned positions have increased by ndx
        alnlen += ndx;
    }


    //WRITE #matched (aligned) positions 
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        //NOTE: write alignment length at pos==0 for refinement to follow
        uint mloc0 = ((qrycpxndx * maxnsteps + complexstepnumber/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        //uint mloc0 = ((qryndx * maxnsteps + stepnumber) * nTAuxWorkingMemoryVars) * ndbCstrs;
        //NOTE: for complexes, one-to-one chain assignment ensures
        //NOTE: at most only one reference chain (dbstrndx) aligned to one query chain!
        //NOTE: dbstrndx cannot be referenced more than once for the same query complex!
        wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs + dbstrndx] = alnlen;
    }


#ifdef CUDP_MTCH_ALN_TESTPRINT
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        if((CUDP_MTCH_ALN_TESTPRINT>=0)? dbstrndx==CUDP_MTCH_ALN_TESTPRINT: 1)
            printf(" MTCH (pronr=%u): y= %d x= %d alnlen= %d qrylen= %d "
                "dbstrlen= %d\n\n", dbstrndx,y,x,alnlen,qrylen,dbstrlen
            );
    }
#endif
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_BtckToMatched32x(tpANCHORRGN,tpBANDED,tpCOMPLEX,tpMAX_NCHAINS) \
    template \
    __global__ void BtckToMatched32x<tpANCHORRGN,tpBANDED,tpCOMPLEX,tpMAX_NCHAINS>( \
        const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, const uint stepnumber, \
        const char* __restrict__ btckdata, float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpalnpossbuffer, float* __restrict__ wrkmem2, \
        const uint complexstepnumber, \
        const uint assgmaxnsteps, const uint assgstepnumber);

INSTANTIATE_BtckToMatched32x(false,false,false,0);
// INSTANTIATE_BtckToMatched32x(false,true);
// INSTANTIATE_BtckToMatched32x(true,false);
// INSTANTIATE_BtckToMatched32x(true,true);
INSTANTIATE_BtckToMatched32x(false,false,true,0);

INSTANTIATE_BtckToMatched32x(false,false,false,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);
INSTANTIATE_BtckToMatched32x(false,false,true,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);

// -------------------------------------------------------------------------
