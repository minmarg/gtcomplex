/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/cutemplates.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/fields.cuh"
#include "reformatmatch.cuh"

// #define CUDP_RFMY_MTCH_COMPLEX_TESTPRINT 0 //-1 //0

// -------------------------------------------------------------------------
// ReformatMatched4Complexes: reformat matched (aligned) positions of 
// full complex (all chains) to obtain continuous stretch of aligned 
// positions;
// NOTE: MAX_NCHAINS, max #complex chains for this kernel to continue;
// NOTE: CPXTYPENUMBERSCT, memory step number section to keep
// NOTE: alignment-determined complex types;
// ndbCstrs, total number of reference chains in a chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
// assgmaxnsteps, max number of steps for complex chain assignments;
// assgstepnumber, step number for complex chain assignments;
// complexstepnumber, step number corresponding to memory section to read
// matched coordinates from;
// wrkmem2, working memory for chain scores and assignments;
// wrkmemaux, auxiliary working memory;
// tmpdpalnpossbuffer, source/destination of copied coordinates;
// final, final reformat for the production of final scores;
// 
template<
    int MAX_NCHAINS,
    int CPXTYPENUMBERSCT>
__global__
void ReformatMatched4Complexes(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint assgmaxnsteps,
    const uint assgstepnumber,
    const uint complexstepnumber,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpalnpossbuffer,
    const int final)
{
    // blockDim.x == #matched positions read and written in parallel;
    // blockDim.y == 6 coordinate sections read and written in parallel;
    // blockIdx.x is the reference complex serial number;
    // blockIdx.y is the query complex serial number;
    enum {bmQRYNDX, bmRFNNDX, bmTotal};
    enum {
        //max #complex chains:
        nchns = CUSF2_TBSP_INDEX_SCORE_COMPLEX_NCHAINS_CACHE_SIZE,
        ntmpa = 7,//#temporary cells
        lYDIM = CUDP_REFORMAT_MATCHED_DIM_Y,
        lXDIM = CUDP_REFORMAT_MATCHED_DIM_X
    };
    //cache for chain assignment:
    __shared__ int chnassg[nchns + ntmpa];
    int* tmparr = chnassg + nchns;
    //cache for query and reference coordinates:
    // __shared__ float crdCache[nTDPAlignedPoss][lXDIM+1];
    const uint dbcpxndx = blockIdx.x;
    const uint qrycpxndx = blockIdx.y;
    uint dbcpxdst;//distance in positions to the first chain of complex dbcpxndx;
    int dbcpxN, dbcpxdstN;//#chains and #chains up to complex dbcpxndx
    int qrycpxN/* , qrycpxdstN */;//#chains and #chains up to complex qrycpxndx
    int swapped = 0;//signifies if chain assignment is indexed by reference chain sn

    //check convergence first
    if(threadIdx.y == 4 && threadIdx.x == 0) {
        //NOTE: read convergence flag at 0:
        uint mloc0 = ((qrycpxndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        tmparr[4] = wrkmemaux[mloc0 + dbcpxndx];
    }

    if(threadIdx.y == 0 && threadIdx.x == 0)
        tmparr[0] = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);
    if(threadIdx.y == 1 && threadIdx.x == 0)
        tmparr[1] = GetDbStrField<LNTYPE,pcx2DDstN>(dbcpxndx);
    if(threadIdx.y == 2 && threadIdx.x == 0)
        tmparr[2] = GetQueryStrField<INTYPE,pcx2DN>(qrycpxndx);
    // if(threadIdx.y == 3 && threadIdx.x == 0)
    //     tmparr[3] = GetQueryStrField<LNTYPE,pcx2DDstN>(qrycpxndx);

    __syncthreads();

    if((tmparr[4] & (CONVERGED_LOWTMSC_bitval)))
        //not overwriting tmparr[4] until the next sync;
        //all threads in the block exit upon convergence;
        return;

    dbcpxN = tmparr[0]; dbcpxdstN = tmparr[1];
    qrycpxN = tmparr[2]; //qrycpxdstN = tmparr[3];


    if((MAX_NCHAINS > 0) && (MAX_NCHAINS < qrycpxN) && (MAX_NCHAINS < dbcpxN))
        return;//block exits


    if(threadIdx.y == 1 && threadIdx.x == 0)
        tmparr[5] = GetDbStrDst(dbcpxdstN);

    int Q = qrycpxN, R = dbcpxN;
    if(R < Q) { myhdswap(Q, R); swapped = 1; }

    __syncthreads();

    const uint dblen = ndbCposs + dbxpad;
    //target offset to the beginning of complex data along the y axis;
    //NOTE 0 for stepnumber!
    const uint yoffft = (qrycpxndx * maxnsteps + 0) * dblen * nTDPAlignedPoss;
    int totlen = 0;//total #aligned positions for this complex pair
    //resultant alignment-dependent type length and resultant type:
    int typelen = 0, cpxtype = gtmtProtein;
    dbcpxdst = (uint)tmparr[5];

    //copy coordinates to new, complex-specific location
    for(int qb = 0; qb < Q && (final? 1: qb < nchns); qb += nchns)
    {
        //NOTE: assigned chain id required only when NOT swapped!
        for(int q = (threadIdx.y * lXDIM) + threadIdx.x;
           (!swapped) && qb + q < Q && q < nchns; q += lYDIM * lXDIM)
        {
            const uint locdbstrndx = dbcpxdstN + (uint)(qb + q);
            const uint mloct = 
                ((qrycpxndx * assgmaxnsteps + assgstepnumber) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
            int r = fabsf(wrkmem2[mloct + twm2aCCAssignment * ndbCstrs + locdbstrndx]);//int<-float
            chnassg[q] = r;
        }

        __syncthreads();

        for(int q = 0; qb + q < Q && q < nchns; q++) {
            const int r = chnassg[q];
            // const uint qryndx = (uint)(qrycpxdstN + (swapped? r: (qb + q)));//global query chain sn
            const uint dbstrndx = (uint)(dbcpxdstN + (swapped? (qb + q): r));//global reference chain sn
            //source offset to the beginning of the data along the y axis wrt query qryndx:
            const uint yofffs = (qrycpxndx * maxnsteps + complexstepnumber) * dblen * nTDPAlignedPoss;
            uint dbstrdst;
            int alnlen, moltype;
            //read #positions
            if(threadIdx.y == 0 && threadIdx.x == 0) {
                uint mloc0 = ((qrycpxndx * maxnsteps + complexstepnumber) * nTAuxWorkingMemoryVars) * ndbCstrs;
                tmparr[0] = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs + dbstrndx];//int<-float
            }
            if(threadIdx.y == 1 && threadIdx.x == 0) {
                tmparr[1] = dbstrdst = GetDbStrDst(dbstrndx);
                //NOTE: molecules of consistent types aligned only! Take reference:
                tmparr[2] = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst);
            }
            __syncthreads();
            alnlen = tmparr[0];
            moltype = GetMoleculeType(tmparr[2]);
            dbstrdst = (uint)tmparr[1];
            __syncthreads();
            for(int sp = threadIdx.x; sp < alnlen; sp += lXDIM) {
                //read coordinates as is in the order reverse to alignment;
                //threadIdx.y * dblen represents a particular coordinates section in nTDPAlignedPoss:
                uint poss = yofffs + dbstrdst + sp  + threadIdx.y * dblen;//source position
                uint post = yoffft + dbcpxdst + (totlen + sp)  + threadIdx.y * dblen;//target position
                //NOTE: no need for cache when complexstepnumber != 0 (assumed)!
                // crdCache[threadIdx.y][threadIdx.x] = tmpdpalnpossbuffer[poss];
                tmpdpalnpossbuffer[post] = tmpdpalnpossbuffer[poss];//READ/WRITE
            }
            totlen += alnlen;
            GetComplexResultantType(cpxtype, typelen, moltype, alnlen);
        }
    }

    //save totlen:
    if(threadIdx.y == 0 && threadIdx.x == 0) {
        uint mloc0 = ((qrycpxndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs + dbcpxndx] = totlen;
    }
    //save cpxtype:
    if(threadIdx.y == 0 && threadIdx.x == 0) {
        uint mloct = ((qrycpxndx * maxnsteps + CPXTYPENUMBERSCT) * nTAuxWorkingMemoryVars) * ndbCstrs;
        wrkmemaux[mloct + tawmvNAlnPoss * ndbCstrs + dbcpxndx] = cpxtype;
    }
}

// -------------------------------------------------------------------------
// Instantiations
//
#define INSTANTIATE_ReformatMatched4Complexes(tpMAX_NCHAINS,tpCPXTYPENUMBERSCT) \
    template __global__ void ReformatMatched4Complexes<tpMAX_NCHAINS,tpCPXTYPENUMBERSCT>( \
        const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, const uint maxnsteps, \
        const uint assgmaxnsteps, const uint assgstepnumber, const uint complexstepnumber, \
        float* __restrict__ wrkmem2, float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpalnpossbuffer, const int  final);

INSTANTIATE_ReformatMatched4Complexes(0,sfin_cpx_complextypenumber2);
INSTANTIATE_ReformatMatched4Complexes(CUS1_TBSP_CPXSCORE_MAX_NCHAINS,sfin_cpx_complextypenumber2);

// =========================================================================
// -------------------------------------------------------------------------
