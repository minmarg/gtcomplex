/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/cutemplates.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "chainassign.cuh"

// -------------------------------------------------------------------------
// MakeChain2ChainAssignment: make complex chain to chain assignment
// based on TM-scores only using the Hungarian algorithm;
// WRITESCORE, template parameter, falg of writing accumulated score across
// assigned chains;
// WRITEASSG, template parameter, flag of writing the assignment;
// PASS2, template parameter of the 2nd pass, which will take account of
// chain orientations;
// MAX_NCHAINS, max #complex chains for this kernel to continue computation;
// nqystrs, total number of queries in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCcpxs, total number of reference complexes in the chunk;
// maxnqrychains, max number of chains over all query complexes in the chunk;
// maxnrfnchains, max number of chains over all reference complexes in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// maxnstepsmem2, maxnsteps version for wrkmem2;
// NOTE: memory pointers should be aligned!
// tfmmem, transformation matrix address space;
// wrkmemaux, auxiliary working memory;
// wrkmem2, working memory for chain scores and assignments;
// NOTE: thread block is 1D and processes one complex pair;
// NOTE: Code is based on e-maxx::algo
//
template<
    bool WRITESCORE,
    bool WRITEASSG,
    bool PASS2,
    int MAX_NCHAINS>
__global__
void MakeChain2ChainAssignment(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint /*ndbCcpxs*/,
    const uint maxnqrychains,
    const uint maxnrfnchains,
    const uint maxnsteps,
    const uint maxnstepsmem2,
    const float* __restrict__ tfmmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2)
{
    // enum {lTOPNxMAXS = CUS1_TBSP_DPSCORE_TOP_N_CONFIGS};//CLOptions::GetC_DPSCORE_TOP_N_CONFIGS()
    const uint qrycpxndx = blockIdx.x;//query complex serial number
    const uint dbcpxndx = blockIdx.y;//reference complex number
    const uint sfragfct = blockIdx.z;//fragment factor

    uint linkqrychain, linkrfnchain;//unused when PASS2==false

    if(PASS2) {
        //NOTE: sfragfct represents the product of the numbers of query and reference chains;
        //NOTE: linkqrychain and linkrfnchain are the query and reference chain ids
        //NOTE: relative to which all other chain orientations are evaluated;
        //NOTE: choosing maxnqrychains for division is important!
        linkrfnchain = sfragfct / maxnqrychains;//reference chain id
        linkqrychain = sfragfct - linkrfnchain * maxnqrychains;//query chain id
    }

    //number of chains of both complexes
    uint maxn1chains = maxnqrychains;
    uint maxn2chains = maxnrfnchains;
    int  swapped = 0;
    if(maxn2chains < maxn1chains) myhdswap(maxn1chains, maxn2chains);
    const uint maxn2chains1 = maxn2chains + 1;

    //cache for assignments and related data
    //(use dynamically allocated SM to dynamically achieve efficient occupancy):
    extern __shared__ float dataCache[];
    float* meanorientCh = dataCache;//mean orientation
    float* orientsCh = dataCache;//chain orientations
    float* convergCh = dataCache;//cache for divergence values
    float* tmpbuffloat = dataCache;//temporary buffer size for floats
    if(PASS2) {
        orientsCh = meanorientCh + nTTranformMatrix;
        convergCh = orientsCh + nTTranformMatrix * lTFMsperWarp * lXDIMnwarps;
        tmpbuffloat = convergCh + lXDIM;
    }
    float* ys = tmpbuffloat + lXDIMnwarps;//source potentials (max size: maxn1chains)
    float* yt = ys + maxn1chains;//target potentials
    float* min_to = yt + maxn2chains1;//min reduced cost over edges from Z (source subset) to chain c2
    int*   QtoR = (int*)(min_to + maxn2chains1);//chains of complex 1 assigned to complex 2's chains; -1, no assignment
    int*   prv = QtoR + maxn2chains1;//previous (complex 2's) chain on alternating path
    int*   in_Z = prv + maxn2chains1;//flags of whether (complex 2's) chain is in Z
    int*   assgn = in_Z + maxn2chains1;//final result: assignment of max size maxn1chains
    int*   tmpbufint = assgn + maxn1chains;//temporary buffer size for ints of size lXDIMnwarps

    int dbcpxN, qrycpxN;//#reference and query complex chains
    int dbcpxDstN, qrycpxDstN;//#chains up to this reference and query complex

    //check convergence first
    if(threadIdx.x == 0) {
        //NOTE: reuse cache to read convergence flag at 0:
        uint mloc0 = ((qrycpxndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        min_to[0] = wrkmemaux[mloc0 + dbcpxndx];//NOTE: min_to is at least 1 in size
    }

    if(threadIdx.x == 0)
        prv[0] = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);
    if((lWarpsize < lXDIM && threadIdx.x == lWarpsize) || (lXDIM <= lWarpsize && threadIdx.x == 0))
        prv[1] = (int)GetDbStrField<LNTYPE,pcx2DDstN>(dbcpxndx);
    if((2 * lWarpsize < lXDIM && threadIdx.x == 2 * lWarpsize) || (lXDIM <= lWarpsize && threadIdx.x == 0))
        prv[2] = GetQueryStrField<INTYPE,pcx2DN>(qrycpxndx);
    if((3 * lWarpsize < lXDIM && threadIdx.x == 3 * lWarpsize) ||
       (lWarpsize < lXDIM && threadIdx.x == lWarpsize) || (lXDIM <= lWarpsize && threadIdx.x == 0))
        prv[3] = (int)GetQueryStrField<LNTYPE,pcx2DDstN>(qrycpxndx);

    if(PASS2 && threadIdx.x < nTTranformMatrix) meanorientCh[threadIdx.x] = 0.0f;

    __syncthreads();

    if((((int)(min_to[0])) & (CONVERGED_LOWTMSC_bitval)))
        //not overwriting min_to until the next sync;
        //all threads in the block exit upon convergence;
        return;

    dbcpxN = prv[0]; dbcpxDstN = prv[1];
    qrycpxN = prv[2]; qrycpxDstN = prv[3];

    //logical numbers of queries and references (chains):
    int Q = qrycpxN, R = dbcpxN;

    if(R < Q) { myhdswap(Q, R); swapped = 1; }


    if((MAX_NCHAINS > 0) && (MAX_NCHAINS < qrycpxN) && (MAX_NCHAINS < dbcpxN))
        return;//block exits

    if(PASS2) {
        if((qrycpxN <= linkqrychain) || (dbcpxN <= linkrfnchain)) {
            //relative query and/or reference chains are out ofbounds for
            //this complex pair: all threads in the block exit;
            return;
        }
        //read globally best chain-level transformation matrix:
        int mloc0 = ((qrycpxDstN + linkqrychain) * ndbCstrs + (dbcpxDstN + linkrfnchain)) * nTTranformMatrix;
        if(threadIdx.x < nTTranformMatrix)
            meanorientCh[threadIdx.x] = tfmmem[mloc0 + threadIdx.x];
        // obsolete:
        // CalculateMeanOrientationVectorPass2
        //     <lWarpsize, llog2warpsize, lXDIM, lXDIMnwarps, lTFMsperWarp>(
        //         Q, swapped,  qrycpxDstN, dbcpxDstN,  nqystrs, ndbCstrs,  maxnsteps,
        //         tfmmem, wrkmemaux,  assgn, orientsCh, meanorientCh);
    }


    //initialize
    for(int r = threadIdx.x; r <= R; r += lXDIM) {
        QtoR[r] = -1;
        yt[r] = 0.0f;
        if(r < Q) ys[r] = 0.0f, assgn[r] = -1;
    }

    __syncthreads();

    //convert to orientation vector after sync!
    if(PASS2) Tfm2OVec(meanorientCh, 0/*tid*/);

    //assign query chain q
    for(int q = 0; q < Q; q++) {
        int rr = R, il = 0;//running r, inner loop iterations
        if(threadIdx.x == 0) QtoR[rr] = q;

        for(int r = threadIdx.x; r <= R; r += lXDIM) {
            min_to[r] = (float)lInf;
            prv[r] = -1;
            in_Z[r] = 0;//false
        }

        __syncthreads();

        while(QtoR[rr] != -1 && il++ <= q) {//max q+1 cycles
            if(threadIdx.x == 0) in_Z[rr] = 1;//true;
            const int qq = QtoR[rr];
            float delta = (float)lInf;
            int rr_next = lInf;

            __syncthreads();

            for(int r0 = 0; r0 < R; r0 += lXDIM)
            {
                if(PASS2) {
                    convergCh[threadIdx.x] = 0.0f;
                    //read tfm matrices, construct orientation vectors, and calculate
                    //divergence relative to the mean vector:
                    GetDivOrientationVectorsPass2
                        <lWarpsize, llog2warpsize, lXDIM, lXDIMnwarps, lTFMsperWarp>(
                        qq, r0, R,  swapped,  qrycpxDstN, dbcpxDstN,  nqystrs,  ndbCstrs,
                        tfmmem,  in_Z,  meanorientCh, orientsCh, convergCh);
                }
                //read TM-scores for costs
                int r = r0 + threadIdx.x;
                if(r < R && in_Z[r] == 0) {
                    uint qryndx = qrycpxDstN + (swapped? r: qq);
                    uint dbstrndx = dbcpxDstN + (swapped? qq: r);
                    uint mloc2 = ((qryndx * maxnstepsmem2 + (PASS2? 0: sfragfct)) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
                    //TM-score negated for minimization
                    float wgt_qq_r = -wrkmem2[mloc2 + twm2aChnScore * ndbCstrs + dbstrndx];
                    if(PASS2) wgt_qq_r *= convergCh[r];
                    float diff = wgt_qq_r - ys[qq] - yt[r];
                    if(diff < min_to[r]) { min_to[r] = diff; prv[r] = rr; }
                    //minimums across thread block:
                    if(min_to[r] < delta) { delta = min_to[r]; rr_next = r; }
                }
            }

            __syncthreads();

            //find minimum across min_to[.] and thread warps in parallel:
            for(int ldim = (lWarpsize>>1); ldim >= 1; ldim >>= 1) {
                int lid = threadIdx.x & (lWarpsize-1);//lane id
                float deltaoff = __shfl_down_sync(0xffffffff, delta, ldim);
                int rr_nextoff = __shfl_down_sync(0xffffffff, rr_next, ldim);
                if(lid < ldim && deltaoff < delta) {
                    delta = deltaoff; rr_next = rr_nextoff;
                }
            }
            __syncthreads();
            //write warp-level minimums to smem:
            if((threadIdx.x & (lWarpsize-1)) == 0) {
                tmpbuffloat[(threadIdx.x >> llog2warpsize)] = delta;
                tmpbufint[(threadIdx.x >> llog2warpsize)] = rr_next;
            }
            __syncthreads();
            //find minimum in the buffer
            for(int xdim = (lXDIMnwarps>>1); xdim >= 1; xdim >>= 1) {
                if(threadIdx.x < xdim &&
                   tmpbuffloat[threadIdx.x + xdim] < tmpbuffloat[threadIdx.x])
                {
                    tmpbuffloat[threadIdx.x] = tmpbuffloat[threadIdx.x + xdim];
                    tmpbufint[threadIdx.x] = tmpbufint[threadIdx.x + xdim];
                }
                __syncthreads();
            }
            delta = tmpbuffloat[0];
            rr = tmpbufint[0];//rr=rr_next;
            __syncthreads();

            //update arrays:
            //(delta is non-negative except the first cycle)
            for(int r = threadIdx.x; r <= R; r += lXDIM) {
                if(in_Z[r]) {
                    atomicAdd(ys + QtoR[r], delta); yt[r] -= delta;
                } else
                    min_to[r] -= delta;
            }
            __syncthreads();
        }//while(QtoR[rr] != -1)

        __syncthreads();//mystic

        //update assignments along alternating path
        //(can be parallel in two-stages /reusing in_Z/ but it's not as efficient)
        if(threadIdx.x == 0)//safe serial version
            for(int r; rr != R; rr = r) {
                if((r = prv[rr]) < 0) break;
                int pv = QtoR[r];
                QtoR[rr] = pv;
                assgn[pv] = rr;//final assignment
            }

        __syncthreads();
    }//for(;q < Q;)

    if(WRITESCORE && threadIdx.x == 0) {
        uint qryndx = qrycpxndx;
        uint sfragfct2 = sfragfct;
        if(PASS2) {//partitioning sfragfct into query and fragfct ids
            qryndx = sfragfct / maxnstepsmem2;
            sfragfct2 = sfragfct - qryndx * maxnstepsmem2;
            qryndx += qrycpxDstN;
        }
        //write TM-score for each complex pair serially in the corresponding section:
        const uint mloc2 = ((qryndx * maxnstepsmem2 + sfragfct2) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
        //-yt[R] is the sum of all deltas; take abs(.) for positive scores:
        float cpxscore = yt[R];
        //write complex (cumulative) TM-score, yt[R];
        wrkmem2[mloc2 + twm2aCpxScore * ndbCstrs + dbcpxndx] = cpxscore;
    }

    if(WRITEASSG) {
        //write assignment (chain indices) to gmem;
        //NOTE: Q < dbcpxN always
        for(int q = threadIdx.x; q < Q; q += lXDIM) {
            const uint qryndx = qrycpxndx;//complex's sn;//qrycpxDstN;//query complex's 1st chain index
            const uint dbstrndx = dbcpxDstN + q;//reference complex's 1st chain index + running index
            //NOTE: leave space (lTOPNxMAXS) for selected assignments before writing them at sfragfct:
            // const uint mloc0 = ((qryndx * maxnsteps + ((ADDDELTA2ASSGADDR? lTOPNxMAXS: 0) + sfragfct)) *
                    // nTAuxWorkingMemoryVars) * ndbCstrs;
            const uint mloc2 = ((qryndx * maxnstepsmem2 + sfragfct) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
            float rr = (float)assgn[q];//chain index: int->float
            //NOTE: minus in the first position indicates swapped indices: query instead of reference!
            if(q == 0 && swapped) rr = -rr;
            wrkmem2[mloc2 + twm2aCCAssignment * ndbCstrs + dbstrndx] = rr;
        }
    }
}

// -------------------------------------------------------------------------
// Instantiations
//
#define INSTANTIATE_MakeChain2ChainAssignment( \
    tpWRITESCORE,tpWRITEASSG,tpPASS2,tpMAX_NCHAINS) \
    template __global__ void MakeChain2ChainAssignment \
        <tpWRITESCORE,tpWRITEASSG,tpPASS2,tpMAX_NCHAINS>( \
            const uint nqystrs, const uint ndbCstrs, const uint ndbCcpxs, \
            const uint maxnqrychains, const uint maxnrfnchains, \
            const uint maxnsteps, const uint maxnstepsmem2, \
            const float* __restrict__ tfmmem, \
            float* __restrict__ wrkmemaux, float* __restrict__ wrkmem2);

INSTANTIATE_MakeChain2ChainAssignment(true,true,false,0);
INSTANTIATE_MakeChain2ChainAssignment(false,true,false,0);
INSTANTIATE_MakeChain2ChainAssignment(true,false,true,0);

INSTANTIATE_MakeChain2ChainAssignment(true,false,true,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);
INSTANTIATE_MakeChain2ChainAssignment(false,true,false,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);

// =========================================================================
