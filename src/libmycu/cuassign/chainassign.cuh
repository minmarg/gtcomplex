/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __chainassign_cuh__
#define __chainassign_cuh__

#include "libutil/cnsts.h"
#include "libgenp/gproc/gproc.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/cutemplates.h"
#include "libmycu/cuproc/cuprocconf.h"

enum {//kernel constants:
    lInf = 999999,
    lXDIM = CUAP_MAKEASSIGNMENT_XDIM,
    //#warps within lXDIM; NOTE: #warps should be a power of two!
    lXDIMnwarps = (lXDIM >> llog2warpsize),//#warps per lXDIM
    lTFMsperWarp = lWarpsize / nTTranformMatrix//#tfm matrices per warp
};

enum {//orientation vector dimensions:
    oVecU0, oVecU1, oVecU2,//rotation axis
    oVeccost, //angle cosine
    oVecTrl0, oVecTrl1, oVecTrl2,//translation
    noVec
};

// -------------------------------------------------------------------------
// MakeChain2ChainAssignment: make complex chain to chain assignment
// based on TM-scores only using the Hungarian algorithm;
template<
    bool WRITESCORE,
    bool WRITEASSG,
    bool PASS2,
    int MAX_NCHAINS = 0>
__global__
void MakeChain2ChainAssignment(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCcpxs,
    const uint maxnqrychains,
    const uint maxnrfnchains,
    const uint maxnsteps,
    const uint maxnstepsmem2,
    const float* __restrict__ tfmmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2
);

// GetSmemSizeForMakeChain2ChainAssignment: get the size of dynamically 
// allocted smem for the MakeChain2ChainAssignment kernel;
// maxnqrychains, max #query chains across the chunk;
// maxnrfnchains, max #reference chains across the chunk;
inline
uint GetSmemSizeForMakeChain2ChainAssignment(
    const uint maxnqrychains, const uint maxnrfnchains)
{
    const uint maxn1chains = myhdmin(maxnqrychains, maxnrfnchains);
    const uint maxn2chains1 = myhdmax(maxnqrychains, maxnrfnchains) + 1;
    return
        (nTTranformMatrix + //mean orientation
         nTTranformMatrix * lTFMsperWarp * lXDIMnwarps + //chain orientation
         CUAP_MAKEASSIGNMENT_XDIM + //divergence factors
         lXDIMnwarps + //temp. buffer
         maxn1chains + //source potentials
         maxn2chains1 + //target potentials
         maxn2chains1 //min reduced cost over edges from Z (source subset) to chain c2
        ) * sizeof(float) + 
        (
         maxn2chains1 + //chains of complex 1 assigned to complex 2's chains
         maxn2chains1 + //previous (complex 2's) chain on alternating path
         maxn2chains1 + //flags of whether (complex 2's) chain is in Z
         maxn1chains + //final assignment
         lXDIMnwarps //temp. buffer
        ) * sizeof(int);
}



// -------------------------------------------------------------------------
// Tfm2OVec: construct orientation vector in-place from transformation 
// matrix;
// orientsCh, cache containing transformation matrix on input;
// tid, computing thread id;
//
__device__ __forceinline__
void Tfm2OVec(float* orientsCh, int tid)
{
    if((int)threadIdx.x == tid) {
        //rotation axis (u0,u1,u2); its norm is 2sint;
        //NOTE: don't scale by 2sint because sint is always non-negative
        //NOTE: (arccos range [0,pi] and hence sint=sqrt(1-(cost)^2))
        float u0 = orientsCh[tfmmRot_2_1] - orientsCh[tfmmRot_1_2];
        float u1 = orientsCh[tfmmRot_0_2] - orientsCh[tfmmRot_2_0];
        float u2 = orientsCh[tfmmRot_1_0] - orientsCh[tfmmRot_0_1];
        //cos and sin of rotation angle:
        float cost = (orientsCh[tfmmRot_0_0] + orientsCh[tfmmRot_1_1] + orientsCh[tfmmRot_2_2] - 1.f) * 0.5f;
        //write values back to orientsCh:
        orientsCh[oVecU0] = u0; orientsCh[oVecU1] = u1; orientsCh[oVecU2] = u2;
        orientsCh[oVeccost] = cost;
        orientsCh[oVecTrl0] = orientsCh[tfmmTrl_0];
        orientsCh[oVecTrl1] = orientsCh[tfmmTrl_1];
        orientsCh[oVecTrl2] = orientsCh[tfmmTrl_2];
    }
}

// -------------------------------------------------------------------------
// GetDivergence: calculate divergence of orientation vector orientsCh from
// the mean vector meanorientCh and write the result to convergCh;
// orientsCh, cache containing transformation matrix on input;
// part, flag of thread participation;
// tid, thread id computing 0th slot;
//
__device__ __forceinline__
void GetDivergence(
    const bool part,
    const float* __restrict__ meanorientCh,
    float* __restrict__ orientsCh,
    float* __restrict__ convergCh,
    int tid)
{
    const int n = part? ((int)threadIdx.x - tid): -1;

    if(0 <= n && n < noVec)
        orientsCh[n] = fabsf(meanorientCh[n] - orientsCh[n]);

    __syncthreads();

    if(n == 0) {
        // float avg = (
        //     (orientsCh[oVecU0] + orientsCh[oVecU1] + orientsCh[oVecU2]) * oneTHIRDf +
        //     (orientsCh[oVeccost]) +
        //     (orientsCh[oVecTrl0] + orientsCh[oVecTrl1] + orientsCh[oVecTrl2]) * oneTHIRDf
        // ) * oneTHIRDf;
        float avg =
            (orientsCh[oVecU0] + orientsCh[oVecU1] + orientsCh[oVecU2]) * oneNINTHf +
            (orientsCh[oVeccost]) * oneTHIRDf +
            (orientsCh[oVecTrl0] + orientsCh[oVecTrl1] + orientsCh[oVecTrl2]) * oneNINTHf;
        // float avg = sqrtf(myhdsqrdv(orientsCh[oVecU0]) + myhdsqrdv(orientsCh[oVecU1]) +
        //                   myhdsqrdv(orientsCh[oVecU2])) + orientsCh[oVeccost];
        *convergCh = __expf(-avg);
    }
}



// -------------------------------------------------------------------------
// CalculateMeanOrientationVectorPass2: calculate mean orientation 
// vector in the second pass of chain-to-chain assignment;
// lwarpsize, template parameter, warp size;
// llog2warpsize, template parameter, log2(lwarpsize);
// lXDIM, template parameter, thread block's x-dimension;
// lXDIMnwarps, template parameter, #warps per lXDIM;
// lTFMsperWarp, template parameter, #tfm matrices per warp;
// Q, number of query (reference) chains;
// swapped, flag of whether Q represents #reference chains instead of query;
// qrycpxDstN, #chains up to this query complex;
// dbcpxDstN, #chains up to this reference complex;
// nqystrs, total number of queries in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// tfmmem, transformation matrix address space;
// wrkmemaux, auxiliary working memory;
// assgn, assignment obtained in the first pass;
// orientsCh, cache of chain orientations;
// meanorientCh, cache for mean orientation; 
//
template<
    int lwarpsize, int llog2warpsize, int lXDIM,
    int lXDIMnwarps, int lTFMsperWarp>
__device__ __forceinline__
void CalculateMeanOrientationVectorPass2(
    const int Q, const int swapped,
    const int qrycpxDstN, const int dbcpxDstN, 
    const int nqystrs, const int ndbCstrs,
    const int maxnsteps,
    const float* __restrict__ tfmmem,
    const float* __restrict__ wrkmemaux,
    int* __restrict__ assgn,
    float* __restrict__ orientsCh,
    float* __restrict__ meanorientCh)
{
    //read previous assignment from gmem:
    //NOTE: Q < dbcpxN always
    for(int q = threadIdx.x; q < Q; q += lXDIM) {
        const int qryndx = qrycpxDstN;//query complex's 1st chain index
        const int dbstrndx = dbcpxDstN + q;//reference complex's 1st chain index + running index
        const int mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        assgn[q] = (int)wrkmemaux[mloc0 + tawmvInitialBest * ndbCstrs + dbstrndx];//chain index: float->int
    }
    __syncthreads();

    const int warpn = threadIdx.x >> (llog2warpsize);//warp number
    const int lid = threadIdx.x & (lwarpsize - 1);//lane id
    int tfmn = 0;//tfm index within a warp
    for(int t = nTTranformMatrix-1; t < lwarpsize && t < lid; t += nTTranformMatrix, tfmn++);
    //chain index relative to threadIdx:
    int qt = (tfmn + warpn * lTFMsperWarp);

    for(int q = 0; q < Q; q += lTFMsperWarp * lXDIMnwarps) {
        int qq = q + qt;
        int rr = (tfmn < lTFMsperWarp && qq < Q)? assgn[qq]: -1;
        if(qq == 0 && rr < 0) rr = -rr;//swap flag set above
        const int qryndx = qrycpxDstN + (swapped? rr: qq);//query complex's chain index
        const int dbstrndx = dbcpxDstN + (swapped? qq: rr);//reference complex's 1st chain index + running index
        const bool participating =
            (0 <= qq && 0 <= rr && qryndx < nqystrs && dbstrndx < ndbCstrs);
        if(participating) {
            //NOTE: condition also implies tfmn < lTFMsperWarp!
            //NOTE: dont sync: incomplete warps participate!
            //read (lTFMsperWarp x lXDIMnwarps) globally best transformation matrices at once:
            int mloc0 = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
            int xi = lid - tfmn * nTTranformMatrix;
            orientsCh[qt * nTTranformMatrix + xi] = tfmmem[mloc0 + xi];
        }
        __syncthreads();
        //convert transformation matrices to orientation vectors:
        if(participating)
            //NOTE: condition also implies tfmn < lTFMsperWarp!
            Tfm2OVec(orientsCh + qt * nTTranformMatrix, warpn * lwarpsize + tfmn * nTTranformMatrix);
        __syncthreads();
        //sum orientation vectors for average:
        if(participating) {
            int xi = lid - tfmn * nTTranformMatrix;
            atomicAdd(&meanorientCh[xi], orientsCh[qt * nTTranformMatrix + xi]);
        }
    }
    __syncthreads();
    //calculate mean orientation vector:
    if(threadIdx.x < noVec && 0 < Q)
        meanorientCh[threadIdx.x] = __fdividef(meanorientCh[threadIdx.x], (float)Q);
}

// -------------------------------------------------------------------------
// GetDivOrientationVectorsPass2: cache orientation vectors in the second 
// pass of chain-to-chain assignment and calculate their divergence from the
// mean vector;
// lwarpsize, template parameter, warp size;
// llog2warpsize, template parameter, log2(lwarpsize);
// lXDIM, template parameter, thread block's x-dimension;
// lXDIMnwarps, template parameter, #warps per lXDIM;
// lTFMsperWarp, template parameter, #tfm matrices per warp;
// qq, query chain index;
// r0, starting reference chain index;
// R, number of reference (query) chains;
// swapped, flag of whether R represents #query chains instead of reference;
// qrycpxDstN, #chains up to this query complex;
// dbcpxDstN, #chains up to this reference complex;
// nqystrs, total number of queries in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// tfmmem, transformation matrix address space;
// in_Z, flags of whether (complex 2's) chain is in Z;
// meanorientCh, cache of the mean orientation;
// orientsCh, cache of chain orientations;
// convergCh, cache for divergence/convergence values;
//
template<
    int lwarpsize, int llog2warpsize, int lXDIM,
    int lXDIMnwarps, int lTFMsperWarp>
__device__ __forceinline__
void GetDivOrientationVectorsPass2(
    const int qq, const int r0, const int R, const int swapped,
    const int qrycpxDstN, const int dbcpxDstN, 
    const int nqystrs, const int ndbCstrs,
    const float* __restrict__ tfmmem,
    const int* __restrict__ in_Z,
    const float* __restrict__ meanorientCh,
    float* __restrict__ orientsCh,
    float* __restrict__ convergCh)
{
    const int warpn = threadIdx.x >> (llog2warpsize);//warp number
    const int lid = threadIdx.x & (lwarpsize - 1);//lane id
    int tfmn = 0;//tfm index within a warp
    for(int t = nTTranformMatrix-1; t < lwarpsize && t < lid; t += nTTranformMatrix, tfmn++);
    //chain index relative to threadIdx:
    int rt = (tfmn + warpn * lTFMsperWarp);

    for(int rl = 0; r0 + rl < R && rl < lXDIM; rl += lTFMsperWarp * lXDIMnwarps) {
        int rr = r0 + rl + rt;
        //only threads processing full tfms participate:
        if(lTFMsperWarp <= tfmn || R <= rr || in_Z[rr] != 0) rr = -1;
        const int qryndx = qrycpxDstN + (swapped? rr: qq);//query complex's chain index
        const int dbstrndx = dbcpxDstN + (swapped? qq: rr);//reference complex's 1st chain index + running index
        const bool participating =
            (0 <= rr && qryndx < nqystrs && dbstrndx < ndbCstrs);
        if(participating) {
            //NOTE: condition also implies tfmn < lTFMsperWarp!
            //NOTE: dont sync: incomplete warps participate!
            //read (lTFMsperWarp x lXDIMnwarps) globally best transformation matrices at once:
            int mloc0 = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
            int xi = lid - tfmn * nTTranformMatrix;
            orientsCh[rt * nTTranformMatrix + xi] = tfmmem[mloc0 + xi];
        }
        __syncthreads();
        //convert transformation matrices to orientation vectors:
        if(participating)
            //NOTE: condition also implies tfmn < lTFMsperWarp!
            Tfm2OVec(&orientsCh[rt * nTTranformMatrix], warpn * lwarpsize + tfmn * nTTranformMatrix);
        __syncthreads();
        //calculate divergence from the mean vector
        //NOTE: GetDivergence syncs and requires thread participation condition!!
        GetDivergence(
            participating,
            meanorientCh,
            &orientsCh[rt * nTTranformMatrix],
            &convergCh[rl + rt],
            warpn * lwarpsize + tfmn * nTTranformMatrix);
        __syncthreads();
    }
}

// -------------------------------------------------------------------------

#endif//__chainassign_cuh__
