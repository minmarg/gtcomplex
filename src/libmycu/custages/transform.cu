/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/fields.cuh"
#include "transform.cuh"


// -------------------------------------------------------------------------
// RevertTfmMatrices: revert transformation matrices;
// ndbCstrs, total number of reference structures in the chunk;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// tfmmem, memory section for transformation matrices;
// NOTE: unroll by a factor of CUS1_TBINITSP_TFMINIT_XFCT;
// 
__global__ void RevertTfmMatrices(
    const uint ndbCstrs,
    const float* __restrict__ /* wrkmemaux */,
    float* __restrict__ tfmmem,
    const uint ndbCchns)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * CUS1_TBINITSP_TFMINIT_XFCT;
    uint qryndx = blockIdx.y;//query serial number
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;
    __shared__ float tfmCache[CUS1_TBINITSP_TFMINIT_XFCT * nTTranformMatrix];
    uint ndx = 0;//relative reference index < CUS1_TBINITSP_TFMINIT_XFCT
    uint mloc = (qryndx * ndbCtrgs + dbstrndx) * nTTranformMatrix;

    #pragma unroll
    for(int i = 1; i < CUS1_TBINITSP_TFMINIT_XFCT; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    if(dbstrndx + ndx < ndbCstrs) {
        //READ
        if(threadIdx.x < nTTranformMatrix * (ndx+1))
            tfmCache[threadIdx.x] = tfmmem[mloc + threadIdx.x];
    }

    __syncthreads();

    if(threadIdx.x < CUS1_TBINITSP_TFMINIT_XFCT) {
        //only CUS1_TBINITSP_TFMINIT_XFCT threads/tfms in charge
        InvertRotMtx(&tfmCache[threadIdx.x * nTTranformMatrix]);
        InvertTrlVec(&tfmCache[threadIdx.x * nTTranformMatrix]);
    }

    __syncthreads();

    if(dbstrndx + ndx < ndbCstrs) {
        //WRITE
        if(threadIdx.x < nTTranformMatrix * (ndx+1))
            tfmmem[mloc + threadIdx.x] = tfmCache[threadIdx.x];
    }
}

// -------------------------------------------------------------------------
// InitGTfmMatrices: device code for initializing grand best transformation 
// matrices between query and reference structures;
// nqycpxs, #query complexes in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// NOTE: memory pointers should be aligned!
// tfmmem, memory section for transformation matrices;
// NOTE: unroll by a factor of CUS1_TBINITSP_TFMINIT_XFCT: this number of 
// structures initialized by a thread block
// 
__device__ __forceinline__
void InitGTfmMatrices(
    const uint nqycpxs,
    const uint ndbCstrs,
    float* __restrict__ tfmmem)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * CUS1_TBINITSP_TFMINIT_XFCT;
    uint qryndx = blockIdx.y;//query serial number
    uint ndx = 0;//relative reference index < CUS1_TBINITSP_TFMINIT_XFCT

    if(nqycpxs <= qryndx) return;//no sync: exit

    #pragma unroll
    for(int i = 1; i < CUS1_TBINITSP_TFMINIT_XFCT; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    if(ndbCstrs <= dbstrndx + ndx) return;//no sync: exit


    if(threadIdx.x < nTTranformMatrix * (ndx+1))
        tfmmem[
            (qryndx * ndbCstrs + dbstrndx + ndx) * nTTranformMatrix + 
            threadIdx.x - nTTranformMatrix * ndx] = 0.0f;

    //assign the diagonal of ration matrix to 1s
    if(threadIdx.x == tfmmRot_0_0 + nTTranformMatrix * ndx ||
       threadIdx.x == tfmmRot_1_1 + nTTranformMatrix * ndx ||
       threadIdx.x == tfmmRot_2_2 + nTTranformMatrix * ndx)
        tfmmem[
            (qryndx * ndbCstrs + dbstrndx + ndx) * nTTranformMatrix + 
            threadIdx.x - nTTranformMatrix * ndx] = 1.0f;
}

// -------------------------------------------------------------------------
// InitAlnData: device code for initializing alignment data for query and 
// reference pairs;
// ndbCstrs, total number of reference structures in the chunk;
// NOTE: memory pointers should be aligned!
// alndatamem, memory for full alignment information, including scores;
// NOTE: unroll by a factor of CUS1_TBINITSP_TFMINIT_XFCT: this number of 
// structures initialized by a thread block
// 
__device__ __forceinline__
void InitAlnData(
    uint ndbCstrs,
    float* __restrict__ alndatamem)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    const uint dbstrndx = blockIdx.x * CUS1_TBINITSP_TFMINIT_XFCT;
    const uint qryndx = blockIdx.y;//query serial number

    if(ndbCstrs <= dbstrndx) return;

    for(uint blk = 0;
        blk + threadIdx.x < nTDP2OutputAlnData * (ndbCstrs - dbstrndx) &&
        blk + threadIdx.x < nTDP2OutputAlnData * CUS1_TBINITSP_TFMINIT_XFCT;
        blk += blockDim.x)
    {
        uint mloc = (qryndx * ndbCstrs + dbstrndx) * nTDP2OutputAlnData + blk + threadIdx.x;
        alndatamem[mloc] = 0.0f;
        #pragma unroll
        for(int i = 0; i < CUS1_TBINITSP_TFMINIT_XFCT; i++)
            if(blk + threadIdx.x == dp2oadRMSD + nTDP2OutputAlnData * i)
                alndatamem[mloc] = 9999.9f;
    }
}

// -------------------------------------------------------------------------
// InitGTfmMatricesAndAData: device code for initializing grand best
// transformation matrices and associated alignment data;
// ndbCstrs, total number of reference structures in the chunk;
// NOTE: memory pointers should be aligned!
// tfmmem, memory section for transformation matrices;
// alndatamem, memory for full alignment information, including scores;
// NOTE: unroll by a factor of CUS1_TBINITSP_TFMINIT_XFCT: this number of 
// structures initialized by a thread block
// 
__global__ void InitGTfmMatricesAndAData(
    const uint nqycpxs,
    const uint ndbCstrs,
    float* __restrict__ tfmmem,
    float* __restrict__ alndatamem)
{
    InitGTfmMatrices(nqycpxs, ndbCstrs, tfmmem);
    InitAlnData(ndbCstrs, alndatamem);
}



// -------------------------------------------------------------------------
// InitTfmMatrices: device code for initializing transformation matrices 
// between query and reference structures;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// minfraglen, minimum fragment length for which maxnsteps is calculated;
// sfragstep, step with which fragments progress;
// checkfragos, check whether calculated fragment position is within boundaries;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, memory section for transformation matrices;
// NOTE: unroll by a factor of CUS1_TBINITSP_TFMINIT_XFCT: this number of 
// structures initialized by a thread block
// 
__global__ void InitTfmMatrices(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint minfraglen,
    const int sfragstep,
    const bool checkfragos,
    float* __restrict__ wrkmemtmibest)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * CUS1_TBINITSP_TFMINIT_XFCT;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    constexpr int qslot = CUS1_TBINITSP_TFMINIT_XFCT;//query slot in wrtCache
    __shared__ int wrtCache[CUS1_TBINITSP_TFMINIT_XFCT+1];
    uint ndx = 0;//relative reference index < CUS1_TBINITSP_TFMINIT_XFCT

    if(threadIdx.x < CUS1_TBINITSP_TFMINIT_XFCT) {
        wrtCache[threadIdx.x] = 
            ((dbstrndx + threadIdx.x < ndbCstrs) && (checkfragos == false));
    }

    #pragma unroll
    for(int i = 1; i < CUS1_TBINITSP_TFMINIT_XFCT; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    if(checkfragos) {
        uint sfragpos = sfragfct * sfragstep;//fragment position

        if(threadIdx.x < CUS1_TBINITSP_TFMINIT_XFCT && dbstrndx + threadIdx.x < ndbCstrs) {
            wrtCache[threadIdx.x] = GetDbStrLength(dbstrndx+threadIdx.x);
            if(threadIdx.x == 0) wrtCache[qslot] = GetQueryLength(qryndx);
        }

        __syncthreads();

        if(threadIdx.x < CUS1_TBINITSP_TFMINIT_XFCT && dbstrndx + threadIdx.x < ndbCstrs) {
            uint maxalnlen = myhdmin(wrtCache[threadIdx.x], wrtCache[qslot]);
            wrtCache[threadIdx.x] = FragPosWithinAlnBoundaries(maxalnlen, sfragstep, sfragpos, minfraglen);
        }
    }

    __syncthreads();

    if(wrtCache[ndx] && threadIdx.x < nTTranformMatrix * (ndx+1)) {
        uint memloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        wrkmemtmibest[memloc + threadIdx.x] = 0.0f;
    }

    //assign the diagonal of ration matrix to 1s
    if(wrtCache[ndx] && (
       threadIdx.x == tfmmRot_0_0 + nTTranformMatrix * ndx ||
       threadIdx.x == tfmmRot_1_1 + nTTranformMatrix * ndx ||
       threadIdx.x == tfmmRot_2_2 + nTTranformMatrix * ndx)) {
        uint memloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        wrkmemtmibest[memloc + threadIdx.x] = 1.0f;
    }
}



// =========================================================================
// CalcTfmMatrices_DynamicOrientation: calculate tranformation matrices wrt
// query or reference structure, whichever is longer;
// NOTE: thread block is 1D and calculates multiple matrices simulatneously;
// NOTE: works only for CUS1_TBSP_TFM_N == 32 because of warp sync!
// 
template<int COMPLEX>
__global__
void CalcTfmMatrices_DynamicOrientation(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmem2)
{
    // blockIdx.x is the reference serial number;
    // blockIdx.y is the query serial number (index in the chunk);
    // blockIdx.z is the fragment factor;
    int absndx = blockIdx.x * CUS1_TBSP_TFM_N + threadIdx.x;
    int qryndx = blockIdx.y;//query index in the chunk
    // int sfragfct = blockIdx.z;//fragment factor
    int qrylen, dbstrlen, dummyaddr;//query and reference length

    if(ndbCstrs <= absndx)
        //no sync below: each thread runs independently: exit
        return;

    if(COMPLEX) {
        if(threadIdx.x == 0) GetQueryComplexAddressLength(qryndx, dummyaddr, qrylen);
        GetDbComplexAddressLength(absndx, dummyaddr, dbstrlen);
    } else {
        if(threadIdx.x == 0) qrylen = GetQueryLength(qryndx);
        dbstrlen = GetDbStrLength(absndx);
    }

    qrylen = __shfl_sync(0xffffffff, qrylen, 0/*srcLane*/);

    //stable and symmetric:
    if(qrylen < dbstrlen)
        //original
        CalcTfmMatricesHelper<false/* REVERSE */>(ndbCstrs, maxnsteps, wrkmem2);
    else
        CalcTfmMatricesHelper<true/* REVERSE */>(ndbCstrs, maxnsteps, wrkmem2);
}

// -------------------------------------------------------------------------
// Instantiations
//
#define INSTANTIATE_CalcTfmMatrices_DynamicOrientation(COMPLEX) \
    template __global__ void CalcTfmMatrices_DynamicOrientation<COMPLEX>( \
        const uint ndbCstrs, const uint maxnsteps, \
        float* __restrict__ wrkmem2);

INSTANTIATE_CalcTfmMatrices_DynamicOrientation(0);
INSTANTIATE_CalcTfmMatrices_DynamicOrientation(1);

// -------------------------------------------------------------------------
// CalcTfmMatrices: calculate tranformation matrices;
// NOTE: thread block is 1D and calculates multiple matrices simulatneously;
// REVERSE, template parameter, change places of query and reference sums 
// so that transformation matrices are calculated wrt queries;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmem2, working memory, including the section of CC data;
// NOTE: keep #registers <64 when CUS1_TBSP_TFM_N == 32!
// NOTE: Based on the original Kabsch algorithm:
template<bool REVERSE, bool TFM_DINV, int COMPLEX>
__global__
void CalcTfmMatrices(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmem2)
{
    if(! TFM_DINV) {
        CalcTfmMatricesHelper<REVERSE>(ndbCstrs, maxnsteps, wrkmem2);
        return;
    }

    // blockIdx.x is the reference serial number;
    // blockIdx.y is the query serial number (index in the chunk);
    // blockIdx.z is the fragment factor;
    int absndx = blockIdx.x * CUS1_TBSP_TFM_N + threadIdx.x;
    int qryndx = blockIdx.y;//query index in the chunk
    // int sfragfct = blockIdx.z;//fragment factor
    int qrylen, dbstrlen, dummyaddr;//query and reference length

    if(ndbCstrs <= absndx)
        //no sync below: each thread runs independently: exit
        return;

    if(COMPLEX) {
        if(threadIdx.x == 0) GetQueryComplexAddressLength(qryndx, dummyaddr, qrylen);
        GetDbComplexAddressLength(absndx, dummyaddr, dbstrlen);
    } else {
        if(threadIdx.x == 0) qrylen = GetQueryLength(qryndx);
        dbstrlen = GetDbStrLength(absndx);
    }

    qrylen = __shfl_sync(0xffffffff, qrylen, 0/*srcLane*/);

    //stable and symmetric:
    if(qrylen < dbstrlen)
        CalcTfmMatricesHelper<!REVERSE, true/* REVERT_BACK */>(ndbCstrs, maxnsteps, wrkmem2);
    else
        CalcTfmMatricesHelper<REVERSE>(ndbCstrs, maxnsteps, wrkmem2);
}

// -------------------------------------------------------------------------
// Instantiations:
// 
#define INSTANTIATE_CalcTfmMatrices(tpREVERSE,tpTFM_DINV,tpCOMPLEX) \
    template __global__ void CalcTfmMatrices<tpREVERSE,tpTFM_DINV,tpCOMPLEX>( \
        const uint ndbCstrs, const uint maxnsteps, \
        float* __restrict__ wrkmem2);

INSTANTIATE_CalcTfmMatrices(TFMTX_REVERSE_FALSE,false,0);
INSTANTIATE_CalcTfmMatrices(TFMTX_REVERSE_FALSE,false,1);
INSTANTIATE_CalcTfmMatrices(TFMTX_REVERSE_TRUE,false,0);
INSTANTIATE_CalcTfmMatrices(TFMTX_REVERSE_TRUE,false,1);

INSTANTIATE_CalcTfmMatrices(TFMTX_REVERSE_FALSE,true,0);
INSTANTIATE_CalcTfmMatrices(TFMTX_REVERSE_FALSE,true,1);
INSTANTIATE_CalcTfmMatrices(TFMTX_REVERSE_TRUE,true,0);
INSTANTIATE_CalcTfmMatrices(TFMTX_REVERSE_TRUE,true,1);

// -------------------------------------------------------------------------
