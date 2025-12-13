/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "stagecnsts.cuh"
#include "covariance.cuh"
#include "covariance_plus.cuh"

// -------------------------------------------------------------------------
// Transform_CalcCCM_Score64: transform queries, calculate/reduce score for 
// obtained superpositions, and calculate cross-covariances for the 
// positions within calculated distances for multiple query-reference pairs
// simultaneously; save partial sums;
// NOTE: thread block is 1D and processes alignment along structure
// positions;
// NOTE: smem size used is half CUS1_TBINITSP_CCMCALC_ITRD_XDIM;
// ndbCstrs, total number of reference structures in the chunk;
// qrypos, starting query position;
// rfnpos, starting reference position;
// alnlen, maximum alignment length which corresponds to the minimum 
// length of the structures being compared;
// NOTE: memory pointers should be aligned!
// tfmmem, memory for transformation matrices;
// wrkmem, working memory, including the section of CC data;
// wrkmemaux, auxiliary working memory;
// 
template<int READCNST>
__global__ void Transform_CalcCCM_Score(
    uint ndbCstrs,
    int qrypos,
    int rfnpos,
    const float* __restrict__ tfmmem,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the reference serial number;
    // blockIdx.y is the query serial number;
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    constexpr int smidim = twmvEndOfCCDataExt+1;
    //__shared__ float ccmCache[smidim * CUS1_TBINITSP_CCMCALC_ITRD_XDIM];
    __shared__ float ccmCache[smidim * (CUS1_TBINITSP_CCMCALC_ITRD_XDIM>>1) + nTTranformMatrix];
    //pointer to transformation matrix;
    //NOTE: nTTranformMatrix < CUS1_TBINITSP_CCMCALC_ITRD_XDIM necessarily!
    //__shared__ float tfmCache[nTTranformMatrix];
    //NOTE: one smem block uses less registers
    float* tfmCache = ccmCache + smidim * (CUS1_TBINITSP_CCMCALC_ITRD_XDIM>>1);
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    float d02, d02s;

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(blockIdx.x, (int*)ccmCache);
        GetQueryLenDst(blockIdx.y, (int*)ccmCache + 2);
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlen = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylen = ((int*)ccmCache)[2]; qrydst = ((int*)ccmCache)[3];

    __syncthreads();


    if(PositionsOutofBounds(qrylen, dbstrlen, qrypos, rfnpos))
        //all threads in the block exit;
        //nalnposs is left as initialized
        return;


    if(READCNST == READCNST_CALC2) {
        if(threadIdx.x == 0) {
            //NOTE: reuse ccmCache[0] to contain twmvLastD02s. ccmCache[1] twmvNalnposs
            ccmCache[1] = 
                wrkmem[(blockIdx.y/*qryndx*/ * ndbCstrs + blockIdx.x/*dbstrndx*/) *
                    nTWorkingMemoryVars + twmvNalnposs];
        }
    }

    int maxnalnposs = GetNAlnPoss(qrylen, dbstrlen, qrypos, rfnpos);

    __syncthreads();

    d02 = GetD02(qrylen, dbstrlen);
    d02s = GetD02s(d02);
    if(READCNST == READCNST_CALC2) d02s += D02s_PROC_INC;

    if(READCNST == READCNST_CALC2) {
        int nalnposs = ccmCache[1];
        if(nalnposs == maxnalnposs)
            //all threads in the block exit;
            return;
        //cache will be overwritten below, sync
        __syncthreads();
    }

    //read transformation matrix for query-reference pair
    if(threadIdx.x < nTTranformMatrix)
        tfmCache[threadIdx.x] = 
            tfmmem[(blockIdx.y/*qryndx*/ * ndbCstrs + blockIdx.x/*dbstrndx*/) *
                nTTranformMatrix + threadIdx.x];

    __syncthreads();


    // chck_iter, iteration number of checking whether sufficient number of 
    // atoms are selected;
    for(int chck_iter = 0; chck_iter < N_ITER_DSTs_INCREASE; chck_iter++)
    {
        int nflds = chck_iter? smidim-1: smidim;

        if(threadIdx.x < (CUS1_TBINITSP_CCMCALC_ITRD_XDIM>>1)) {
            //initialize cache
            #pragma unroll
            for(int i = 0; i < nflds; i++)
                ccmCache[threadIdx.x * smidim + i] = 0.0f;
        }

        #pragma unroll
        for(int rpos = threadIdx.x; qrypos + rpos < qrylen && rfnpos + rpos < dbstrlen;
            rpos += blockDim.x)
        {
            //manually unroll along alignment
            UpdateOneAlnPosExtended<smidim>(
                nflds,
                d02, d02s,
                qrydst + qrypos + rpos,//query position
                dbstrdst + rfnpos + rpos,//reference position
                tfmCache,
                ccmCache
            );
            //no sync: every thread works in its own space (of ccmCache)
        }
        //sync now:
        __syncthreads();

        //unroll until reaching warpSize; 
        //(UpdateOneAlnPosExtended used two-fold unrolling in each iteration)
        #pragma unroll
        for(int xdim = CUS1_TBINITSP_CCMCALC_ITRD_XDIM>>2; xdim >= 32; xdim >>= 1) {
            if(threadIdx.x < xdim) {
                for(int i = 0; i < nflds; i++)
                    ccmCache[threadIdx.x * smidim + i] +=
                        ccmCache[(threadIdx.x + xdim) * smidim + i];
            }
            __syncthreads();
        }

        //unroll warp
        if(threadIdx.x < 32/*warpSize*/) {
            #pragma unroll
            for(int i = 0; i < nflds; i++) {
                float sum = ccmCache[threadIdx.x * smidim + i];
                sum = mywarpreducesum(sum);
                //write to the first data slot of SMEM
                if(threadIdx.x == 0) ccmCache[i] = sum;
            }
        }

        __syncthreads();

        if(ccmCache[twmvNalnposs] >= 3 || maxnalnposs <= 3)
            break;

        d02s += D02s_CHCK_INC;//original
        //d02s += D02s_CHCK_INC * (float)(1<<chck_iter);
        //d02s = 64.f + D02s_CHCK_INC * SQRD(chck_iter+1);


    }//for(chck_iter)

//     //TODO:REMOVE: the check performed during data copy
//     if(ccmCache[twmvNalnposs] == maxnalnposs) {
//         //write 0 so that the rotation matrix is not calculated for the 
//         //same cross-covarinace data
//         if(threadIdx.x == 0) ccmCache[twmvNalnposs] = 0.0f;
//         __syncthreads();
//     }


    //1st warp writes the result to global memory
    if(threadIdx.x < twmvEndOfCCDataExt)
        wrkmem[(blockIdx.y/*qryndx*/ * ndbCstrs + blockIdx.x/*dbstrndx*/) *
            nTWorkingMemoryVars + threadIdx.x] = ccmCache[threadIdx.x];


    //2nd warp writes the score to global memory
    if(threadIdx.x == 32/*warpSize*/) {
        wrkmemaux[(blockIdx.y/*qryndx*/ * nTAuxWorkingMemoryVars + tawmvScore) *
            ndbCstrs + blockIdx.x/*dbstrndx*/] =
            //ccmCache[twmvEndOfCCDataExt] is the reduced score
            ccmCache[twmvEndOfCCDataExt];
    }
}

// =========================================================================
// Instantiations
//
template
__global__ void Transform_CalcCCM_Score<READCNST_CALC>(
    uint ndbCstrs,
    int qrypos,
    int rfnpos,
    const float* __restrict__ tfmmem,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux);

template
__global__ void Transform_CalcCCM_Score<READCNST_CALC2>(
    uint ndbCstrs,
    int qrypos,
    int rfnpos,
    const float* __restrict__ tfmmem,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux);

// -------------------------------------------------------------------------
