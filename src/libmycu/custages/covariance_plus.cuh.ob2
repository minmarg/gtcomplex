/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_plus_h__
#define __covariance_plus_h__

// #include "libutil/macros.h"
// #include "libgenp/gproc/gproc.h"
// #include "libgenp/gdats/PM2DVectorFields.h"
// #include "libmycu/cucom/cucommon.h"
// #include "libmycu/culayout/cuconstant.cuh"
#include "covariance.cuh"
#include "transform.cuh"
#include "scoring.cuh"

// template constants for kernel Transform_CalcCCM_Score64:
// calculate constants:
#define READCNST_CALC 0
// calculate constants in the next pass:
#define READCNST_CALC2 1

// increment for d02s when the condition of # aln. positions is not met
#define D02s_CHCK_INC 0.5f
// increment for d02s when proceeding
#define D02s_PROC_INC 1.0f

// number of iterations for increasing distance threshold to ensure 
// sufficient number of atom pairs (>=3) for the calculation of rotation 
// matrices:
#define N_ITER_DSTs_INCREASE 10000

// transform queries, calculate/reduce score for obtained 
// superpositions, and calculate cross-covariances for positions 
// within calculated distances; save partial sums;
// READCNST, template parameter controlling how constants are 
// defined, see above;
template<int READCNST>
__global__ void Transform_CalcCCM_Score(
    uint ndbCstrs,
    int qrypos,
    int rfnpos,
    const float* __restrict__ tfmmem,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux
);

// -------------------------------------------------------------------------
// UpdateOneAlnPosExtended: update one position contributing to the 
// cross-covariance matrix between the query and reference structures 
// only if transformed query is within the given distance from reference;
// update score unconditionally;
// SMIDIM, template parameter: inner-most dimensions of the cache matrix;
// d02, d0 squared used for calculating score;
// d02s, d0 squared used for the inclusion of pairs in the alignment;
//
template<int SMIDIM>
__device__ __forceinline__
void UpdateOneAlnPosExtended(
    int nflds,
    float d02, float d02s,
    int qrypos,
    int rfnpos,
    const float* __restrict__ tfm,
    float* __restrict__ ccmCache)
{
    float qx = GetQueryCoord<pmv2DX>(qrypos);
    float qy = GetQueryCoord<pmv2DY>(qrypos);
    float qz = GetQueryCoord<pmv2DZ>(qrypos);

    float rx = GetDbStrCoord<pmv2DX>(rfnpos);
    float ry = GetDbStrCoord<pmv2DY>(rfnpos);
    float rz = GetDbStrCoord<pmv2DZ>(rfnpos);

    float dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);

    int tslot = (
        (threadIdx.x >=(CUS1_TBINITSP_CCMCALC_ITRD_XDIM>>1))? 
        (threadIdx.x - (CUS1_TBINITSP_CCMCALC_ITRD_XDIM>>1)): threadIdx.x) * SMIDIM;

    if(twmvEndOfCCDataExt < nflds)
        //calculate score unconditionally
        atomicAdd(&ccmCache[tslot + twmvEndOfCCDataExt], GetPairScore(d02, dst));

    if(d02s < dst)
        //distant positions do not contribute to cross-covariance
        return;

    atomicAdd(&ccmCache[tslot + twmvCCM_0_0], qx * rx);
    atomicAdd(&ccmCache[tslot + twmvCCM_0_1], qx * ry);
    atomicAdd(&ccmCache[tslot + twmvCCM_0_2], qx * rz);

    atomicAdd(&ccmCache[tslot + twmvCCM_1_0], qy * rx);
    atomicAdd(&ccmCache[tslot + twmvCCM_1_1], qy * ry);
    atomicAdd(&ccmCache[tslot + twmvCCM_1_2], qy * rz);

    atomicAdd(&ccmCache[tslot + twmvCCM_2_0], qz * rx);
    atomicAdd(&ccmCache[tslot + twmvCCM_2_1], qz * ry);
    atomicAdd(&ccmCache[tslot + twmvCCM_2_2], qz * rz);

    atomicAdd(&ccmCache[tslot + twmvCVq_0], qx);
    atomicAdd(&ccmCache[tslot + twmvCVq_1], qy);
    atomicAdd(&ccmCache[tslot + twmvCVq_2], qz);

    atomicAdd(&ccmCache[tslot + twmvCVr_0], rx);
    atomicAdd(&ccmCache[tslot + twmvCVr_1], ry);
    atomicAdd(&ccmCache[tslot + twmvCVr_2], rz);

    //update the number of positions
    atomicAdd(&ccmCache[tslot + twmvNalnposs], 1.0f);
}

#endif//__covariance_plus_h__
