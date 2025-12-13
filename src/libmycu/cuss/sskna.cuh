/***************************************************************************
 *   Copyright (C) 2021-2025 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __sskna_cuh__
#define __sskna_cuh__

#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/fields.cuh"
#include "sskbase.cuh"

// =========================================================================
enum{
        lTMPBUFFNDX_POSIT = 0,//pairig position with least deviation
        lTMPBUFFNDX_DEVIA = 1,//minimum deviation
        lTMPBUFFNDX_MUTEX = 2//positional mutex
};
// =========================================================================
// CalcNASecStrs: calculate secondary structures for all query OR reference 
// nucleic acid structures in the chunk;
template<int STRUCTS>
__global__ void CalcSecStrs_NASS(
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers
);

// Initialize_NASS: initialize temporary memory buffer for calculating
// nucleic acid secondary structures
__global__ void Initialize_NASS(
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers
);

// CalcDistances_NASS: calculate relevant pairwise distances between
// residues for all query OR reference nucleic acid structures in the chunk;
template<int STRUCTS>
__global__ void CalcDistances_NASS(
    const int atomtype,
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers
);

// CalcDistances_NASS_CC7: calculate relevant pairwise distances between
// residues for all query OR reference nucleic acid structures in the chunk;
// NOTE: version for compute capability starting with No. 7;
template<int STRUCTS>
__global__ void CalcDistances_NASS_CC7(
    const int atomtype,
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers
);

// =========================================================================
// -------------------------------------------------------------------------
// SSKNAGetPairingCondition: get the condition for pairing bases;
// rsdy, rsdx, two nucleic acid bases;
//
__device__ __forceinline__
bool SSKNAGetPairingCondition(const char rsdy, const char rsdx)
{
    return
        ((rsdy == 'T' || rsdy == 'U') && (rsdx == 'A')) ||
        ((rsdy == 'A') && (rsdx == 'T' || rsdx == 'U')) ||
        ((rsdy == 'G') && (rsdx == 'C' || rsdx == 'U')) ||
        ((rsdy == 'C' || rsdy == 'U') && (rsdx == 'G'));
}

// -------------------------------------------------------------------------
// SSKNAGetDstDeviation: get distance deviation from the statistical 
// average of distances between paired nucleic acid atoms of given type;
// atype, nucleic acid atom type;
// dst, observed distance between atoms of given type;
//
__device__ __forceinline__
float SSKNAGetDstDeviation(const int atype, float dst)
{
    float dev;
    const float devl = 9999.9f;
    if(atype == gtnaatC3p) {dev = fabsf(dst - gtnaatC3p_LUB_AVG); return (gtnaatC3p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatC4p) {dev = fabsf(dst - gtnaatC4p_LUB_AVG); return (gtnaatC4p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatC5p) {dev = fabsf(dst - gtnaatC5p_LUB_AVG); return (gtnaatC5p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatO3p) {dev = fabsf(dst - gtnaatO3p_LUB_AVG); return (gtnaatO3p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatO5p) {dev = fabsf(dst - gtnaatO5p_LUB_AVG); return (gtnaatO5p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatP) {dev = fabsf(dst - gtnaatP_LUB_AVG); return (gtnaatP_LUB_DLT < dev)? devl: dev;}
    return devl;
}

// =========================================================================

#endif//__sskna_cuh__
