/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __coverage_cuh__
#define __coverage_cuh__

// CheckMaxCoverage: calculate maximum coverage between the queries and 
// reference structures and set the skip flag (convergence) if it is 
// below the threshold; the function actually resets convergence;
__global__ void CheckMaxCoverage(
    const float covthreshold,
    const int ntotqcpxs,
    const uint ndbCstrs,
    const uint maxnstepsmem2,
    float* __restrict__ wrkmem2
);

#endif//__coverage_cuh__
