/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __reformatter_cuh__
#define __reformatter_cuh__

// MakeCandidateComplexList: make list of candidate reference (database)
// complexes proceeding to stages of more detailed superposition search and
// refinement;
__global__ void MakeCandidateComplexList(
    const uint nqycpxs, const uint ndbCmpxs,
    const uint nqystrs, const uint ndbCstrs,
    const uint maxnsteps, const uint maxnstepsmem2,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    uint* __restrict__ globvarsbuf
);

// MakeCandidateComplexList2: make list of candidate reference (database)
// complexes proceeding to stages of more detailed superposition search and
// refinement;
// This corresponds to structure-based prefiltering and is based on single
// flags set for each complex pair once;
__global__ void MakeCandidateComplexList2(
    const uint nqycpxs, const uint ndbCcpxs,
    const uint nqystrs, const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    uint* __restrict__ globvarsbuf
);

// MakeDbCandidateList: make list of reference structure (database)
// candidates proceeding to more detailed stages of superposition search and
// refinement;
__global__ void MakeDbCandidateList(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    uint* __restrict__ globvarsbuf
);

// ReformatStructureDataPartStore: reformat a reference database chunk to
// include candidates proceeding to stages of more detailed superposition
// search and refinement; this part corresponds to storing data to
// secondary (temporary) location first;
__global__ void ReformatStructureDataPartStore(
    uint nqystrs,
    const uint ndbCstrs,
    const uint maxndbCposs,
    const uint maxnsteps,
    // const uint ndbCstrs2,
    // const uint ndbCposs2,
    // const uint dbstr1len2,
    const uint* __restrict__ globvarsbuf,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tfmmem,
    float* __restrict__ tmpdpdiagbuffers,
    const uint nqycpxs = 0,
    const uint ndbCcpxs = 0
);

// ReformatStructureDataPartLoad: reformat a reference database chunk to
// include candidates proceeding to stages of more detailed superposition
// search and refinement; this part corresponds to data load from secondary
// (temporary) location;
__global__ void ReformatStructureDataPartLoad(
    uint nqystrs,
    const uint maxndbCposs,
    const uint maxnsteps,
    const uint ndbCstrs2,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem,
    const uint nqycpxs = 0,
    const uint ndbCcpxs = 0
);

#endif//__reformatter_cuh__
