/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __scoring_cuh__
#define __scoring_cuh__

#include "libgenp/gproc/gproc.h"
#include "libmymp/mpstages/scoringbase.h"
#include "covariance.cuh"
#include "transform.cuh"


// SetCurrentFragSpecs: set the specifications of the current fragment 
// under process;
__global__ void SetCurrentFragSpecs(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragndx,
    float* __restrict__ wrkmemaux
);

// SetConvergenceForUnmatchedTypes: set convergence flag for chain pairs of
// inconsistent molecule types;
__global__ void SetConvergenceForUnmatchedTypes(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

// SetConvergenceForUnmatchedTypesComplex: set convergence flag for complex
// pairs of inconsistent by type;
__global__ void SetConvergenceForUnmatchedTypesComplex(
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

// SetLowScoreConvergenceFlag: set the appropriate convergence flag for 
// the pairs for which the score is below the threshold;
template<bool COMPLEX = false>
__global__ void SetLowScoreConvergenceFlag(
    const float scorethld,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns = 0
);

// SetLowScoreConvergenceFlagComplexInitial: set the convergence flag for 
// complex pairs for which the initial score originating from individual
// chain processing is below the threshold;
template<int MAX_NCHAINS = 0>
__global__ void SetLowScoreConvergenceFlagComplexInitial(
    const float scorethld,
    const uint nqrycpxs,
    const uint ndbCcpxs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    const uint ndbCchns
);

// SetLowScoreConvergenceFlagComplexIntermediate: set the convergence 
// flag for complex pairs for which intermediate provisional score is 
// below the threshold;
template<int MIN_NCHAINS = 0>
__global__ void SetLowScoreConvergenceFlagComplexIntermediate(
    const float scorethld,
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint nconfigs,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2
);

// InitWrkmem2: initialize the slot of the wrkmem2 memory used
// temporarily for convergence;
__global__ void InitWrkmem2(
    const uint ndbCstrs,
    const uint maxnstepsmem2,
    float* __restrict__ wrkmem2,
    const float value
);

// InitScores: initialize best and current scores to 0;
// INITOPT, template parameter controlling which scores are to be 
// initialized, see above;
template<int INITOPT>
__global__ void InitScores(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint minfraglen,
    const bool checkfragos,
    float* __restrict__ wrkmemaux
);

// SaveLastScore0: save last calculated score;
__global__ void SaveLastScore0(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns = 0
);

// SaveBestScore: save best score along with query and reference 
// positions for which this score is observed;
__global__ void SaveBestScore(
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    float* __restrict__ wrkmemaux
);

// SaveBestScoreAmongBests: save best score along with query and reference 
// positions by considering all partial best scores calculated over all 
// fragment factors; write it to the location of fragment factor 0;
__global__ void SaveBestScoreAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    float* __restrict__ wrkmemaux
);

// CheckScoreConvergence: check whether the score of the last two 
// procedures converged, i.e., the difference is small;
__global__ void CheckScoreConvergence(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns = 0
);

// CheckScoreProgression: check whether the difference between the maximum 
// score and the score of the last procedure is large enough; if not, set 
// the appropriate convergence flag;
__global__ void CheckScoreProgression(
    uint ndbCstrs,
    float maxscorefct,
    float* __restrict__ wrkmemaux
);

// SaveBestScoreAndTM: save best scores along with transformation matrices;
template<bool WRITEFRAGINFO>
__global__
void SaveBestScoreAndTM(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux
);



// SaveBestScoreAndTMAmongBests: save best scores with transformation 
// matrices by considering all partial best scores calculated over all 
// fragment factors; write it to the location of fragment factor 0;
template<
    bool WRITEFRAGINFO,
    int GRANDUPDATE = tawmvGrandBest,
    bool FORCEWRITEFRAGINFO = false,
    int SECONDARYUPDATE = SECONDARYUPDATE_NOUPDATE>
__global__
void SaveBestScoreAndTMAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ tfmmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemtmibest2nd = NULL,
    const uint ndbCchns = 0
);

// ProductionSaveBestScoresAndTMAmongBests: save best scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; production version;
template<
    bool WRITEFRAGINFO,
    bool CONDITIONAL,
    bool COMPLEX = false,
    int CPXTYPENUMBERSCT = sfin_cpx_complextypenumber2>
__global__
void ProductionSaveBestScoresAndTMAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem,
    const uint ndbCchns = 0
);



// SaveTopNScoresAndTMsAmongBests: save top N scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors;
__global__ void SaveTopNScoresAndTMsAmongBests(
    const uint topndpscores,
    const uint ndbCstrs,
    const uint maxnsteps,
//     const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers = NULL,
    const uint ndbCchns = 0,
    const int windowsize = 0,
    const float scorethreshold = 0.0f
);

// ConditionalInitToComplex: save and initialize aux memory variables after 
// finishing the spatial index-based stage;
__global__ void ConditionalInitToComplex(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

// ConditionalInitFromComplex: copy back aux memory variables after
// finishing the spatial index-based stage;
__global__ void ConditionalInitFromComplex(
    const uint topndpscores,
    const int grandupdate,
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux
);

// // ReformatTopNFlagsComplex: populate flags of top N scores for the next
// // stage of complex processing;
// __global__ void ReformatTopNFlagsComplex(
//     const uint nqrycpxs,
//     const uint ndbCcpxs,
//     const uint maxnsteps,
//     float* __restrict__ wrkmemaux,
//     const uint ndbCchns
// );

// SaveTopNScoresAndTMsAmongSecondaryBests: save secondary top N scores and 
// respective transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the first
// N locations of fragment factors;
template<bool COMPLEX = false>
__global__ void SaveTopNScoresAndTMsAmongSecondaryBests(
    const int topndpscores,
    const int depth,
    const bool firstit,
    const bool twoconfs,
    const int rfnfragfctinit,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns = 0,
    const int seedapproachstruct = 0
);



#if 0
// SaveBestDPscoreAndTMAmongDPswifts: save best DP scores and respective 
// transformation matrices by considering all partial DP swift scores 
// calculated over all fragment factors;
// template<bool WRITEFRAGINFO, bool READSCORE, bool STEPx5>
__global__
void SaveBestDPscoreAndTMAmongDPswifts(
    bool WRITEFRAGINFO, bool READSCORE, bool STEPx5,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint effnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux
);
#endif



// SortBestDPscoresAndTMsAmongDPswifts: sort best DP scores and then save 
// them along with respective transformation matrices by considering all 
// partial DP swift scores calculated over all fragment factors; 
template<bool COMPLEX = false>
__global__ void SortBestDPscoresAndTMsAmongDPswifts(
    const uint topndpscores,
    const uint nbranches,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns = 0
);

enum {WRKMEMTMALT, WRKMEMTMIBEST, TFMMEM};
// SortBestDPscoresAndTMsAmongDPswiftsComplex: sort best DP scores and then
// save them along with respective transformation matrices by considering all 
// partial DP swift scores calculated over all fragment factors; complex version;
__global__ void SortBestDPscoresAndTMsAmongDPswiftsComplex(
    const uint topndpscores,
    const uint nbranches,
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint maxnstepsmem2,
    const int tfmtarget,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2
);



// -------------------------------------------------------------------------
// SortBestScoresAndTMsFromCPAssignmentsComplex: sort best complex scores
// originating from individual chain processing assignments;
// save these best scores along with respective transformation
// matrices to the first fragment factor locations;
__global__ void SortBestScoresAndTMsFromCPAssignmentsComplex(
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnqrychains,
    const uint maxnrfnchains,
    const uint maxnsteps,
    const uint effnsteps,
    const uint maxnstepsmem2,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2
);



// -------------------------------------------------------------------------
// ReformatDPswiftScores: reformat DP swift scores calculated over all 
// fragment factors previously and store them in the wrkmemaux memory 
// area for efficient parallel processing;
__global__ void ReformatDPswiftScores(
    const uint topndpscores,
    const uint nqystrstartindex,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint maxnstepschains,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2
);

// ReformatDPScores: reformat DP scores (calculated for a single fragment
// factor) and store them in the wrkmem2 memory area for efficient
// parallel processing;
__global__ void ReformatDPScores(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnstepsmem2,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem2,
    const bool localseqaln = false
);

// ReformatCCRfnScores: reformat chain-to-chain refinement scores and store
// them in the wrkmem2 memory area for efficient parallel processing;
__global__ void ReformatCCRfnScores(
    const uint ndbCstrs,
    const uint maxnsteps0,
    const uint maxnstepsmem2,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2
);



// -------------------------------------------------------------------------
// UpdateOneAlnPosScore: update score unconditionally for one alignment 
// position;
// SAVEPOS, template parameter to request saving positional scores;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// d02, d0 squared used for calculating score;
// d82, distance threshold for reducing scores;
// qrypos, starting query position;
// rfnpos, starting reference position;
// scrpos, position index to write the score obtained at the alignment 
// position;
// tfm, address of the transformation matrix;
// scv, address of the vector of scores;
// tmpdpdiagbuffers, global memory address for saving positional scores;
//
template<int SAVEPOS, int CHCKDST>
__device__ __forceinline__
float UpdateOneAlnPosScore(
    float d02, float d82,
    int qrypos, int rfnpos, int scrpos,
    const float* __restrict__ tfm,
    float* __restrict__ scv,
    float* __restrict__ tmpdpdiagbuffers)
{
    float qx = GetQueryCoord<pmv2DX>(qrypos);
    float qy = GetQueryCoord<pmv2DY>(qrypos);
    float qz = GetQueryCoord<pmv2DZ>(qrypos);

    float rx = GetDbStrCoord<pmv2DX>(rfnpos);
    float ry = GetDbStrCoord<pmv2DY>(rfnpos);
    float rz = GetDbStrCoord<pmv2DZ>(rfnpos);

    float dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);

    constexpr int reduce = (CHCKDST == CHCKDST_CHECK)? 0: 1;

    if(reduce || dst <= d82)
        //calculate score
        scv[threadIdx.x] += GetPairScore(d02, dst);

    if(SAVEPOS == SAVEPOS_SAVE)
        tmpdpdiagbuffers[scrpos] = dst;

    return dst;
}

// -------------------------------------------------------------------------
// UpdateOneAlnPosScore_frg2: update score unconditionally for one alignment 
// position;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// REVERSE, flag of reverse transformation;
// d02, d0 squared used for calculating score;
// d82, distance threshold for reducing scores;
// qrypos, starting query position;
// rfnpos, starting reference position;
// tfm, address of the transformation matrix;
// scv, address of the vector of scores;
//
template<int CHCKDST>
__device__ __forceinline__
void UpdateOneAlnPosScore_frg2(
    const bool REVERSE,
    float d02, float d82,
    int qrypos, int rfnpos,
    const float* __restrict__ tfm,
    float* __restrict__ scv)
{
    float qx = GetQueryCoord<pmv2DX>(qrypos);
    float qy = GetQueryCoord<pmv2DY>(qrypos);
    float qz = GetQueryCoord<pmv2DZ>(qrypos);

    float rx = GetDbStrCoord<pmv2DX>(rfnpos);
    float ry = GetDbStrCoord<pmv2DY>(rfnpos);
    float rz = GetDbStrCoord<pmv2DZ>(rfnpos);

    float dst =
        REVERSE
        ? transform_and_distance2(tfm, rx, ry, rz,  qx, qy, qz)
        : transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);

    constexpr int reduce = (CHCKDST == CHCKDST_CHECK)? 0: 1;

    if(reduce || dst <= d82)
        //calculate score
        scv[threadIdx.x] += GetPairScore(d02, dst);
}

#endif//__scoring_cuh__
