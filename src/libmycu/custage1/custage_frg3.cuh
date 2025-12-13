/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage_frg3_h__
#define __custage_frg3_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage1_complex.cuh"

// -------------------------------------------------------------------------
// class stagefrg for implementing structure comparison by exhaustive 
// fragment matching; serves for finding most favorable initial 
// superposition state for further refinement
//
class stagefrg3: public stage1, public stage1_complex {
public:
    static void run_stagefrg3_complexes(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const int stage3asfilter,
        const int depth,
        const int nbranches,
        const int simthreshold,
        const int maxndpiters,
        const uint maxnsteps,
        const uint minfraglen,
        const float scorethld,
        const float prescore,
        uint nqycpxs, uint ndbCcpxs,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qycpx1len, uint dbcpx1len,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        const uint maxnqrychains, const uint maxnrfnchains,
        float* __restrict__ scores, 
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ maxscoordsbuf,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtmalt,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        uint* __restrict__ globvarsbuf
    );

protected:
    static void stagefrg3_score_based_on_fragmatching3(
        const int depth,
        const int napiterations,
        const bool scoreachiteration,
        const bool dynamicorientation,
        const float thrsimilarityperc,
        const float thrscorefactor,
        cudaStream_t streamproc,
        const int maxfraglen,
        const int qryfragfct, const int rfnfragfct, const int fragndx,
        const uint maxnsteps,
        const uint actualnsteps,
        const uint qystr1len, const uint dbstr1len,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        const char* __restrict__ dpscoremtx,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm,
        const float thrpercentage_chainoccupancy = 0.0f
    );

    static void stagefrg3_score_based_on_fragmatching3_helper(
        const int depth,
        const int napiterations,
        const bool scoreachiteration,
        const bool dynamicorientation,
        const float thrsimilarityperc,
        const float thrscorefactor,
        cudaStream_t streamproc,
        const int maxfraglen,
        const int qryfragfct, const int rfnfragfct, const int fragndx,
        const uint maxnsteps,
        const uint actualnsteps,
        const uint qystr1len, const uint dbstr1len,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        const char* __restrict__ dpscoremtx,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm,
        const float thrpercentage_chainoccupancy = 0.0f
    );


    static void prepare_for_index_stage_complex(
        cudaStream_t streamproc,
        const float prescore,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqycpxs, uint ndbCcpxs,
        uint /* nqystrs */, uint ndbCstrs,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ /* wrkmemtm */,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux
    );

    static void stagefrg3_extensive_frg_swift_complexes(
        cudaStream_t streamproc,
        const int depth,
        const int nbranches,
        const int simthreshold,
        const float prescore,
        const uint maxnsteps,
        uint nqycpxs, uint ndbCcpxs,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qycpx1len, uint dbcpx1len,
        uint qystr1len, uint dbstr1len,
        uint dbxpad,
        const uint maxnqrychains, const uint maxnrfnchains,
        const int tfmtarget,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* /*__restrict__*/ wrkmemtmalt,
        float* __restrict__ wrkmemtm,
        float* /*__restrict__*/ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* /*__restrict__*/ tfmmem
    );

    static void stagefrg3_refinement_tfmaltconfig_complexes(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const int depth,
        const int nbranches,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqycpxs, uint ndbCcpxs,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qycpx1len, uint dbcpx1len,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtmalt,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem);

private:

    //----------------------------------------------------------------------

    static void stagefrg3_fragmatching3_Initial1(
        const int depth,
        const bool dynamicorientation,
        const float thrsimilarityperc,
        const float thrscorefactor,
        cudaStream_t streamproc,
        const int qryfragfct, const int rfnfragfct, const int fragndx,
        const uint maxnsteps, const uint actualnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        const char* __restrict__ dpscoremtx,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_locsim, const dim3& nthrds_locsim,
        const dim3& nblcks_simeval, const dim3& nthrds_simeval,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    static void stagefrg3_fragmatching3_alignment2(
        const int depth,
        const bool dynamicorientation,
        const float thrsimilarityperc,
        const bool secstrmatchaln,
        const bool complete,
        const bool writeqrypss,
        cudaStream_t streamproc,
        const int qryfragfct, const int rfnfragfct, const int fragndx,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmemtm,
        //
        const dim3& nblcks_linscos, const dim3& nthrds_linscos,
            const uint szdsmem_linscos, const uint stacksize_linscos,
        const dim3& nblcks_linalign, const dim3& nthrds_linalign
    );

    static void stagefrg3_fragmatching3_superposition3(
        const bool dynamicorientation,
        const bool reversetfms,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    static void stagefrg3_fragmatching3_aln_scoring4(
        cudaStream_t streamproc,
        const int depth,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        const int qryfragfct, const int rfnfragfct,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ /*wrkmem2*/,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        //
        const dim3& /*nblcks_init*/, const dim3& /*nthrds_init*/,
        const dim3& /*nblcks_ccmtx*/, const dim3& /*nthrds_ccmtx*/,
        const dim3& /*nblcks_findd2*/, const dim3& /*nthrds_findd2*/,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_scores, const dim3& nthrds_scores,
        const dim3& nblcks_savetm, const dim3& nthrds_savetm,
        const dim3& /*nblcks_copyto*/, const dim3& /*nthrds_copyto*/,
        const dim3& /*nblcks_copyfrom*/, const dim3& /*nthrds_copyfrom*/,
        const dim3& /*nblcks_tfm*/, const dim3& /*nthrds_tfm*/,
        const float thrpercentage_chainoccupancy
    );

    // ---------------------------------------------------------------------
};

// -------------------------------------------------------------------------

#endif//__custage_frg3_h__
