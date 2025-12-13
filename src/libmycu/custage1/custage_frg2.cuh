/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage_frg2_h__
#define __custage_frg2_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custage1/custage1.cuh"

// -------------------------------------------------------------------------
// class stagefrg for implementing structure comparison by exhaustive 
// fragment matching; serves for finding most favorable initial 
// superposition state for further refinement
//
class stagefrg2: public stage1 {
public:
    static void run_stagefrg2(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        float scorethld,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ scores, 
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ maxscoordsbuf,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        uint* __restrict__ globvarsbuf
    );

protected:
    static void stagefrg2_extensive_frg_swift(
        cudaStream_t streamproc,
        const uint maxnsteps,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );


    static void stagefrg2_align_based_on_fragmatching3(
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
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm
    );

    static void stagefrg2_score_alignments3(
        const int napiterations,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint actualnsteps,
        const uint qystr1len, const uint dbstr1len,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm
    );


private:

    //----------------------------------------------------------------------

    static void stagefrg2_fragmatching3_subiter1(
        cudaStream_t streamproc,
        const int qryfragfct, const int rfnfragfct, const int fragndx,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
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

    static void stagefrg2_fragmatching3_subiter2(
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

    // ---------------------------------------------------------------------

    static void stagefrg2_score_alignments3_subiter1(
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

    static void stagefrg2_score_alignments3_subiter2(
        const int napiterations,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_findd2, const dim3& nthrds_findd2,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_scores, const dim3& nthrds_scores,
        const dim3& nblcks_savetm, const dim3& nthrds_savetm,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    static void stagefrg2_score_alignments3_subiter3(
        const int napiterations,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint ndbCposs, const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_findd2, const dim3& nthrds_findd2,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_scores, const dim3& nthrds_scores,
        const dim3& nblcks_savetm, const dim3& nthrds_savetm,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );
};

// -------------------------------------------------------------------------

#endif//__custage_frg2_h__
