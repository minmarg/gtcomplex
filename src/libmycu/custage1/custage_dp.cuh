/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage_dp_h__
#define __custage_dp_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custage1/custage1.cuh"

// -------------------------------------------------------------------------
// class stagedp for implementing structure comparison by exhaustive 
// application of DP
//
class stagedp: public stage1 {
public:
    static void run_stagedp(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const uint maxnsteps,
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
    static void stagedp_extensive_dp_complete(
        cudaStream_t streamproc,
        const uint maxnsteps,
        uint nqystrs, uint ndbCstrs,
        uint /*nqyposs*/, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint /*qystrnlen*/, uint /*dbstrnlen*/,
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

    static void stagedp_extensive_dp_swift(
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

    static void stagedp_score_alignment3(
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
    static void stagedp_tfms_from_fragments(
        cudaStream_t streamproc,
        int qryfragfct, int rfnfragfct, int fragndx,
        const uint maxnsteps,
        uint nqystrs, uint ndbCstrs,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx_var, const dim3& nthrds_ccmtx_var,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    //----------------------------------------------------------------------

    static void stagedp_scorealn3_subiter1(
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

    static void stagedp_scorealn3_subiter2(
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

    static void stagedp_scorealn3_subiter3(
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

#endif//__custage_dp_h__
