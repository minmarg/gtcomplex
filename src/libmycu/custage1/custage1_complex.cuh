/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage1_complex_h__
#define __custage1_complex_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custage1/custage1.cuh"

// -------------------------------------------------------------------------
// class stage1 for implementing structure comparison at stage 1
//
class stage1_complex {
public:
    // static void preinitialize1(
    //     cudaStream_t streamproc,
    //     const bool condition4filter1,
    //     const uint maxnsteps,
    //     const uint minfraglen,
    //     uint nqystrs, uint ndbCstrs,
    //     uint nqyposs, uint ndbCposs,
    //     float* __restrict__ wrkmemtmibest,
    //     float* __restrict__ wrkmemaux,
    //     float* __restrict__ tfmmem,
    //     float* __restrict__ alndatamem
    // );

    // static void run_stage1(
    //     std::map<CGKey,MyCuGraph>& stgraphs,
    //     cudaStream_t streamproc,
    //     const int maxndpiters,
    //     const uint maxnsteps,
    //     const uint minfraglen,
    //     const float scorethld,
    //     const float prescore,
    //     int stepinit,
    //     uint nqystrs, uint ndbCstrs,
    //     uint nqyposs, uint ndbCposs,
    //     uint qystr1len, uint dbstr1len,
    //     uint qystrnlen, uint dbstrnlen,
    //     uint dbxpad,
    //     float* __restrict__ scores, 
    //     float* __restrict__ tmpdpdiagbuffers,
    //     float* __restrict__ tmpdpbotbuffer,
    //     float* __restrict__ tmpdpalnpossbuffer,
    //     uint* __restrict__ maxscoordsbuf,
    //     char* __restrict__ btckdata,
    //     float* __restrict__ wrkmem,
    //     float* __restrict__ wrkmemccd,
    //     float* __restrict__ wrkmemtm,
    //     float* __restrict__ wrkmemtmibest,
    //     float* __restrict__ wrkmemaux,
    //     float* __restrict__ wrkmem2,
    //     float* __restrict__ tfmmem,
    //     uint* __restrict__ globvarsbuf
    // );

    //{{ --- DP ---
    template<
        bool ANCHORRGN,
        bool BANDED,
        bool GAP0,
        int MAX_NCHAINS = 0>
    static void RunDP(
        cudaStream_t streamproc,
        const float gapcost,
        const uint maxnsteps,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux);
    //}}

protected:
    // static void stage1_findfrag2(
    //     cudaStream_t streamproc,
    //     int stepinit,
    //     const uint maxnsteps,
    //     const uint minfraglen,
    //     uint nqystrs, uint ndbCstrs,
    //     uint nqyposs, uint ndbCposs,
    //     uint qystr1len, uint dbstr1len,
    //     uint qystrnlen, uint dbstrnlen,
    //     uint dbxpad,
    //     float* __restrict__ tmpdpdiagbuffers,
    //     float* __restrict__ wrkmem,
    //     float* __restrict__ wrkmemaux,
    //     float* __restrict__ wrkmem2,
    //     float* __restrict__ wrkmemtm
    // );

    template<
        bool CONDITIONAL,
        int SECONDARYUPDATE = SECONDARYUPDATE_NOUPDATE,
        int MAX_NCHAINS = 0>
    static void stage1_refinefrag_complex(
        std::map<CGKey,MyCuGraph>& stgraphs,
        const int fragbydp,
        const int nmaxconvit,
        cudaStream_t streamproc,
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
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

public:
    template<
        bool GAP0,
        bool PRESCREEN = false,
        bool WRKMEMTM1 = false,
        int MAX_NCHAINS = 0>
    static void stage1_dprefine_complex(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const int maxndpiters,
        const float prescore,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqycpxs, uint ndbCcpxs,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qycpx1len, uint dbcpx1len,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        const uint maxnqrychains, const uint maxnrfnchains,
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
        float* __restrict__ tfmmem
    );

private:

    template<
        bool CONDITIONAL,
        int SECONDARYUPDATE,
        int MAX_NCHAINS = 0>
    static void stage1_refinefrag_helper2_complex(
        const int fragbydp,
        const int nmaxconvit,
        cudaStream_t streamproc,
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
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem
    );

    //----------------------------------------------------------------------
};

// -------------------------------------------------------------------------

#endif//__custage1_complex_h__
