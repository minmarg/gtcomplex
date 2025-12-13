/***************************************************************************
 *   Copyright (C) 2021-2025 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage_chor_complex_h__
#define __custage_chor_complex_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage1_complex.cuh"

// -------------------------------------------------------------------------
// class stage2_complex for implementing complex comparison at stage 2
//
class stage_chor_complex: public stage1, public stage1_complex {
public:
    static void run_stage_chor_complex(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const float scorethld,
        const float prescore,
        const int stepinit,
        const int maxndpiters,
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
        const uint minnqrychains, const uint minnrfnchains,
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
    static void stage_chor_find_chain_level_orientations_complex(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const float prescore,
        const int stepinit,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
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

    static void stage_chor_identify_top_orientations_complex(
        cudaStream_t streamproc,
        const uint maxnsteps,
        uint nqycpxs, uint ndbCcpxs,
        uint nqystrs, uint ndbCstrs,
        uint /* nqyposs */, uint /* ndbCposs */,
        uint /* qystr1len */, uint /* dbstr1len */,
        const uint maxnqrychains, const uint maxnrfnchains,
        float* __restrict__ wrkmemtmalt,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2
    );

    static void stage_chor_refine_on_top_orientations_complex(
        std::map<CGKey,MyCuGraph>& stgraphs,
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
        const uint maxnqrychains, const uint maxnrfnchains,
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
        float* __restrict__ tfmmem
    );

private:
};

// -------------------------------------------------------------------------

#endif//__custage_chor_complex_h__
