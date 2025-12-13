/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage2_complex_h__
#define __custage2_complex_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage1_complex.cuh"

// -------------------------------------------------------------------------
// class stage2_complex for implementing complex comparison at stage 2
//
class stage2_complex: public stage1, public stage1_complex {
public:
    template<bool GAP0, bool USESS, int D02IND>
    static void run_stage2_complex(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const bool check_for_low_scores,
        const float scorethld,
        const float prescore,
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
    template<bool GAP0, bool USESS, int D02IND>
    static void stage2_dpss_align_complex(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqycpxs, const uint ndbCcpxs,
        const uint nqystrs, const uint ndbCstrs,
        const uint /* nqyposs */, const uint ndbCposs,
        const uint qystr1len, const uint dbstr1len,
        const uint /* qystrnlen */, const uint /* dbstrnlen */,
        const uint dbxpad,
        const uint maxnqrychains, const uint maxnrfnchains,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ /* maxscoordsbuf */,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

private:
};

// -------------------------------------------------------------------------

#endif//__custage2_complex_h__
