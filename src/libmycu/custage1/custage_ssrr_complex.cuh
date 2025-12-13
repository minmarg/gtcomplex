/***************************************************************************
 *   Copyright (C) 2021-2025 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage_ssrr_complex_h__
#define __custage_ssrr_complex_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage1_complex.cuh"

// -------------------------------------------------------------------------
// class stage_ssrr_complex for implementing complex comparison by matching 
// secondary structure and sequence similarity
//
class stage_ssrr_complex: public stage1, public stage1_complex {
public:
    template<bool USESEQSCORING>
    static void run_stage_ssrr_complex(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
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
    template<bool USESEQSCORING>
    static void stage_ssrr_align_complex(
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
        float* __restrict__ wrkmem2
    );

private:
};

// -------------------------------------------------------------------------

#endif//__custage_ssrr_complex_h__
