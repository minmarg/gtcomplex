/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage_fin_complex_h__
#define __custage_fin_complex_h__

#include "libutil/macros.h"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage1_complex.cuh"

// -------------------------------------------------------------------------
// class stagefin_complex for implementing final complex alignment
// refinement for output
//
class stagefin_complex: public stage1, public stage1_complex {
public:
    static void run_stagefin_complex(
        cudaStream_t streamproc,
        const float d2equiv,
        const float scorethld,
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
        float* __restrict__ alndatamem,
        float* __restrict__ tfmmem,
        char* __restrict__ alnsmem,
        uint* __restrict__ globvarsbuf
    );

protected:
    static void stagefin_align_complex(
        const bool constrainedbtck,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqycpxs, const uint ndbCcpxs,
        const uint nqystrs, const uint ndbCstrs,
        const uint /*nqyposs*/, const uint ndbCposs,
        const uint qystr1len, const uint dbstr1len,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ /*maxscoordsbuf*/,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

    static void stagefin_align_constrained_complex(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqycpxs, const uint ndbCcpxs,
        const uint nqystrs, const uint ndbCstrs,
        const uint /*nqyposs*/, const uint ndbCposs,
        const uint qystr1len, const uint dbstr1len,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ /*maxscoordsbuf*/,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

    static void stagefin_align_unconstrained_complex(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqycpxs, const uint ndbCcpxs,
        const uint nqystrs, const uint ndbCstrs,
        const uint /*nqyposs*/, const uint ndbCposs,
        const uint qystr1len, const uint dbstr1len,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        const uint maxnqrychains, const uint maxnrfnchains,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ /*maxscoordsbuf*/,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

    static void stagefin_refine_complex(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        const uint nqycpxs, const uint ndbCcpxs,
        const uint nqystrs, const uint ndbCstrs,
        const uint /* nqyposs */, const uint ndbCposs,
        const uint qycpx1len, const uint dbcpx1len,
        const uint qystr1len, const uint dbstr1len,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem
    );

    static void stagefin_produce_output_alignment_complex(
        const float d2equiv,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqycpxs, const uint ndbCcpxs,
        const uint nqystrs, const uint ndbCstrs,
        const uint /* nqyposs */, const uint ndbCposs,
        const uint /* qystr1len */, const uint /* dbstr1len */,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ alndatamem,
        char* __restrict__ alnsmem
    );

    static void stagefin_produce_output_scores_complex(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        const uint nqycpxs, const uint ndbCcpxs,
        const uint nqystrs, const uint ndbCstrs,
        const uint /* nqyposs */, const uint ndbCposs,
        const uint qycpx1len, const uint dbcpx1len,
        const uint qystr1len, const uint dbstr1len,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ alndatamem,
        float* __restrict__ tfmmem
    );

    static void stagefin_produce_CS_scores_complex(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint /*nqyposs*/, const uint ndbCposs,
        const uint /*qystr1len*/, const uint /*dbstr1len*/,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        const float* __restrict__ tmpdpalnpossbuffer,
        const float* __restrict__ wrkmemaux,
        const float* __restrict__ wrkmem2,
        const float* __restrict__ tfmmem,
        float* __restrict__ alndatamem
    );

    static void stagefin_produce_2TMscores_complex(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint /*nqyposs*/, const uint ndbCposs,
        const uint /*qystr1len*/, const uint /*dbstr1len*/,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        const float* __restrict__ tmpdpalnpossbuffer,
        const float* __restrict__ wrkmemaux,
        const float* __restrict__ wrkmem2,
        const float* __restrict__ tfmmem,
        float* __restrict__ alndatamem
    );

    static void stagefin_adjust_tfms_complex(
        cudaStream_t streamproc,
        const uint nqycpxs, const uint ndbCcpxs,
        const uint /*nqystrs*/, const uint ndbCstrs,
        const uint /*nqyposs*/, const uint /*ndbCposs*/,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem
    );

private:
};

// -------------------------------------------------------------------------

#endif//__custage_fin_complex_h__
