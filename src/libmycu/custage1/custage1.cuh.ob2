/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage1_h__
#define __custage1_h__

// -------------------------------------------------------------------------
// class stage1 for implementing structure comparison at stage 1
//
class stage1 {
public:
    static void run_stage1(
        cudaStream_t streamproc,
        float scorethld,
        int stepinit,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ scores, 
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        uint* __restrict__ maxscoordsbuf,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        uint* __restrict__ globvarsbuf
    );
private:
    static void stage1_subiter1(
        cudaStream_t streamproc,
        uint ndbCstrs,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        //
        int qrypos, int rfnpos,
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    static void stage1_subiter2(
        cudaStream_t streamproc,
        uint ndbCstrs,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        //
        int qrypos, int rfnpos,
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    static void stage1_subiter3(
        cudaStream_t streamproc,
        uint ndbCstrs,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        //
        int qrypos, int rfnpos,
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_scores, const dim3& nthrds_scores,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );
};

// -------------------------------------------------------------------------


#endif//__custage1_h__
