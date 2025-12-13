/***************************************************************************
 *   Copyright (C) 2021-2025 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <string>
#include <vector>
#include <map>

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/dputils.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/culayout/CuDeviceMemory.cuh"

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/dpssw_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/cudp/reformatmatch.cuh"
#include "libmycu/cuassign/chainassign.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage1_complex.cuh"
#include "custage_ssrr_complex.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stage_ssrr_complex: search for superposition between multiple 
// complexes simultaneoulsy and identify alignments by DP using secondary 
// structure information and sequence similarity criteria;
// USESEQSCORING, template parameter, flag of using sequence similarity scoring;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool USESEQSCORING>
void stage_ssrr_complex::run_stage_ssrr_complex(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const float /*scorethld*/,
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
    float* __restrict__ /*scores*/, 
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
    uint* __restrict__ /*globvarsbuf*/)
{
    //alignment based on ss information and sequence similarity:
    stage_ssrr_align_complex<USESEQSCORING>(
        streamproc,
        maxnsteps,
        nqycpxs, ndbCcpxs,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        maxnqrychains, maxnrfnchains,
        tmpdpdiagbuffers,
        tmpdpbotbuffer,
        tmpdpalnpossbuffer,
        maxscoordsbuf,
        btckdata,
        wrkmemaux, wrkmem2);

    //refine alignment boundaries to improve scores
    stage1_complex::stage1_refinefrag_complex<false/* CONDITIONAL */>(
        stgraphs,
        stg1REFINE_INITIAL_DP/*fragments identified by DP*/,
        FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
        streamproc,
        maxnsteps, minfraglen,
        nqycpxs, ndbCcpxs,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qycpx1len, dbcpx1len,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers,
        tmpdpalnpossbuffer,
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem);

    //refine alignment boundaries identified in the previous
    //substage by applying DP;
    //1. With a gap cost:
    stage1_dprefine_complex<false/*GAP0*/,false/*PRESCREEN*/>(
        stgraphs,
        streamproc,
        2/* maxndpiters */,
        prescore,
        maxnsteps, minfraglen,
        nqycpxs, ndbCcpxs,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qycpx1len, dbcpx1len,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,  maxnqrychains, maxnrfnchains,
        tmpdpdiagbuffers,
        tmpdpbotbuffer,
        tmpdpalnpossbuffer,
        maxscoordsbuf,
        btckdata,
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem/*out*/);

    //2. No gap cost:
    stage1_dprefine_complex<true/*GAP0*/,false/*PRESCREEN*/>(
        stgraphs,
        streamproc,
        maxndpiters,
        prescore,
        maxnsteps, minfraglen,
        nqycpxs, ndbCcpxs,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qycpx1len, dbcpx1len,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,  maxnqrychains, maxnrfnchains,
        tmpdpdiagbuffers,
        tmpdpbotbuffer,
        tmpdpalnpossbuffer,
        maxscoordsbuf,
        btckdata,
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem/*out*/);
}

// Instantiations
// 
#define INSTANTIATE_stage_ssrr_complex__run_stage_ssrr_complex(USESEQSCORING) \
    template void stage_ssrr_complex::run_stage_ssrr_complex<USESEQSCORING>( \
    std::map<CGKey,MyCuGraph>& stgraphs, \
    cudaStream_t streamproc, \
    const float scorethld, \
    const float prescore, \
    const int maxndpiters, \
    const uint maxnsteps, \
    const uint minfraglen, \
    uint nqycpxs, uint ndbCcpxs, \
    uint nqystrs, uint ndbCstrs, \
    uint nqyposs, uint ndbCposs, \
    uint qycpx1len, uint dbcpx1len, \
    uint qystr1len, uint dbstr1len, \
    uint qystrnlen, uint dbstrnlen, \
    uint dbxpad, \
    const uint maxnqrychains, const uint maxnrfnchains, \
    float* __restrict__ /*scores*/,  \
    float* __restrict__ tmpdpdiagbuffers, \
    float* __restrict__ tmpdpbotbuffer, \
    float* __restrict__ tmpdpalnpossbuffer, \
    uint* __restrict__ maxscoordsbuf, \
    char* __restrict__ btckdata, \
    float* __restrict__ wrkmem, \
    float* __restrict__ wrkmemccd, \
    float* __restrict__ wrkmemtm, \
    float* __restrict__ wrkmemtmibest, \
    float* __restrict__ wrkmemaux, \
    float* __restrict__ wrkmem2, \
    float* __restrict__ tfmmem, \
    uint* __restrict__ /*globvarsbuf*/);

// INSTANTIATE_stage_ssrr_complex__run_stage_ssrr_complex(false);
INSTANTIATE_stage_ssrr_complex__run_stage_ssrr_complex(true);



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage_ssrr_align_complex: get alignment based on secondary structure
// information and sequence similarity;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool USESEQSCORING>
void stage_ssrr_complex::stage_ssrr_align_complex(
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
    float* __restrict__ wrkmem2)
{
    MYMSG("stage_ssrr_complex::stage_ssrr_align_complex", 5);
    // static std::string preamb = "stage_ssrr_complex::stage_ssrr_align_complex: ";

    constexpr float gcost = -1.0f;
    static const float wgt4ss = 1.0f;//weight for scoring ss
    static const float wgt4rr = 0.2f;//weight for pairwise residue scoring
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //CUDP_2DCACHE_DIM_D x CUDP_2DCACHE_DIM_X;
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;

    //execution configuration for extracting matched positions
    //identified during DP:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);

    //execution configuration for reformating match for complexes:
    dim3 nthrds_rfmtmtch(CUDP_REFORMAT_MATCHED_DIM_X,CUDP_REFORMAT_MATCHED_DIM_Y,1);
    dim3 nblcks_rfmtmtch(ndbCcpxs,nqycpxs,1);

    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPSSwBtck3264x<USESEQSCORING>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                wgt4ss, wgt4rr, gcost,
                wrkmemaux, tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
        MYCUDACHECKLAST;
    }

    //{{ASSIGNMENT
    //execution configuration for reformatting DP scores:
    //each block processes one query and CUS1_TBSP_DPSCORE1_MAX_XDIM references:
    dim3 nthrds_dpsfmt(CUS1_TBSP_DPSCORE1_MAX_XDIM,1,1);
    dim3 nblcks_dpsfmt(
        (ndbCstrs + CUS1_TBSP_DPSCORE1_MAX_XDIM - 1)/CUS1_TBSP_DPSCORE1_MAX_XDIM,
        nqystrs, 1);

    //reformat DP scores for efficient further processing:
    ReformatDPScores<<<nblcks_dpsfmt,nthrds_dpsfmt,0,streamproc>>>(
        ndbCstrs, ndbCposs, dbxpad,  maxnstepsmem2,  tmpdpdiagbuffers, wrkmem2,
        true/*localseqaln*/);
    MYCUDACHECKLAST;

    //size of dynamically allocted smem:
    const uint szdsmem_ch2ch = GetSmemSizeForMakeChain2ChainAssignment(maxnqrychains, maxnrfnchains);

    //execution configuration for chain-to-chain assignment:
    dim3 nthrds_ch2ch(CUAP_MAKEASSIGNMENT_XDIM,1,1);
    dim3 nblcks_ch2ch(nqycpxs, ndbCcpxs, 1);

    //make chain assignemnts and calculate scores for full complexes
    MakeChain2ChainAssignment<false/*WRITESCORE*/, true/*WRITEASSG*/,false/*PASS2*/>
        <<<nblcks_ch2ch,nthrds_ch2ch,szdsmem_ch2ch,streamproc>>>(
            nqystrs, ndbCstrs, ndbCcpxs,  maxnqrychains, maxnrfnchains,  maxnsteps, maxnstepsmem2,
            NULL/*tfmmemory*/, wrkmemaux, wrkmem2);
    MYCUDACHECKLAST;
    //}}

    //produce alignment for superposition
    BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/,true/*COMPLEX*/>
        <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
            btckdata, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
            ssrr_cpx_complexstepnumber/*!=0*/,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/);
    MYCUDACHECKLAST;

    //reformat match;
    ReformatMatched4Complexes<<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/,
            ssrr_cpx_complexstepnumber/*!=0*/,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer);
    MYCUDACHECKLAST;
}
