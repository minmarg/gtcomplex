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
#include "libmycu/cudp/dpssw_tfm_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/cudp/reformatmatch.cuh"
#include "libmycu/cuassign/chainassign.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage1_complex.cuh"
#include "custage_chor_complex.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stage_chor_complex: a stage of search for superposition and alignment
// identification between multiple complexes simultaneoulsy, initially
// based on chain-level orientation compatibility;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage_chor_complex::run_stage_chor_complex(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const float /* scorethld */,
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
    float* __restrict__ /*scores*/, 
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
    uint* __restrict__ /*globvarsbuf*/)
{
    static const std::string preamb = "stage_chor_complex::run_stage_chor_complex: ";
    static const uint maxnquerychains = (uint)(CLOptions::GetDEV_QRS_MAX_CHAINS());

    //skip the stage of individual chain processing for complexes if all complexes
    //have more than CUS1_TBSP_CPXSCORE_MAX_NCHAINS chains:
    if(CUS1_TBSP_CPXSCORE_MAX_NCHAINS < mymin(minnqrychains, minnrfnchains))
        return;

    if(maxnsteps < maxnquerychains)
        throw MYRUNTIME_ERROR(preamb + "#max chains exceeds #max concurrent steps.");

    //NOTE: chain-specific maxnsteps to not exceed memory limits:
    const uint maxnsteps0 = mymax((uint)1, maxnsteps / maxnquerychains);
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //find suboptimal TFMs (chain orientations) for all chain pairs
    stage_chor_find_chain_level_orientations_complex(
        stgraphs, streamproc,
        prescore, stepinit,
        maxnsteps0, minfraglen,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers,
        tmpdpbotbuffer,
        tmpdpalnpossbuffer,
        maxscoordsbuf,
        btckdata,
        wrkmem, wrkmemccd, wrkmemtm/*out*/, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem/*unused*/);

    dim3 nthrds_ccrfmt(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_ccrfmt(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, 1);

    ReformatCCRfnScores<<<nblcks_ccrfmt,nthrds_ccrfmt,0,streamproc>>>(
        ndbCstrs, maxnsteps0, maxnstepsmem2,  wrkmemaux, wrkmem2);
    MYCUDACHECKLAST;

    //identify top scoring complex TFMs from chain-level orientations for all complexes;
    stage_chor_identify_top_orientations_complex(
        streamproc, maxnsteps,
        nqycpxs, ndbCcpxs,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs ,
        qystr1len, dbstr1len ,
        maxnqrychains, maxnrfnchains,
        wrkmemtmalt/*out*/, wrkmemtm, wrkmemaux, wrkmem2);

    //refine complex alignment boundaries to improve scores
    stage_chor_refine_on_top_orientations_complex(
        stgraphs,
        streamproc,
        maxnsteps, minfraglen,
        nqycpxs, ndbCcpxs,  nqystrs, ndbCstrs,  nqyposs, ndbCposs,
        qycpx1len, dbcpx1len,  qystr1len, dbstr1len,  qystrnlen, dbstrnlen,  dbxpad,
        maxnqrychains, maxnrfnchains,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, btckdata,
        wrkmem, wrkmemccd, wrkmemtmalt, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem/*out*/);

    //refine complex alignment by applying DP;
    stage1_dprefine_complex
        <true/*GAP0*/,false/*PRESCREEN*/,false/*WRKMEMTM1*/,
         CUS1_TBSP_CPXSCORE_MAX_NCHAINS>(
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





// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage_chor_find_chain_level_orientations_complex: find suboptimal TFMs
// (chain orientations) for all chain pairs of complexes;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage_chor_complex::stage_chor_find_chain_level_orientations_complex(
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
    float* __restrict__ /* tfmmem */)
{
    MYMSG("stage_chor_complex::stage_chor_find_chain_level_orientations_complex", 5);
    // static std::string preamb = "stage_chor_complex::stage_chor_find_chain_level_orientations_complex: ";

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

    dim3 nthrds_cvinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_cvinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, 1);

    //initialize memory for current scores (tawmvScore) too;
    //they're used as a substitute for grand scores here!
    InitScores<INITOPT_ALL>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,
            wrkmemaux);
    MYCUDACHECKLAST;

    //set convergence flags for pairs of unmatched types
    SetConvergenceForUnmatchedTypes
        <<<nblcks_cvinit,nthrds_cvinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, wrkmemaux);
    MYCUDACHECKLAST;

    //find best scoring aligned gapless fragment
    stage1_findfrag2<CUS1_TBSP_CPXSCORE_MAX_NCHAINS>(
        streamproc, stepinit,
        maxnsteps, minfraglen,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers,
        wrkmem, wrkmemaux, wrkmem2, wrkmemtm/*unused*/);

    //refine fragment boundaries to improve scores
    //NOTE: write best obtained scores to the tawmvScore section!
    stage1_refinefrag
        <false/*CONDITIONAL*/,SECONDARYUPDATE_NOUPDATE,
         tawmvScore/*GRANDUPDATE*/,CUS1_TBSP_CPXSCORE_MAX_NCHAINS>(
            stgraphs,
            stg1REFINE_INITIAL/*fragments identified initially*/,
            FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
            streamproc,
            maxnsteps, minfraglen,
            nqystrs, ndbCstrs,
            nqyposs, ndbCposs,
            qystr1len, dbstr1len,
            qystrnlen, dbstrnlen,
            dbxpad,
            tmpdpdiagbuffers,
            tmpdpalnpossbuffer,
            wrkmem, wrkmemccd, NULL/*wrkmemtm*used only for SECONDARYUPDATEs*/, wrkmemtmibest,
            wrkmemaux, wrkmem2, wrkmemtm/*tfmmem, write suboptimal TFMs here*/);

    //refine alignment boundaries identified in the previous 
    //substage by applying DP
    //NOTE: write best obtained scores to the tawmvScore section!
    stage1_dprefine
        <true/*GAP0*/,false/*PRESCREEN*/,false/*WRKMEMTM1*/,
         tawmvScore/*GRANDUPDATE*/,CUS1_TBSP_CPXSCORE_MAX_NCHAINS>(
            stgraphs,
            streamproc,
            2/*maxndpiters*/,
            prescore,
            maxnsteps, minfraglen,
            nqystrs, ndbCstrs,
            nqyposs, ndbCposs,
            qystr1len, dbstr1len,
            qystrnlen, dbstrnlen,
            dbxpad,
            tmpdpdiagbuffers,
            tmpdpbotbuffer,
            tmpdpalnpossbuffer,
            maxscoordsbuf,
            btckdata,
            wrkmem, wrkmemccd, NULL/*wrkmemtm*used only for SECONDARYUPDATEs*/, wrkmemtmibest,
            wrkmemaux, wrkmem2, wrkmemtm/*tfmmem [out], write suboptimal TFMs here*/);
}

// -------------------------------------------------------------------------
// stage_chor_identify_top_orientations_complex: identify top scoring complex
// TFM from chain-level orientations for all complexes;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage_chor_complex::stage_chor_identify_top_orientations_complex(
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
    float* __restrict__ wrkmem2)
{
    MYMSG("stage_chor_complex::stage_chor_identify_top_orientations_complex", 5);
    // static std::string preamb = "stage_chor_complex::stage_chor_identify_top_orientations_complex: ";
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, maxnsteps);

    //initialize memory for best scores (tawmvBestScore) only;
    //they'll contain best heuristic complex scores!
    //also, 1) reset chain-specific convergence flags:
    InitScores<INITOPT_ALL>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    dim3 nthrds_cvinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_cvinit(
        (ndbCcpxs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, 1);

    // 2) set convergence for incompatible complexes:
    SetConvergenceForUnmatchedTypesComplex
        <<<nblcks_cvinit,nthrds_cvinit,0,streamproc>>>(
            ndbCcpxs, ndbCstrs, maxnsteps, wrkmemaux);
    MYCUDACHECKLAST;

    //size of dynamically allocted smem:
    const uint szdsmem_ch2ch = GetSmemSizeForMakeChain2ChainAssignment(maxnqrychains, maxnrfnchains);
    const uint effnsteps = maxnqrychains * maxnrfnchains;
    const uint maxeffnsteps =
        (CUS1_TBSP_CPXSCORE_MAX_NCHAINS < maxnqrychains)? maxnqrychains * CUS1_TBSP_CPXSCORE_MAX_NCHAINS: effnsteps;

    //execution configuration for chain-to-chain assignment:
    //NOTE: make sure the product of the number of chains doesn't exceed the allowed limit!
    dim3 nthrds_ch2ch(CUAP_MAKEASSIGNMENT_XDIM,1,1);
    dim3 nblcks_ch2ch(nqycpxs, ndbCcpxs, maxeffnsteps);

    //make chain assignemnts and calculate scores for full complexes;
    //NOTE: that at least one of a pair of complexes has <= 
    //NOTE: CUS1_TBSP_CPXSCORE_MAX_NCHAINS chains (<=CUS1_TBSP_DPSCORE_TOP_N)
    //NOTE: ensures wrkmem2's maxnstepsmem2 slots don't overflow!
    MakeChain2ChainAssignment
        <true/*WRITESCORE*/, false/*WRITEASSG*/, true/*PASS2*/, CUS1_TBSP_CPXSCORE_MAX_NCHAINS>
        <<<nblcks_ch2ch,nthrds_ch2ch,szdsmem_ch2ch,streamproc>>>(
            nqystrs, ndbCstrs, ndbCcpxs,  maxnqrychains, maxnrfnchains,  maxnsteps, maxnstepsmem2,
            wrkmemtm, wrkmemaux, wrkmem2);
    MYCUDACHECKLAST;

    //execution configuration for finding the maximum among complex scores:
    //each block processes one query and CUS1_TBSP_CPXSCORE_MAX_XDIM references:
    dim3 nthrds_cpxscmax(CUS1_TBSP_CPXSCORE_MAX_XDIM,CUS1_TBSP_CPXSCORE_MAX_XDIM,1);
    dim3 nblcks_cpxscmax(
        (ndbCcpxs + CUS1_TBSP_CPXSCORE_MAX_XDIM - 1)/CUS1_TBSP_CPXSCORE_MAX_XDIM,
        nqycpxs, 1);

    //sort the best complex scores and save the corresponding tfms:
    //NOTE: using attributes of complexes rather than chains for compact packing:
    SortBestScoresAndTMsFromCPAssignmentsComplex
        <<<nblcks_cpxscmax,nthrds_cpxscmax,0,streamproc>>>(
            ndbCcpxs, ndbCstrs,  maxnqrychains, maxnrfnchains,
            maxnsteps, maxeffnsteps, maxnstepsmem2,
            wrkmemtm/*in*/, wrkmemtmalt/*out*/, wrkmemaux, wrkmem2);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage_chor_refine_on_top_orientations_complex: refine all alternative 
// best-performing complex superpositions obtained through chain-level
// processing assignments;
// qycpx1len, dbcpx1len, lengths of the largest query & reference complexes;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage_chor_complex::stage_chor_refine_on_top_orientations_complex(
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
    float* __restrict__ tfmmem)
{
    MYMSG("stage_chor_complex::stage_chor_refine_on_top_orientations_complex", 5);
    static const std::string preamb = "stage_chor_complex::stage_chor_refine_on_top_orientations_complex: ";
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //execution configuration for DP:
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);
    //number of regular DIAGONAL block diagonal series;
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;

    //execution configuration for DP-matched positions:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);

    //execution configuration for reformating match for complexes:
    dim3 nthrds_rfmtmtch(CUDP_REFORMAT_MATCHED_DIM_X,CUDP_REFORMAT_MATCHED_DIM_Y,1);
    dim3 nblcks_rfmtmtch(ndbCcpxs,nqycpxs,1);

    //process top N best-performing tfms found from chain-level
    //processing assignments:
    for(int rci = 0; rci < CUS1_TBSP_CPXSCORE_TOP_N; rci++)
    {
        //launch blocks along block diagonals to perform DP;
        //nblkdiags, total number of diagonals:
        for(uint d = 0; d < nblkdiags; d++)
        {
            ExecDPwBtck3264x
                <false/*ANCHORRGN*/,false/*BANDED*/,true/*GAP0*/,D02IND_SEARCH,
                 true/*ALTSCTMS*/,true/*COMPLEX*/,CUS1_TBSP_CPXSCORE_MAX_NCHAINS>
                <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                    d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, rci/*stepnumber*/,
                    0.0f/*gap open cost*/,
                    wrkmemtmalt/*in*/, wrkmemaux,
                    tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
                    //NOTE: all-vs-all chain alignment using complex-level tfms when
                    //NOTE: assgmaxnsteps is not provided!
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
            ndbCstrs, ndbCposs, dbxpad,  maxnstepsmem2,  tmpdpdiagbuffers, wrkmem2);
        MYCUDACHECKLAST;

        //size of dynamically allocted smem:
        const uint szdsmem_ch2ch = GetSmemSizeForMakeChain2ChainAssignment(maxnqrychains, maxnrfnchains);

        //execution configuration for chain-to-chain assignment:
        dim3 nthrds_ch2ch(CUAP_MAKEASSIGNMENT_XDIM,1,1);
        dim3 nblcks_ch2ch(nqycpxs, ndbCcpxs, 1);

        //make chain assignemnts and calculate scores for full complexes
        MakeChain2ChainAssignment
            <false/*WRITESCORE*/, true/*WRITEASSG*/, false/*PASS2*/, CUS1_TBSP_CPXSCORE_MAX_NCHAINS>
                <<<nblcks_ch2ch,nthrds_ch2ch,szdsmem_ch2ch,streamproc>>>(
                    nqystrs, ndbCstrs, ndbCcpxs,  maxnqrychains, maxnrfnchains,  maxnsteps, maxnstepsmem2,
                    NULL/*tfmmemory*/, wrkmemaux, wrkmem2);
        MYCUDACHECKLAST;
        //}}

        //produce alignment given superposition and chain assignments
        BtckToMatched32x
            <false/*ANCHORRGN*/,false/*BANDED*/,true/*COMPLEX*/,CUS1_TBSP_CPXSCORE_MAX_NCHAINS>
                <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                    ndbCstrs, ndbCposs, dbxpad, maxnsteps, rci/*stepnumber*/,
                    btckdata, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
                    chor_cpx_complexstepnumber,
                    maxnstepsmem2/*assgmaxnsteps*/,
                    (0)/*assgstepnumber*/);
        MYCUDACHECKLAST;

        //reformat match;
        ReformatMatched4Complexes<CUS1_TBSP_CPXSCORE_MAX_NCHAINS>
            <<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
                ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                maxnstepsmem2/*assgmaxnsteps*/,
                (0)/*assgstepnumber*/,
                chor_cpx_complexstepnumber,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer);
        MYCUDACHECKLAST;

        //refine complex alignment boundaries to improve scores
        stage1_complex::stage1_refinefrag_complex
            <false/*CONDITIONAL*/,SECONDARYUPDATE_NOUPDATE,CUS1_TBSP_CPXSCORE_MAX_NCHAINS>(
                stgraphs,
                stg1REFINE_INITIAL_DP/*fragments identified by DP*/,
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                streamproc,
                maxnsteps, minfraglen,
                nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs,
                qycpx1len, dbcpx1len, qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
                tmpdpdiagbuffers, tmpdpalnpossbuffer,
                wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
                wrkmemaux, wrkmem2, tfmmem);
    }//rci
}
