/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
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
#include "libmycu/cucom/cutimer.cuh"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/culayout/CuDeviceMemory.cuh"

#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/covariance_dp_refn.cuh"
#include "libmycu/custages2/covariance_complete.cuh"
#include "libmycu/custages2/covariance_refn_complete.cuh"
#include "libmycu/custages2/covariance_dp_refn_complete.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/cudp/reformatmatch.cuh"

#include "libmycu/cuassign/chainassign.cuh"

#include "custage1_complex.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stage1: initial (stage-1) search for superposition between multiple 
// molecules simultaneoulsy and identify fragments for further refinement;
// tmalign correspondence: get_initial, detailed_search, DP_iter;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
// void stage1::run_stage1(
//     std::map<CGKey,MyCuGraph>& stgraphs,
//     cudaStream_t streamproc,
//     const int maxndpiters,
//     const uint maxnsteps,
//     const uint minfraglen,
//     const float /* scorethld */,
//     const float prescore,
//     int stepinit,
//     uint nqystrs, uint ndbCstrs,
//     uint nqyposs, uint ndbCposs,
//     uint qystr1len, uint dbstr1len,
//     uint qystrnlen, uint dbstrnlen,
//     uint dbxpad,
//     float* __restrict__ /*scores*/, 
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
//     uint* __restrict__ /* globvarsbuf */)
// {
//     //find best scoring aligned gapless fragment
//     stage1_findfrag2(
//         streamproc, stepinit,
//         maxnsteps, minfraglen,
//         nqystrs, ndbCstrs,
//         nqyposs, ndbCposs,
//         qystr1len, dbstr1len,
//         qystrnlen, dbstrnlen,
//         dbxpad,
//         tmpdpdiagbuffers,
//         wrkmem, wrkmemaux, wrkmem2, wrkmemtm);

//     //refine fragment boundaries to improve scores
//     stage1_refinefrag<false/* CONDITIONAL */>(
//         stgraphs,
//         stg1REFINE_INITIAL/*fragments identified initially*/,
//         FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
//         streamproc,
//         maxnsteps, minfraglen,
//         nqystrs, ndbCstrs,
//         nqyposs, ndbCposs,
//         qystr1len, dbstr1len,
//         qystrnlen, dbstrnlen,
//         dbxpad,
//         tmpdpdiagbuffers,
//         tmpdpalnpossbuffer,
//         wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
//         wrkmemaux, wrkmem2, tfmmem);

//     //refine alignment boundaries identified in the previous 
//     //substage by applying DP
//     stage1_dprefine<true/*GAP0*/,true/*PRESCREEN*/>(
//         stgraphs,
//         streamproc,
//         maxndpiters,
//         prescore,
//         maxnsteps, minfraglen,
//         nqystrs, ndbCstrs,
//         nqyposs, ndbCposs,
//         qystr1len, dbstr1len,
//         qystrnlen, dbstrnlen,
//         dbxpad,
//         tmpdpdiagbuffers,
//         tmpdpbotbuffer,
//         tmpdpalnpossbuffer,
//         maxscoordsbuf,
//         btckdata,
//         wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
//         wrkmemaux, wrkmem2, tfmmem);
// }



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage1_findfrag: search for superposition between multiple molecules 
// simultaneously and identify fragments for further refinement;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
// void stage1::stage1_findfrag2(
//     cudaStream_t streamproc,
//     int stepinit,
//     const uint maxnsteps,
//     const uint /*minfraglen*/,
//     uint nqystrs, uint ndbCstrs,
//     uint /*nqyposs*/, uint ndbCposs,
//     uint qystr1len, uint dbstr1len,
//     uint qystrnlen, uint dbstrnlen,
//     uint /*dbxpad*/,
//     float* __restrict__ tmpdpdiagbuffers,
//     float* __restrict__ /*wrkmem*/,
//     float* __restrict__ wrkmemaux,
//     float* __restrict__ /*wrkmem2*/,
//     float* __restrict__ /*wrkmemtm*/)
// {
//     const int symmetric = CLOptions::GetC_SYMMETRIC();
//     //NOTE: minlen, minimum of the largest structures to compare, assumed >=3
//     //minimum length among largest
//     int minlenmax = myhdmin(qystr1len, dbstr1len);
//     //minimum length among smallest
//     int minlenmin = myhdmin(qystrnlen, dbstrnlen);
//     int minalnmin = myhdmax(minlenmin >> 1, 5);
//     int n1 = minalnmin - dbstr1len;
//     int n2 = qystr1len - minalnmin;

//     uint maxnsteps1 = 0;

//     //launch kernels to process a bulk of maxnsteps fragments over all query-reference
//     //pairs in the chunk:
//     for(; n1 <= n2; n1 += stepinit * (int)maxnsteps)
//     {
//         //reduce #thread blocks to be launched if maxnsteps implies exceeding the limit
//         int nlocsteps = 
//             (n1 + stepinit * (int)maxnsteps <= n2)? maxnsteps: (n2-n1)/stepinit + 1;

//         if(!maxnsteps1) maxnsteps1 = nlocsteps;

//         //execution configuration for complete fragment identification:
//         //block processes one subfragment of one query-reference pair:
//         //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
//         dim3 nthrds_ficmpl(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
//         dim3 nblcks_ficmpl(ndbCstrs, nlocsteps, nqystrs);

//         if(symmetric)
//             FindGaplessAlignedFragment<true/*TFM_DINV*/>
//                 <<<nblcks_ficmpl,nthrds_ficmpl,0,streamproc>>>(
//                     ndbCstrs, ndbCposs, maxnsteps, n1/*arg1*/, stepinit/*arg2*/, 0/*arg3*/,
//                     tmpdpdiagbuffers, wrkmemaux);
//         else
//             FindGaplessAlignedFragment<false/*TFM_DINV*/>
//                 <<<nblcks_ficmpl,nthrds_ficmpl,0,streamproc>>>(
//                     ndbCstrs, ndbCposs, maxnsteps, n1/*arg1*/, stepinit/*arg2*/, 0/*arg3*/,
//                     tmpdpdiagbuffers, wrkmemaux);
//     }

//     //execution configuration for finding the maximum among scores 
//     //calculated for each fragment factor:
//     //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
//     dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
//     dim3 nblcks_scmax(
//         (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
//         nqystrs, 1);

//     SaveBestScoreAmongBests<<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
//         ndbCstrs,  maxnsteps, maxnsteps1,  wrkmemaux);
//     MYCUDACHECKLAST;
// }










// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage1_refinefrag: refine identified fragments to improve the scores of 
// superposition;
// CONDITIONAL, template parameter, flag of writing scores on ly if they're
// greater at the same location;
// SECONDARYUPDATE, indication of whether and how secondary update of best scores is done;
// MAX_NCHAINS, max #complex chains for the procedure to take effect;
// fragbydp, flag of whether fragments are identified by DP;
// nmaxconvit, maximum number of iterations for convergence check;
// qycpx1len, dbcpx1len, lengths of the largest query & reference complexes;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<
    bool CONDITIONAL,
    int SECONDARYUPDATE,
    int MAX_NCHAINS>
void stage1_complex::stage1_refinefrag_complex(
    std::map<CGKey,MyCuGraph>& /* stgraphs */,
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
    float* __restrict__ /* wrkmem */,
    float* __restrict__ /* wrkmemccd */,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ /* wrkmem2 */,
    float* __restrict__ tfmmem)
{
    stage1_refinefrag_helper2_complex
        <CONDITIONAL,SECONDARYUPDATE,MAX_NCHAINS>(
            fragbydp, nmaxconvit, streamproc,
            maxnsteps, minfraglen, nqycpxs, ndbCcpxs, nqystrs, ndbCstrs,
            nqyposs, ndbCposs, qycpx1len, dbcpx1len, qystr1len, dbstr1len,
            qystrnlen, dbstrnlen, dbxpad,
            tmpdpdiagbuffers, tmpdpalnpossbuffer,
            wrkmemtm, wrkmemtmibest, wrkmemaux, tfmmem);
}

// Instantiations
// 
#define INSTANTIATE_stage1_complex__stage1_refinefrag_complex( \
        tpCONDITIONAL,SECONDARYUPDATE,tpMAX_NCHAINS) \
    template void stage1_complex::stage1_refinefrag_complex \
        <tpCONDITIONAL,SECONDARYUPDATE,tpMAX_NCHAINS>( \
            std::map<CGKey,MyCuGraph>& stgraphs, \
            const int fragbydp, \
            const int nmaxconvit, \
            cudaStream_t streamproc, \
            const uint maxnsteps, \
            const uint minfraglen, \
            uint nqycpxs, uint ndbCcpxs, \
            uint nqystrs, uint ndbCstrs, \
            uint nqyposs, uint ndbCposs, \
            uint qycpx1len, uint dbcpx1len, \
            uint qystr1len, uint dbstr1len, \
            uint qystrnlen, uint dbstrnlen, \
            uint dbxpad, \
            float* __restrict__ tmpdpdiagbuffers, \
            float* __restrict__ tmpdpalnpossbuffer, \
            float* __restrict__ wrkmem, \
            float* __restrict__ wrkmemccd, \
            float* __restrict__ wrkmemtm, \
            float* __restrict__ wrkmemtmibest, \
            float* __restrict__ wrkmemaux, \
            float* __restrict__ wrkmem2, \
            float* __restrict__ tfmmem);

INSTANTIATE_stage1_complex__stage1_refinefrag_complex(true,SECONDARYUPDATE_NOUPDATE,0);
INSTANTIATE_stage1_complex__stage1_refinefrag_complex(false,SECONDARYUPDATE_NOUPDATE,0);
INSTANTIATE_stage1_complex__stage1_refinefrag_complex(false,SECONDARYUPDATE_UNCONDITIONAL,0);
INSTANTIATE_stage1_complex__stage1_refinefrag_complex(false,SECONDARYUPDATE_CONDITIONAL,0);

INSTANTIATE_stage1_complex__stage1_refinefrag_complex(false,SECONDARYUPDATE_NOUPDATE,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);

// -------------------------------------------------------------------------
// stage1_refinefrag_helper2_complex: refine identified fragments to 
// improve the superposition score; complete version;
// CONDITIONAL, template parameter, flag of writing scores on ly if they're
// greater at the same location;
// SECONDARYUPDATE, indication of whether and how secondary update of best scores is done;
// MAX_NCHAINS, max #complex chains for the procedure to take effect;
// fragbydp, flag of whether fragments are identified by DP;
// nmaxconvit, maximum number of iterations for convergence check;
// qycpx1len, dbcpx1len, lengths of the largest query & reference complexes;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<
    bool CONDITIONAL,
    int SECONDARYUPDATE,
    int MAX_NCHAINS>
void stage1_complex::stage1_refinefrag_helper2_complex(
    const int fragbydp,
    const int nmaxconvit,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqycpxs, uint ndbCcpxs,
    uint /* nqystrs */, uint ndbCstrs,
    uint /* nqyposs */, uint ndbCposs,
    uint qycpx1len, uint dbcpx1len,
    uint /*qystr1len*/, uint /*dbstr1len*/,
    uint /*qystrnlen*/, uint /*dbstrnlen*/,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    static std::string preamb = "stage1_complex::stage1_refinefrag_helper2_complex: ";
    const int symmetric = CLOptions::GetC_SYMMETRIC();
    //NOTE: minimum of the largest structures to compare is assumed >=3
    //minimum length among largest
    int minlenmax = myhdmin(qycpx1len, dbcpx1len);
    //maximum alignment length can be minimum of the lengths
    int maxalnmax = minlenmax;

    //maximum number of subdivisions of identified fragments
    // (for all structures in the chunk)
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    const int sfragstep = FRAGREF_SFRAGSTEP;
    //NOTE: frag info is written only at the stage of final refinement
    //(before production); this saves a lot of simple writes to gmem!
    constexpr bool WRITEFRAGINFO = false;//true

    if(fragbydp == stg1REFINE_INITIAL ||
       //NOTE: assignment can write scores to the same section: initialize:
       fragbydp == stg1REFINE_INITIAL_DP)
    {
        //execution configuration for scores initialization:
        //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
        dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
        dim3 nblcks_scinit(
            (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
            nqycpxs, maxnsteps);

        //initialize memory for best scores only;
        InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
        MYCUDACHECKLAST;
    }

//     //reset convergence flag; NOTE: unused; ensure reset
//     InitScores<INITOPT_CONVFLAG_FRAGREF>
//         <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
//         ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
//     MYCUDACHECKLAST;

    uint nlocsteps = 0;
    nlocsteps = (uint)myhdmax(1, GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen));
    nlocsteps *= (uint)nmaxsubfrags;//total number across all fragment lengths

    // if(nlocsteps < 1 || (int)maxnsteps < nlocsteps)
    //     throw MYRUNTIME_ERROR(preamb +
    //     "Invalid number of superposition tests: "+std::to_string(nlocsteps));



    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_arcmpl(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    dim3 nblcks_arcmpl(ndbCcpxs, myhdmin(nlocsteps, maxnsteps), nqycpxs);

    if(fragbydp == stg1REFINE_INITIAL) {
        if(symmetric)
            FragmentBasedAlignmentRefinement<WRITEFRAGINFO,true/*TFM_DINV*//*,true*//*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    nmaxconvit, ndbCstrs, ndbCposs,
                    nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                    tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FragmentBasedAlignmentRefinement<WRITEFRAGINFO,false/*TFM_DINV*//*,true*//*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    nmaxconvit, ndbCstrs, ndbCposs,
                    nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                    tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    } else if(fragbydp == stg1REFINE_INITIAL_DP) {
        if(symmetric)
            FragmentBasedDPAlignmentRefinement
                <WRITEFRAGINFO,CONDITIONAL,true/*TFM_DINV*/,true/*COMPLEX*/,MAX_NCHAINS>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    false/*readlocalconv*/,
                    nmaxconvit, ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FragmentBasedDPAlignmentRefinement
                <WRITEFRAGINFO,CONDITIONAL,false/*TFM_DINV*/,true/*COMPLEX*/,MAX_NCHAINS>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    false/*readlocalconv*/,
                    nmaxconvit, ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    } else if(fragbydp == stg1REFINE_ITERATIVE_DP) {
        if(symmetric)
            FragmentBasedDPAlignmentRefinement
                <WRITEFRAGINFO,CONDITIONAL,true/*TFM_DINV*/,true/*COMPLEX*/,MAX_NCHAINS>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    true/*readlocalconv*/,
                    nmaxconvit, ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FragmentBasedDPAlignmentRefinement
                <WRITEFRAGINFO,CONDITIONAL,false/*TFM_DINV*/,true/*COMPLEX*/,MAX_NCHAINS>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    true/*readlocalconv*/,
                    nmaxconvit, ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    } else
        throw MYRUNTIME_ERROR(preamb +
        "Unknown alignment refinement strategy: "+std::to_string(fragbydp));



    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCcpxs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqycpxs, 1);

    SaveBestScoreAndTMAmongBests
        <WRITEFRAGINFO,tawmvGrandBest,false/*FORCEWRITEFRAGINFO*/,SECONDARYUPDATE>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCcpxs,  maxnsteps, myhdmin(nlocsteps, maxnsteps),
            wrkmemtmibest, tfmmem, wrkmemaux,  wrkmemtm,
            ndbCstrs);
    MYCUDACHECKLAST;
}









// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage1_dprefine: refine ungapped alignment identified during fragment 
// boundary refinement in the previous substage by iteratively applying 
// (gapped) DP followed by the same ungapped alignment boundary refinement;
// GAP0, template parameter, flag of using gap cost 0;
// PRESCREEN, template parameter, whether to verify scores for pre-termination;
// WRKMEMTM1, use for the 1st iteration tfms saved in wrkmemtm;
// MAX_NCHAINS, max #complex chains for refinement to take effect;
// prescorethr, provisional TM-score threshold for prescreening;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<
    bool GAP0,
    bool PRESCREEN,
    bool WRKMEMTM1,
    int MAX_NCHAINS>
void stage1_complex::stage1_dprefine_complex(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const int maxndpiters,
    const float prescorethr,
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
    uint* __restrict__ /* maxscoordsbuf */,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    enum{ncosts = 1};
    static const float gcosts[ncosts] = {GAP0? 0.0f: -0.6f};

    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    constexpr bool vANCHORRGN = false;//using anchor region
    constexpr bool vBANDED = false;//banded alignment

    //execution configuration for the initialization of scores of complexes:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_cpxinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_cpxinit(
        (ndbCcpxs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, maxnsteps);

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, maxnsteps);

    //execution configuration for operating on scores at a fragment factor position of 0:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_cpxinit0(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_cpxinit0(
        (ndbCcpxs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, 1);

    //execution configuration for extracting matched positions
    //identified during DP:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);


    //initialize memory for best scores only;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;


    for(int gi = 0; gi < ncosts; gi++)
    {
        //reset the score convergence flag first;
        InitScores<INITOPT_CONVFLAG_FRAGREF|INITOPT_CONVFLAG_SCOREDP>
            <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
                ndbCstrs, maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
        MYCUDACHECKLAST;

        for(int dpi = 0; dpi < maxndpiters; dpi++)
        {
            RunDP<vANCHORRGN,vBANDED,GAP0,MAX_NCHAINS>(
                streamproc, gcosts[gi], maxnsteps, nqystrs, ndbCstrs, 
                nqyposs, ndbCposs, qystr1len, dbstr1len, dbxpad,
                tmpdpdiagbuffers, tmpdpbotbuffer, btckdata,
                ((WRKMEMTM1 && gi < 1 && dpi < 1)? wrkmemtm: wrkmemtmibest),
                wrkmemaux);

            //{{ASSIGNMENT
            //execution configuration for reformatting DP swift scores:
            //each block processes one query and CUS1_TBSP_DPSCORE_MAX_XDIM references:
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
                <false/*WRITESCORE*/, true/*WRITEASSG*/, false/*PASS2*/, MAX_NCHAINS>
                    <<<nblcks_ch2ch,nthrds_ch2ch,szdsmem_ch2ch,streamproc>>>(
                        nqystrs, ndbCstrs, ndbCcpxs,  maxnqrychains, maxnrfnchains,  maxnsteps, maxnstepsmem2,
                        NULL/*tfmmemory*/, wrkmemaux, wrkmem2);
            MYCUDACHECKLAST;
            //}}

            //{{COMBINED ALIGNMENT
            //proces the result of DP;
            //stepnumber==0: write aligned positions at slot 0:
            BtckToMatched32x
                <vANCHORRGN,vBANDED,true/*COMPLEX*/,MAX_NCHAINS>
                <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                    ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                    btckdata, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
                    stg1_cpx_complexstepnumber,
                    maxnstepsmem2/*assgmaxnsteps*/,
                    (0)/*assgstepnumber*/);
            MYCUDACHECKLAST;

            //execution configuration for reformating match for complexes:
            dim3 nthrds_rfmtmtch(CUDP_REFORMAT_MATCHED_DIM_X,CUDP_REFORMAT_MATCHED_DIM_Y,1);
            dim3 nblcks_rfmtmtch(ndbCcpxs,nqycpxs,1);

            //reformat match;
            ReformatMatched4Complexes<MAX_NCHAINS>
                <<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
                    ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                    maxnstepsmem2/*assgmaxnsteps*/,
                    (0)/*assgstepnumber*/,
                    stg1_cpx_complexstepnumber,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer);
            MYCUDACHECKLAST;
            //}}


            stage1_refinefrag_complex
                <false/*CONDITIONAL*/,SECONDARYUPDATE_NOUPDATE,MAX_NCHAINS>(
                    stgraphs,
                    stg1REFINE_ITERATIVE_DP/*fragments identified by DP*/,
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


            if(0 < dpi)
                CheckScoreConvergence<<<nblcks_cpxinit,nthrds_cpxinit,0,streamproc>>>(
                    ndbCcpxs, maxnsteps, wrkmemaux, ndbCstrs);
            MYCUDACHECKLAST;


            if(dpi+1 < maxndpiters)
                SaveLastScore0<<<nblcks_cpxinit0,nthrds_cpxinit0,0,streamproc>>>(
                    ndbCcpxs, maxnsteps, wrkmemaux, ndbCstrs);
            MYCUDACHECKLAST;

            if(PRESCREEN && maxndpiters <= dpi+1 && 0.0f < prescorethr)
                SetLowScoreConvergenceFlag<true/*COMPLEX*/>
                    <<<nblcks_cpxinit0,nthrds_cpxinit0,0,streamproc>>>(
                        prescorethr, ndbCcpxs, maxnsteps, wrkmemaux, ndbCstrs);
            MYCUDACHECKLAST;
        }
    }

    //reset the score convergence flag for the steps to follow not to halt
    InitScores<INITOPT_CONVFLAG_SCOREDP>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, maxnsteps, minfraglen, false/*checkfragos*/, wrkmemaux);
    MYCUDACHECKLAST;
}

// Instantiations
// 
#define INSTANTIATE_stage1_complex__stage1_dprefine_complex( \
        tpGAP0,tpPRESCREEN,tpWRKMEMTM1,tpMAX_NCHAINS) \
    template void stage1_complex::stage1_dprefine_complex \
        <tpGAP0,tpPRESCREEN,tpWRKMEMTM1,tpMAX_NCHAINS>( \
            std::map<CGKey,MyCuGraph>& stgraphs, \
            cudaStream_t streamproc, \
            const int maxndpiters, \
            const float prescore, \
            const uint maxnsteps, \
            const uint minfraglen, \
            uint nqycpxs, uint ndbCcpxs, \
            uint nqystrs, uint ndbCstrs, \
            uint nqyposs, uint ndbCposs, \
            uint qycpx1len, uint dbcpx1len, \
            uint qystr1len, uint dbstr1len, \
            uint qystrnlen, uint dbstrnlen, \
            uint dbxpad, const uint maxnqrychains, const uint maxnrfnchains, \
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
            float* __restrict__ tfmmem);

INSTANTIATE_stage1_complex__stage1_dprefine_complex(false,true,true,0);
INSTANTIATE_stage1_complex__stage1_dprefine_complex(false,false,true,0);
INSTANTIATE_stage1_complex__stage1_dprefine_complex(false,false,false,0);
INSTANTIATE_stage1_complex__stage1_dprefine_complex(true,false,false,0);
INSTANTIATE_stage1_complex__stage1_dprefine_complex(true,true,false,0);

INSTANTIATE_stage1_complex__stage1_dprefine_complex(true,false,false,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);

// -------------------------------------------------------------------------
// RunDP: run thread block-intensive DP;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
//
template<
    bool ANCHORRGN,
    bool BANDED,
    bool GAP0,
    int MAX_NCHAINS>
void stage1_complex::RunDP(
    cudaStream_t streamproc,
    const float gapcost,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    //execution configuration for DP:
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);
    //number of regular DIAGONAL block diagonal series;
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;

    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPwBtck3264x
            <ANCHORRGN,BANDED,GAP0,D02IND_SEARCH,
             false/*ALTSCTMS*/,true/*COMPLEX*/,MAX_NCHAINS>
                <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                    d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                    gapcost,
                    wrkmemtmibest/*in*/, wrkmemaux,
                    tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
                    //NOTE: all-vs-all chain alignment using complex-level tfms when
                    //NOTE: assgmaxnsteps is not provided!
        MYCUDACHECKLAST;
    }
}
