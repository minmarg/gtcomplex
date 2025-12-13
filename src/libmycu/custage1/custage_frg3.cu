/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <math.h>

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

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_swift_scan.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage1_complex.cuh"

#include "libmycu/custgfrg/local_similarity02.cuh"
#include "libmycu/custgfrg/linear_scoring.cuh"
#include "libmycu/custgfrg/linear_scoring2.cuh"
#include "libmycu/custgfrg2/linear_scoring2_complete.cuh"

#include "libmycu/cudp/dpsslocal.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/dpw_score.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/cudp/reformatmatch.cuh"

#include "libmycu/cuassign/chainassign.cuh"

#include "custage_frg3.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagefrg3_score_based_on_fragmatching3: obtain alignments in a couple of 
// iterations for massive number of variants obtained by fragment matching 
// between query and reference structures; score them;
// maxnsteps, max #steps that can be executed in parallel for one 
// query-reference pair; it corresponds to different #variants for a pair;
// actualnsteps, actual #variants (steps);
//
void stagefrg3::stagefrg3_score_based_on_fragmatching3(
    const int depth,
    const int napiterations,
    const bool scoreachiteration,
    const bool dynamicorientation,
    const float thrsimilarityperc,
    const float thrscorefactor,
    cudaStream_t streamproc,
    const int maxfraglen,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint actualnsteps,
    const uint qystr1len, const uint dbstr1len,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    const float thrpercentage_chainoccupancy)
{
//     if(1)
        stagefrg3_score_based_on_fragmatching3_helper(
            depth,
            napiterations,
            scoreachiteration,
            dynamicorientation,
            thrsimilarityperc,
            thrscorefactor,
            streamproc,
            maxfraglen,
            qryfragfct, rfnfragfct, fragndx,
            maxnsteps, actualnsteps,
            qystr1len, dbstr1len,
            nqystrs, ndbCstrs, ndbCposs, dbxpad,
            tmpdpdiagbuffers, tmpdpalnpossbuffer, dpscoremtx,
            wrkmem, wrkmemtmibest, wrkmemaux, wrkmem2, wrkmemtm,
            thrpercentage_chainoccupancy
        );
}

// -------------------------------------------------------------------------
// stagefrg3_score_based_on_fragmatching3: helper method to obtain 
// alignments in a couple of iterations for massive number of variants 
// obtained by fragment matching between query and reference structures;
//
void stagefrg3::stagefrg3_score_based_on_fragmatching3_helper(
    const int depth,
    const int napiterations,
    const bool scoreachiteration,
    const bool dynamicorientation,
    const float thrsimilarityperc,
    const float thrscorefactor,
    cudaStream_t streamproc,
    const int maxfraglen,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint actualnsteps,
    const uint qystr1len, const uint dbstr1len,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    const float thrpercentage_chainoccupancy)
{
    static const int windowsize = CLOptions::GetC_WINDOW();
    //max #aligned positions over the pairs in a chunk
    //int minlenmax = myhdmin(qystr1len, dbstr1len);
    //alignment length can be >min due to the linear algorithm:
    const int minlenmax = dbstr1len;
    const uint maxlenmax =
        //stack size depends on the length of the structure indexed:
        dynamicorientation? myhdmax(dbstr1len, qystr1len): qystr1len;


    //execution configuration for CCM initialization:
    //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
    dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
    dim3 nblcks_init(
        (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
        nqystrs, actualnsteps);

    //execution configuration for reduction:
    //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_ccmtx_frg(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
    dim3 nblcks_ccmtx_frg(
        (maxfraglen + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //execution configuration for reduction:
    //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_ccmtx(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
    dim3 nblcks_ccmtx(
        (minlenmax + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
    dim3 nblcks_copyto(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs, actualnsteps);

    //execution configuration for calculating transformation matrices:
    //each block processes one query and CUS1_TBSP_TFM_N references:
    dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
    dim3 nblcks_tfm(
        (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
        nqystrs, actualnsteps);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
    dim3 nblcks_copyfrom(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs, actualnsteps);

    //execution configuration for calculating provisional scores (reduction):
    //block processes one fragment-based configuration of a query-reference pair:
    // //NOTE: previous version for calculating scores from global dp matrices!
    // dim3 nthrds_locsim(CUSF_TBSP_LOCAL_SIMILARITY_XDIM,1,1);
    // dim3 nblcks_locsim(
    //     (ndbCstrs + CUSF_TBSP_LOCAL_SIMILARITY_XDIM - 1)/CUSF_TBSP_LOCAL_SIMILARITY_XDIM,
    //     actualnsteps, nqystrs);
    dim3 nthrds_locsim(CUSF_TBSP_LOCAL_SIMILARITY_XDIM,CUSF_TBSP_LOCAL_SIMILARITY_YDIM,1);
    dim3 nblcks_locsim(ndbCstrs, actualnsteps, nqystrs);

    //execution configuration for calculating provisional scores (reduction):
    //block processes one fragment-based configuration of a query-reference pair:
    dim3 nthrds_simeval(CUS1_TBSP_SCORE_FRG2_HALT_CHK_XDIM,1,1);
    dim3 nblcks_simeval(ndbCstrs, actualnsteps, nqystrs);


    //execution configuration for linearly finding best-matching coordinates at each position (no dp):
    //block processes CUSF_TBSP_INDEX_SCORE_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_linscos(CUSF_TBSP_INDEX_SCORE_XDIM,1,1);
    dim3 nblcks_linscos((myhdmin((uint)windowsize, dbstr1len) +
            CUSF_TBSP_INDEX_SCORE_XDIMLGL - 1)/CUSF_TBSP_INDEX_SCORE_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //dynamically determined stack size for the linscos kernel:
    uint stacksize_linscos = 1;
    if(0 < maxlenmax)
        stacksize_linscos = myhdmin((uint)17, (uint)ceilf(log2f(maxlenmax)) + 1);

    //size of dynamically allocted smem for the linscos kernel:
    uint szdsmem_linscos = sizeof(float) * (
#if (CUSF_TBSP_INDEX_SCORE_XFCT > 1)
        nTTranformMatrix +
#endif
        CUSF_TBSP_INDEX_SCORE_XDIM * stacksize_linscos * nStks_);

    MYCUDACHECK(cudaFuncSetAttribute(
        ProduceAlignmentUsingDynamicIndex2<0,1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, szdsmem_linscos));

    MYCUDACHECK(cudaFuncSetAttribute(
        ProduceAlignmentUsingDynamicIndex2<1,1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, szdsmem_linscos));


    //execution configuration for linearly drawing alignment (calculated before):
    //block processes CUSF2_TBSP_INDEX_ALIGNMENT_XDIMLGL positions of one query-reference pair:
    dim3 nthrds_linalign(CUSF2_TBSP_INDEX_ALIGNMENT_XDIM,CUSF2_TBSP_INDEX_ALIGNMENT_YDIM,1);
    dim3 nblcks_linalign((myhdmin((uint)windowsize, dbstr1len) +
            CUSF2_TBSP_INDEX_ALIGNMENT_XDIMLGL - 1)/CUSF2_TBSP_INDEX_ALIGNMENT_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);


    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, actualnsteps);

    //execution configuration for calculating scores (reduction):
    //block processes CUS1_TBSP_SCORE_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_scores(CUS1_TBSP_SCORE_XDIM,1,1);
    dim3 nblcks_scores(
        (minlenmax + CUS1_TBSP_SCORE_XDIMLGL - 1)/CUS1_TBSP_SCORE_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //execution configuration for saving best performing transformation matrices:
    //each block processes one query and CUS1_TBINITSP_TMSAVE_XFCT references:
    dim3 nthrds_savetm(CUS1_TBINITSP_TMSAVE_XDIM,1,1);
    dim3 nblcks_savetm(
        (ndbCstrs + CUS1_TBINITSP_TMSAVE_XFCT - 1)/CUS1_TBINITSP_TMSAVE_XFCT,
        nqystrs, actualnsteps);

    //execution configuration for minimum score reduction:
    //block processes all positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_findd2(CUS1_TBINITSP_FINDD02_ITRD_XDIM,1,1);
    dim3 nblcks_findd2(ndbCstrs, nqystrs, actualnsteps);


    //reset alignment lengths;
    InitScores<INITOPT_NALNPOSS|INITOPT_CONVFLAG_SCOREDP>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    stagefrg3_fragmatching3_Initial1(
        depth,
        dynamicorientation,
        thrsimilarityperc,
        thrscorefactor,
        streamproc,
        qryfragfct, rfnfragfct, fragndx,
        maxnsteps, actualnsteps,
        nqystrs, ndbCstrs, ndbCposs, dbxpad,
        dpscoremtx, wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
        nblcks_init, nthrds_init,
        nblcks_ccmtx_frg, nthrds_ccmtx_frg,
        nblcks_copyto, nthrds_copyto,
        nblcks_copyfrom, nthrds_copyfrom,
        nblcks_locsim, nthrds_locsim,
        nblcks_simeval, nthrds_simeval,
        nblcks_tfm, nthrds_tfm);

    for(int n = 0; n < napiterations; n++)
    {
        bool secstrmatchaln = (n+1 < napiterations);
        bool completealn = true;//(napiterations <= n+1);
        bool reversetfms = !scoreachiteration && (n+1 < napiterations);
        bool writeqrypss = !reversetfms;//write query positions

        stagefrg3_fragmatching3_alignment2(
            depth,
            dynamicorientation,
            thrsimilarityperc,
            secstrmatchaln,
            completealn,
            writeqrypss,
            streamproc,
            qryfragfct, rfnfragfct, fragndx,
            maxnsteps,
            nqystrs, ndbCstrs, ndbCposs, dbxpad,
            tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, wrkmemtm,
            nblcks_linscos, nthrds_linscos, szdsmem_linscos, stacksize_linscos,
            nblcks_linalign, nthrds_linalign);

        stagefrg3_fragmatching3_superposition3(
            dynamicorientation,
            reversetfms,
            streamproc,
            maxnsteps,
            nqystrs, ndbCstrs, ndbCposs, dbxpad,
            tmpdpalnpossbuffer,
            wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
            nblcks_init, nthrds_init,
            nblcks_ccmtx, nthrds_ccmtx,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);

        if(reversetfms) continue;

        stagefrg3_fragmatching3_aln_scoring4(
            streamproc,
            depth,
            maxnsteps,
            nqystrs, ndbCstrs, ndbCposs, dbxpad,
            qryfragfct, rfnfragfct,
            tmpdpdiagbuffers, tmpdpalnpossbuffer,
            wrkmem, wrkmemaux, wrkmem2, wrkmemtm, wrkmemtmibest,
            nblcks_init, nthrds_init,
            nblcks_ccmtx, nthrds_ccmtx,
            nblcks_findd2, nthrds_findd2,
            nblcks_scinit, nthrds_scinit,
            nblcks_scores, nthrds_scores,
            nblcks_savetm, nthrds_savetm,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm,
            thrpercentage_chainoccupancy);

        if(scoreachiteration && (n+1 < napiterations)) {
            //revert tfms for indexed alignment:
            stagefrg3_fragmatching3_superposition3(
                dynamicorientation,
                true/*reversetfms*/,
                streamproc,
                maxnsteps,
                nqystrs, ndbCstrs, ndbCposs, dbxpad,
                tmpdpalnpossbuffer,
                wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
                nblcks_init, nthrds_init,
                nblcks_ccmtx, nthrds_ccmtx,
                nblcks_copyto, nthrds_copyto,
                nblcks_copyfrom, nthrds_copyfrom,
                nblcks_tfm, nthrds_tfm);
        }
    }
}



// -------------------------------------------------------------------------
// stagefrg3_fragmatching3_Initial1: subiteration 1 of producing structure 
// alignments by fragment matching in several iterations;
//
inline
void stagefrg3::stagefrg3_fragmatching3_Initial1(
    const int depth,
    const bool dynamicorientation,
    const float thrsimilarityperc,
    const float /* thrscorefactor */,
    cudaStream_t streamproc,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps, const uint actualnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx_frg, const dim3& nthrds_ccmtx_frg,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_locsim, const dim3& nthrds_locsim,
    const dim3& /* nblcks_simeval */, const dim3& /* nthrds_simeval */,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    static const int complexes = 1; //consitent; (CLOptions::GetB_CHAINS() == 0);
    static const int seedapproachstruct = CLOptions::GetC_SeedRuleValue();
    // static const int windowsize = CLOptions::GetC_WINDOW();

    //execution configuration for reduction:
    dim3 nthrds_ccmtx_aln(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
    // dim3 nthrds_ccmtx_aln(CUSF_TBSP_INITIAL_ALN_CCM_XDIM,CUSF_TBSP_INITIAL_ALN_CCM_YDIM,1);
    dim3 nblcks_ccmtx_aln(ndbCstrs, actualnsteps, nqystrs);

    //NOTE: when thrsimilarityperc<=0, the convergence flag is distributed:
    if(complexes) CalcLocalSimilarity2_frg2<1><<<nblcks_locsim,nthrds_locsim,0,streamproc>>>(
        thrsimilarityperc, seedapproachstruct,
        (depth), ndbCstrs, ndbCposs, dbxpad, maxnsteps,
        qryfragfct, rfnfragfct, fragndx,
        dpscoremtx, wrkmemaux);
    else CalcLocalSimilarity2_frg2<<<nblcks_locsim,nthrds_locsim,0,streamproc>>>(
        thrsimilarityperc, seedapproachstruct,
        (depth), ndbCstrs, ndbCposs, dbxpad, maxnsteps,
        qryfragfct, rfnfragfct, fragndx,
        dpscoremtx, wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory for calculating cross covariance matrices
    InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, maxnsteps,  wrkmem, wrkmemaux);
    // InitCCData0_frg2<<<nblcks_init,nthrds_init,0,streamproc>>>(
    //     (depth),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,  wrkmem);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    if(seedapproachstruct) {
        if(complexes) CalcCCMatrices64LocallyAligned64<1><<<nblcks_ccmtx_aln,nthrds_ccmtx_aln,0,streamproc>>>(
            seedapproachstruct, (depth), (thrsimilarityperc), ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            qryfragfct, rfnfragfct, dpscoremtx, wrkmemaux, wrkmem);
        else CalcCCMatrices64LocallyAligned64<0><<<nblcks_ccmtx_aln,nthrds_ccmtx_aln,0,streamproc>>>(
            seedapproachstruct, (depth), (thrsimilarityperc), ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            qryfragfct, rfnfragfct, dpscoremtx, wrkmemaux, wrkmem);
    } else {
        if(complexes) CalcCCMatrices64_frg2<1><<<nblcks_ccmtx_frg,nthrds_ccmtx_frg,0,streamproc>>>(
            (depth),  nqystrs, ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
            wrkmemaux, wrkmem);
        else CalcCCMatrices64_frg2<<<nblcks_ccmtx_frg,nthrds_ccmtx_frg,0,streamproc>>>(
            (depth),  nqystrs, ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
            wrkmemaux, wrkmem);
    }
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable efficient 
    //structure-specific calculation:
    if(seedapproachstruct) {
        if(complexes) CopyCCDataToWrkMem2_frg2<READNPOS_READ,1><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            seedapproachstruct, (depth),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
            wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
        else CopyCCDataToWrkMem2_frg2<READNPOS_READ,0><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            seedapproachstruct, (depth),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
            wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    } else {
        if(complexes) CopyCCDataToWrkMem2_frg2<READNPOS_NOREAD,1><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            seedapproachstruct, (depth),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
            wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
        else CopyCCDataToWrkMem2_frg2<READNPOS_NOREAD,0><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            seedapproachstruct, (depth),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
            wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    }
    MYCUDACHECKLAST;

    if(dynamicorientation) {
        if(complexes) CalcTfmMatrices_DynamicOrientation<1><<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
            ndbCstrs, maxnsteps, wrkmem2);
        else CalcTfmMatrices_DynamicOrientation<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
            ndbCstrs, maxnsteps, wrkmem2);
    } else {
        //NOTE: calculate transformation matrices reversed wrt query-reference structure pair
        if(complexes) CalcTfmMatrices<TFMTX_REVERSE_TRUE,false/*TFM_DINV*/,1>
            <<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(ndbCstrs, maxnsteps, wrkmem2);
        else CalcTfmMatrices<TFMTX_REVERSE_TRUE><<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
            ndbCstrs, maxnsteps, wrkmem2);
    }
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;

    //NOTE: previous version of screening for promising initial superpositions by
    //NOTE: calculating provisional scores; that does not work as intended;
    // CalcScoresUnrl_frg2<<<nblcks_simeval,nthrds_simeval,0,streamproc>>>(
    //     thrscorefactor, dynamicorientation, windowsize,
    //     (depth), ndbCstrs, maxnsteps, qryfragfct, rfnfragfct, fragndx,
    //     wrkmemtm/*in*/, wrkmemaux);
    // MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefrg3_fragmatching3_alignment2: subiteration 2 of producing structure 
// alignments by fragment matching in several iterations;
//
inline
void stagefrg3::stagefrg3_fragmatching3_alignment2(
    const int depth,
    const bool dynamicorientation,
    const float thrsimilarityperc,
    const bool secstrmatchaln,
    const bool complete,
    const bool writeqrypss,
    cudaStream_t streamproc,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint /*dbxpad*/,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_linscos, const dim3& nthrds_linscos,
        const uint szdsmem_linscos, const uint stacksize_linscos,
    const dim3& nblcks_linalign, const dim3& nthrds_linalign)
{
    static const int complexes = 1; //consitent; (CLOptions::GetB_CHAINS() == 0);
    static const int seedapproachstruct = CLOptions::GetC_SeedRuleValue();
    static const int windowsize = CLOptions::GetC_WINDOW();

    if(dynamicorientation) {
        if(secstrmatchaln) {
            if(complexes) ProduceAlignmentUsingDynamicIndex2<1/*SECSTRFILT*/,1/*COMPLEX*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, windowsize,
                    seedapproachstruct, (depth), (thrsimilarityperc),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
            else ProduceAlignmentUsingDynamicIndex2<1/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, windowsize,
                    seedapproachstruct, (depth), (thrsimilarityperc),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        } else {
            if(complexes) ProduceAlignmentUsingDynamicIndex2<0/*SECSTRFILT*/,1/*COMPLEX*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, windowsize,
                    seedapproachstruct, (depth), (thrsimilarityperc),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
            else ProduceAlignmentUsingDynamicIndex2<0/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, windowsize,
                    seedapproachstruct, (depth), (thrsimilarityperc),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        }
        MYCUDACHECKLAST;
    }
    else if(complete) {
        if(secstrmatchaln)
            ProduceAlignmentUsingIndex2<1/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, windowsize, (depth), (thrsimilarityperc),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        else
            ProduceAlignmentUsingIndex2<0/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, windowsize, (depth), (thrsimilarityperc),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        MYCUDACHECKLAST;
    } else {
        //calculate positional best-matching atoms using index trees
        if(secstrmatchaln)
            PositionalCoordsFromIndexLinear2<1/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, windowsize, (depth),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,
                    qryfragfct, rfnfragfct, fragndx,
                    wrkmemtm, wrkmemaux, tmpdpalnpossbuffer);
        else
            PositionalCoordsFromIndexLinear2<0/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, windowsize, (depth),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,
                    qryfragfct, rfnfragfct, fragndx,
                    wrkmemtm, wrkmemaux, tmpdpalnpossbuffer);
        MYCUDACHECKLAST;

        //draw alignment based on best-matching atom pairs
        MakeAlignmentLinear2<<<nblcks_linalign,nthrds_linalign,0,streamproc>>>(
                complete, windowsize, (depth), nqystrs, ndbCstrs, ndbCposs,  maxnsteps,
                qryfragfct, rfnfragfct, fragndx,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        MYCUDACHECKLAST;
    }
}

// -------------------------------------------------------------------------
// stagefrg3_fragmatching3_superposition3: subiteration 3 of obtaining 
// superpositions based on resulting alignments;
//
inline
void stagefrg3::stagefrg3_fragmatching3_superposition3(
    const bool dynamicorientation,
    const bool reversetfms,
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
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    static const int complexes = 1; //consitent; (CLOptions::GetB_CHAINS() == 0);

    //initialize memory for calculating cross covariance matrices
    InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, maxnsteps,  wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    if(complexes) CalcCCMatrices64_SWFTscan<1><<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
        wrkmemaux, tmpdpalnpossbuffer, wrkmem);
    else CalcCCMatrices64_SWFTscan<<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
        wrkmemaux, tmpdpalnpossbuffer, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable efficient 
    //structure-specific calculation; READNPOS_NOREAD, do not verify whether
    //#positions on which tfms are calculated has changed:
    CopyCCDataToWrkMem2_SWFTscan<READNPOS_NOREAD>
        <<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    if(reversetfms) {
        if(dynamicorientation) {
            if(complexes) CalcTfmMatrices_DynamicOrientation<1><<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
                    ndbCstrs, maxnsteps, wrkmem2);
            else CalcTfmMatrices_DynamicOrientation<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
                    ndbCstrs, maxnsteps, wrkmem2);
        } else {
            if(complexes) CalcTfmMatrices<TFMTX_REVERSE_TRUE,false/*TFM_DINV*/,1>
                <<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(ndbCstrs, maxnsteps, wrkmem2);
            else CalcTfmMatrices<TFMTX_REVERSE_TRUE><<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
                    ndbCstrs, maxnsteps, wrkmem2);
        }
    } else {
        if(complexes) CalcTfmMatrices<TFMTX_REVERSE_FALSE/*REVERSE*/,false/*TFM_DINV*/,1>
            <<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(ndbCstrs, maxnsteps, wrkmem2);
        else CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
            ndbCstrs, maxnsteps, wrkmem2);
    }
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefrg3_fragmatching3_aln_scoring4: subiteration 4 of scoring 
// alignments;
//
inline
void stagefrg3::stagefrg3_fragmatching3_aln_scoring4(
    cudaStream_t streamproc,
    const int depth,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    const int qryfragfct, const int rfnfragfct,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ /*wrkmem2*/,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    //
    const dim3& /*nblcks_init*/, const dim3& /*nthrds_init*/,
    const dim3& /*nblcks_ccmtx*/, const dim3& /*nthrds_ccmtx*/,
    const dim3& /*nblcks_findd2*/, const dim3& /*nthrds_findd2*/,
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_scores, const dim3& nthrds_scores,
    const dim3& nblcks_savetm, const dim3& nthrds_savetm,
    const dim3& /*nblcks_copyto*/, const dim3& /*nthrds_copyto*/,
    const dim3& /*nblcks_copyfrom*/, const dim3& /*nthrds_copyfrom*/,
    const dim3& /*nblcks_tfm*/, const dim3& /*nthrds_tfm*/,
    const float thrpercentage_chainoccupancy)
{
    static const int complexes = 1; //consitent; (CLOptions::GetB_CHAINS() == 0);
    static const int seedapproachstruct = CLOptions::GetC_SeedRuleValue();
    static const int windowsize = CLOptions::GetC_WINDOW();

    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores and temporarily save distances
    if(complexes) CalcScoresUnrl_SWFTscanProgressiveComplex
        <SAVEPOS_NOSAVE/*SAVEPOS_SAVE*/,CHCKALNLEN_NOCHECK>
        <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
            thrpercentage_chainoccupancy, windowsize,
            seedapproachstruct, depth, nqystrs, ndbCstrs, ndbCposs, dbxpad,
            maxnsteps,  qryfragfct, rfnfragfct,
            tmpdpalnpossbuffer, wrkmemtm, wrkmem, wrkmemaux, tmpdpdiagbuffers);
    else CalcScoresUnrl_SWFTscanProgressive<SAVEPOS_NOSAVE/*SAVEPOS_SAVE*/,CHCKALNLEN_NOCHECK>
        <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
            tmpdpalnpossbuffer, wrkmemtm, wrkmem, wrkmemaux, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    //save scores and tfms
    SaveBestScoreAndTM<false/*WRITEFRAGINFO*/>
        <<<nblcks_savetm,nthrds_savetm,0,streamproc>>>(
            ndbCstrs,  maxnsteps, 0/*sfragstep(unused)*/,
            wrkmemtm, wrkmemtmibest, wrkmemaux);
    MYCUDACHECKLAST;
}










// -------------------------------------------------------------------------
// prepare_for_index_stage_complex: prepare relative data for the spatial
// index-based stage;
//
void stagefrg3::prepare_for_index_stage_complex(
    cudaStream_t streamproc,
    const float prescore,
    const uint maxnsteps,
    const uint /* minfraglen */,
    uint nqycpxs, uint ndbCcpxs,
    uint /* nqystrs */, uint ndbCstrs,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ /* wrkmemtm */,
    float* __restrict__ /* wrkmemtmibest */,
    float* __restrict__ wrkmemaux)
{
    //execution configuration for checking scores:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_cpxconvinit(CUS1_CHOR_CONV_SCORE_XDIM,1,1);
    dim3 nblcks_cpxconvinit(1, 1, 1);

    // if(0.0f < prescore) {
        //NOTE: call the function irrespective of prescore to copy grands!
        //NOTE: convergence is considered only when at least one of complexes
        //NOTE: corresponds to a single chain!
        SetLowScoreConvergenceFlagComplexInitial<CUS1_TBSP_CPXSCORE_MAX_NCHAINS_FILTER>
            <<<nblcks_cpxconvinit,nthrds_cpxconvinit,0,streamproc>>>(
                prescore, nqycpxs, ndbCcpxs, maxnsteps, wrkmemaux, tmpdpdiagbuffers, ndbCstrs);
        MYCUDACHECKLAST;
    // }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stagefrg3_complexes: search for superposition between multiple complexes 
// simultaneously by exhaustively matching their fragments for similarity;
// qycpx1len, length of the largest query complex;
// dbcpx1len, length of the largest reference complex;
// qystr1len, length of the largest query chain;
// dbstr1len, length of the largest reference chain;
//
void stagefrg3::run_stagefrg3_complexes(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const int stage3asfilter,
    const int depth,
    const int nbranches,
    const int simthreshold,
    const int maxndpiters,
    const uint maxnsteps,
    const uint minfraglen,
    const float /*scorethld*/,
    const float prescore,
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
    float* __restrict__ wrkmemtmalt,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem,
    uint* __restrict__ /*globvarsbuf*/)
{
    MYMSG("stagefrg3::run_stagefrg3_complexes", 4);


    const int addchainlevelsearch = CLOptions::GetC_ADDCHAINLEVELSEARCH();

    //execution configuration for tfm matrix initialization:
    dim3 nthrds_tfminit(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    dim3 nblcks_tfminit(
        (ndbCcpxs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
        nqycpxs, maxnsteps);

    //initialize transformation matrices;
    InitTfmMatrices<<<nblcks_tfminit,nthrds_tfminit,0,streamproc>>>(
       ndbCcpxs, maxnsteps, 0/*minfraglen(unused)*/, FRAGREF_SFRAGSTEP, false/*checkfragos*/,
       wrkmemtmibest);
    MYCUDACHECKLAST;

    if(addchainlevelsearch)
        prepare_for_index_stage_complex(
            streamproc, prescore,
            maxnsteps, minfraglen,
            nqycpxs, ndbCcpxs,
            nqystrs, ndbCstrs,
            tmpdpdiagbuffers, wrkmemtm, wrkmemtmibest, wrkmemaux);

    //clear scores (tawmvScore) for following stages
    //execution configuration for scores initialization:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCcpxs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, maxnsteps);

    InitScores<
        INITOPT_BEST|INITOPT_CURRENT|INITOPT_QRYRFNPOS|INITOPT_FRAGSPECS|INITOPT_NALNPOSS>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCcpxs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,
            wrkmemaux);
    MYCUDACHECKLAST;

    dim3 nthrds_cvinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_cvinit(
        (ndbCcpxs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, 1);

    SetConvergenceForUnmatchedTypesComplex
        <<<nblcks_cvinit,nthrds_cvinit,0,streamproc>>>(
            ndbCcpxs, ndbCcpxs/*for ndbCstrs*/, maxnsteps, wrkmemaux);
    MYCUDACHECKLAST;


    stagefrg3_extensive_frg_swift_complexes(
        streamproc,
        depth,
        nbranches,
        simthreshold,
        prescore,
        maxnsteps,
        nqycpxs, ndbCcpxs,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qycpx1len, dbcpx1len,
        qystr1len, dbstr1len,
        dbxpad,
        maxnqrychains, maxnrfnchains,
        ((nbranches && maxndpiters)? WRKMEMTMALT: (maxndpiters? WRKMEMTMIBEST: TFMMEM))/*memtype*/,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, btckdata,
        wrkmem, wrkmemccd,
        ((nbranches && maxndpiters)? wrkmemtmalt: (maxndpiters? wrkmemtmibest: tfmmem))/*out*/,
        wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem);

    //TODO: implement filtering out references here!
    //TODO: this will significantly reduce the runtime of speed 7 through 14!

    if(stage3asfilter) return;

    //process top N best-performing tfms found from the extensive 
    //application of spatial index:
    if(nbranches && maxndpiters)
        stagefrg3_refinement_tfmaltconfig_complexes(
            stgraphs, streamproc, depth, nbranches, maxnsteps, minfraglen,
            nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs,
            qycpx1len, dbcpx1len, qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
            tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, btckdata,
            wrkmem, wrkmemccd, wrkmemtmalt, wrkmemtm, wrkmemtmibest,
            wrkmemaux, wrkmem2, tfmmem);

    //refine alignment boundaries identified in the previous 
    //substage by applying DP;
    //1. With a gap cost:
    if(nbranches && maxndpiters)
        stage1_dprefine_complex<false/*GAP0*/,false/*PRESCREEN*/,true/*WRKMEMTM1*/>(
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

    //2. No gap cost:
    if(maxndpiters)
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



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagefrg3_extensive_frg_swift_complexes: calculate approximate tmscores 
// and find most favorable initial superposition based on fragment 
// matching of multiple query and reference complexes;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefrg3::stagefrg3_extensive_frg_swift_complexes(
    cudaStream_t streamproc,
    const int depth,
    const int nbranches,
    const int simthreshold,
    const float prescore,
    const uint maxnsteps,
    uint nqycpxs, uint ndbCcpxs,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qycpx1len, uint dbcpx1len,
    uint qystr1len, uint dbstr1len,
    uint dbxpad,
    const uint maxnqrychains, const uint maxnrfnchains,
    const int tfmtarget,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* /*__restrict__*/ wrkmemtmalt,
    float* __restrict__ wrkmemtm,
    float* /*__restrict__*/ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* /*__restrict__*/ /*tfmmem*/)
{
    MYMSG("stagefrg3::stagefrg3_extensive_frg_swift_complexes", 4);
    static const std::string preamb = "stagefrg3::stagefrg3_extensive_frg_swift_complexes: ";

    static const int addchainlevelsearch = CLOptions::GetC_ADDCHAINLEVELSEARCH();
    static const int seedapproachstruct = CLOptions::GetC_SeedRuleValue();
    static const int topndpscores = CLOptions::GetC_RECALCULATE();
    static const int windowsize = CLOptions::GetC_WINDOW();
    //#iterations on alignment pairs to perform excluding the
    //initial (tfm) and final (score) kernels: 1, 2, or 3:
    static const int napiterations = CLOptions::GetH_N_SPATIAL_ITERATIONS();
    static const bool scoreachiteration = false;
    //NOTE: dynamicorientation must always be set true for complexes!
    static const bool dynamicorientation = true;
    static const float thrpercentage_chainoccupancy = 0.2f;
    static const float thrsimilarityperc = (float)simthreshold * 0.01f;
    static const float thrscorefactor = 1.0f;
    static const float locgapcost = CLOptions::GetC_GapCost();

    enum{nfrags = 2};//number of fragments of different length used
    static const int frags[nfrags] = {20, 100};
    static const int fragndx = seedapproachstruct? 0: 1;
    //int fraglen = GetNAlnPoss_frg(
    //        qystr1len, dbstr1len, 0/*qrypos,unsed*/, 0/*rfnpos,unused*/,
    //        0/*qryfragfct,unsed*/, 0/*rfnfragfct,unused*/, 0/*fraglen index*/);

    const int fctdiv =
        (depth==CLOptions::csdShallow)? GetFragStepSize_frg_shallow_factor(): 1;
    const int minnsteps = 10 / fctdiv;
    int qrystepsz = GetFragStepSize_frg_shallow(qycpx1len);
    int rfnstepsz = GetFragStepSize_frg_shallow(dbcpx1len);
    if(depth == CLOptions::csdDeep) {
        qrystepsz = GetFragStepSize_frg_deep(qycpx1len);
        rfnstepsz = GetFragStepSize_frg_deep(dbcpx1len);
    } else if(depth == CLOptions::csdHigh) {
        qrystepsz = GetFragStepSize_frg_high(qycpx1len);
        rfnstepsz = GetFragStepSize_frg_high(dbcpx1len);
    } else if(depth == CLOptions::csdMedium) {
        qrystepsz = GetFragStepSize_frg_medium(qycpx1len);
        rfnstepsz = GetFragStepSize_frg_medium(dbcpx1len);
    }
    //set minimum #steps to 10 since length 150 leads to 150/15=10,
    //the largest among #steps for medium-sized structures:
    const int nstepsy = myhdmax(minnsteps, (int)qycpx1len/qrystepsz + 1);
    const int nstepsx = myhdmax(minnsteps, (int)dbcpx1len/rfnstepsz + 1);


    //configuration for DP SS: fill in DP matrices for local alignment
    //based on SS match; this will be used for screening for plausible
    //local similarities
    const uint maxblkdiagelems_ss = GetMaxBlockDiagonalElems(
            dbcpx1len, qycpx1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp_ss(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp_ss(maxblkdiagelems_ss,ndbCcpxs,nqycpxs);
    //number of regular DIAGONAL block diagonal series;
    uint nblkdiags_ss = (uint)
        (((dbcpx1len + qycpx1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    nblkdiags_ss += (uint)(qycpx1len - 1) / CUDP_2DCACHE_DIM_D;

    if(0.0f < thrsimilarityperc || seedapproachstruct) {
        //launch blocks along block diagonals to perform DP;
        //nblkdiags_ss, total number of diagonals:
        for(uint d = 0; d < nblkdiags_ss; d++)
        {
            ExecDPSSLocal3264x<1/*COMPLEX*/><<<nblcks_dp_ss,nthrds_dp_ss,0,streamproc>>>(
                d, ndbCcpxs, ndbCposs, dbxpad, maxnsteps, locgapcost,
                wrkmemaux/*cnv*/, tmpdpdiagbuffers/*wrk*/, tmpdpbotbuffer/*wrk*/,
                btckdata/*out*/);
            MYCUDACHECKLAST;
        }//dpss
    }


    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCcpxs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqycpxs, 1);

    //step number (<=maxnsteps) for efficiently launching numerous processing 
    //kernels and calculating scores on the query-reference dimensions:
    int stepnumber = 0;
    int ysndxproc = 0, xsndxproc = 0;//processed indices

    //there are fragment variants; divide max allowed accommodation by 2:
    const uint stepmult = seedapproachstruct? 0: 1;
    const uint maxnstepso2 = (maxnsteps >> stepmult);

    for(int ysndx = 0; ysndx < nstepsy; ysndx++)
    {
        //increase #step indices over query and reference structures;
        //maxnsteps is max allowed steps to be processed in parallel simultaneously
        for(int xsndx = 0; xsndx < nstepsx; xsndx++)
        {
            bool lastiteration = (nstepsy <= ysndx + 1) && (nstepsx <= xsndx + 1);

            stepnumber++;

            if(stepnumber < (int)maxnstepso2 && !lastiteration)
                continue;

            //wrkmemtmibest contains best tfms over all recurses
            stagefrg3_score_based_on_fragmatching3(
                depth,
                napiterations,
                scoreachiteration,
                dynamicorientation,
                thrsimilarityperc,
                thrscorefactor,
                streamproc,
                frags[fragndx],
                ysndxproc, xsndxproc, fragndx,
                maxnsteps, (stepnumber << stepmult),
                qycpx1len, dbcpx1len,
                nqycpxs, ndbCcpxs, ndbCposs, dbxpad,
                tmpdpdiagbuffers, tmpdpalnpossbuffer, btckdata,
                wrkmem, wrkmemtmibest, wrkmemaux, wrkmem2, wrkmemccd/*for wrkmemtm*/,
                thrpercentage_chainoccupancy
            );

            if(depth <= CLOptions::csdHigh) {
                //save top CUS1_TBSP_DPSCORE_TOP_N secondary linear-alignment-based superposition 
                //configurations for selecting the currently best order-dependent superposition
                SaveTopNScoresAndTMsAmongSecondaryBests<true/*COMPLEX*/>
                    <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
                        (topndpscores), (depth),
                        (ysndxproc == 0 && xsndxproc == 0),//firstit
                        (depth <= CLOptions::csdDeep),//twoconfs,
                        (xsndxproc),//rfnfragfctinit
                        ndbCcpxs, maxnsteps, (stepnumber << stepmult)/*effnsteps*/,
                        wrkmemtmibest/*in*/, wrkmemtm/*out*/, wrkmemaux,
                        ndbCstrs,
                        seedapproachstruct);
                MYCUDACHECKLAST;
            }

            stepnumber = 0;
            ysndxproc = ysndx;
            xsndxproc = ysndx * nstepsx + xsndx + 1;

        }//reference positions
    }//query positions


    //scores originating from spatial indexing are not optimal; alignments not refined;
    //multiply by a factor:
    const float approxthreshold = prescore * CLOptions::GetH_PRESCORE_FACTOR();

    //save top CUS1_TBSP_DPSCORE_TOP_N linear-alignment-based superposition 
    //configurations for selecting the currently best order-dependent superposition
    SaveTopNScoresAndTMsAmongBests<<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
        (topndpscores), ndbCcpxs,  maxnsteps,  wrkmemtmibest/*in*/, wrkmemtm/*out*/, wrkmemaux,
        tmpdpdiagbuffers,//temprorary buffer for flags and scores
        ndbCstrs,//tfms written using a consistent spacing of ndbCstrs
        windowsize, approxthreshold);
    MYCUDACHECKLAST;


    // //execution configuration for reformating flags:
    // //each block processes CUS1_TBSP_SCORE_MAX_XDIM references:
    // dim3 nthrds_rflgs(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_DPSCORE_TOP_N,1);
    // dim3 nblcks_rflgs(1, 1, 1);

    // //reformat flags to conform to the final complex format
    // ReformatTopNFlagsComplex<<<nblcks_rflgs,nthrds_rflgs,0,streamproc>>>(
    //         nqycpxs, ndbCcpxs, maxnsteps, wrkmemaux, ndbCstrs);
    // MYCUDACHECKLAST;

    //clear flags and scores for following stages;
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, maxnsteps);

    ConditionalInitToComplex<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmemaux);
    MYCUDACHECKLAST;

    //copy back flags and scores (saved in SaveTopNScoresAndTMsAmongBests);
    dim3 nthrds_scback(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scback(
        (ndbCcpxs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, topndpscores);

    ConditionalInitFromComplex<<<nblcks_scback,nthrds_scback,0,streamproc>>>(
        (topndpscores), addchainlevelsearch,
        ndbCcpxs, ndbCstrs, maxnsteps,  tmpdpdiagbuffers, wrkmemaux);
    MYCUDACHECKLAST;


    //configuration for swift DP: calculate optimal order-dependent scores:
    const uint nconfigsections = 
        (depth <= CLOptions::csdDeep)? 3: ((depth <= CLOptions::csdHigh)? 2: 1);
    const uint nconfigs = (topndpscores) * nconfigsections;
    const uint maxblkdiagelems_swft = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_SWFT_2DCACHE_DIM_D, CUDP_SWFT_2DCACHE_DIM_X);
    uint nqystrspart = myhdmin(nqystrs, (maxnsteps + nconfigs - 1) / nconfigs);
    dim3 nthrds_dp_swft(CUDP_SWFT_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp_swft(maxblkdiagelems_swft, ndbCstrs, nqystrspart * nconfigs);
    uint nblkdiags_swft = (uint)
        (((dbstr1len + qystr1len) + CUDP_SWFT_2DCACHE_DIM_X-1) / CUDP_SWFT_2DCACHE_DIM_X);
    nblkdiags_swft += (uint)(qystr1len - 1) / CUDP_SWFT_2DCACHE_DIM_D;

    for(uint q = 0; q < nqystrs; q += nqystrspart) {
        if(nqystrs < q + nqystrspart) nqystrspart = nqystrs - q;
        nblcks_dp_swft.z = nqystrspart * nconfigs;
        //launch blocks along block diagonals to perform DP;
        //nblkdiags_swft, total number of diagonals:
        for(uint d = 0; d < nblkdiags_swft; d++)
        {
            ExecDPScore3264x
                <false/*ANCHORRGN*/,false/*BANDED*/,true/*GAP0*/,true/*CHECKCONV*/,true/*COMPLEX*/>
                    <<<nblcks_dp_swft,nthrds_dp_swft,0,streamproc>>>(
                        d, nqystrspart, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0.0f/*gcost*/,
                        wrkmemtm, wrkmemaux,
                        //NOTE: bigger buffer, tmpdpalnpossbuffer, used instead of
                        //NOTE: tmpdpbotbuffer to fit data in memory!
                        tmpdpdiagbuffers, tmpdpalnpossbuffer/*tmpdpbotbuffer*/,
                        q, nconfigs);
            MYCUDACHECKLAST;
        }//swift_dp

        //execution configuration for reformatting DP swift scores:
        //each block processes one query and CUS1_TBSP_DPSCORE_MAX_XDIM references:
        dim3 nthrds_dpsfmt(CUS1_TBSP_DPSCORE_MAX_XDIM,CUS1_TBSP_DPSCORE_MAX_YDIM,1);
        dim3 nblcks_dpsfmt(
            (ndbCstrs + CUS1_TBSP_DPSCORE_MAX_XDIM - 1)/CUS1_TBSP_DPSCORE_MAX_XDIM,
            nqystrspart, nconfigsections);

        //reformat DP swift scores for efficient further processing:
        ReformatDPswiftScores<<<nblcks_dpsfmt,nthrds_dpsfmt,0,streamproc>>>(
            (topndpscores), q, ndbCstrs, ndbCposs, dbxpad,  maxnsteps, nconfigs,
            tmpdpdiagbuffers, wrkmemaux, wrkmem2);
        MYCUDACHECKLAST;
    }

    //size of dynamically allocted smem:
    const uint szdsmem_ch2ch = GetSmemSizeForMakeChain2ChainAssignment(maxnqrychains, maxnrfnchains);
    const uint maxnstepsmem2 = nconfigs;//CuMemoryBase::GetMaxNFragStepsMem2();
    assert(nconfigs == maxnstepsmem2);

    //execution configuration for chain-to-chain assignment:
    dim3 nthrds_ch2ch(CUAP_MAKEASSIGNMENT_XDIM,1,1);
    dim3 nblcks_ch2ch(nqycpxs, ndbCcpxs, nconfigs);

    //make chain assignemnts and calculate scores for full complexes
    MakeChain2ChainAssignment<true/*WRITESCORE*/, true/*WRITEASSG*/,false/*PASS2*/>
        <<<nblcks_ch2ch,nthrds_ch2ch,szdsmem_ch2ch,streamproc>>>(
            nqystrs, ndbCstrs, ndbCcpxs,  maxnqrychains, maxnrfnchains,  maxnsteps, maxnstepsmem2,
            NULL/*tfmmemory*/, wrkmemaux, wrkmem2);
    MYCUDACHECKLAST;


    const int complexes = (CLOptions::GetB_CHAINS() == 0);

    //execution configuration for setting convergence flag for low score complexes:
    //each block processes one query and CUS1_TBSP_DPSCORE_MAX_XDIM references:
    dim3 nthrds_cpxconv(CUS1_TBSP_DPSCORE_MAX_XDIM,CUS1_TBSP_DPSCORE_MAX_YDIM,1);
    dim3 nblcks_cpxconv(
        (ndbCcpxs + CUS1_TBSP_DPSCORE_MAX_XDIM - 1)/CUS1_TBSP_DPSCORE_MAX_XDIM,
        nqycpxs, 1);

    //scores originating from spatial indexing are not optimal yet;
    const float dpprescore = prescore;// * 0.5f;

    if(0.0f < prescore && complexes) {
        if(addchainlevelsearch)
            SetLowScoreConvergenceFlagComplexIntermediate
                <CUS1_TBSP_CPXSCORE_MAX_NCHAINS_FILTER>
                <<<nblcks_cpxconv,nthrds_cpxconv,0,streamproc>>>(
                    dpprescore, ndbCcpxs, ndbCstrs, maxnsteps, nconfigs, wrkmemaux, wrkmem2);
        else SetLowScoreConvergenceFlagComplexIntermediate
                <<<nblcks_cpxconv,nthrds_cpxconv,0,streamproc>>>(
                    dpprescore, ndbCcpxs, ndbCstrs, maxnsteps, nconfigs, wrkmemaux, wrkmem2);
        MYCUDACHECKLAST;
    }


    //execution configuration for finding the maximum among dp scores calculated for complexes:
    //each block processes one query and CUS1_TBSP_DPSCORE_MAX_XDIM references:
    dim3 nthrds_dpscmax(CUS1_TBSP_DPSCORE_MAX_XDIM,CUS1_TBSP_DPSCORE_MAX_YDIM,1);
    dim3 nblcks_dpscmax(
        (ndbCcpxs + CUS1_TBSP_DPSCORE_MAX_XDIM - 1)/CUS1_TBSP_DPSCORE_MAX_XDIM,
        nqycpxs, nconfigsections);

    //sort the best of the order-dependent scores and save the corresponding tfms:
    //NOTE: using attributes of complexes rather than chains for compact packing:
    SortBestDPscoresAndTMsAmongDPswiftsComplex
        <<<nblcks_dpscmax,nthrds_dpscmax,0,streamproc>>>(
            (topndpscores), myhdmax(nbranches, 1),  ndbCcpxs, ndbCstrs,  maxnsteps, maxnstepsmem2,
            tfmtarget/*out memtype*/, wrkmemtm/*in*/, wrkmemtmalt/*out*/,
            wrkmemaux, wrkmem2);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagefrg3_refinement_tfmaltconfig_complexes: refine all alternative 
// best-performing complex superpositions obtained through the extensive 
// application of spatial index;
// qycpx1len, dbcpx1len, lengths of the largest query & reference complexes;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
#define STAGE1_REFINEFRAG_COMPLEXES(tpCONDITIONAL,tpSECONDARYUPDATE) \
    stage1_complex::stage1_refinefrag_complex<tpCONDITIONAL,tpSECONDARYUPDATE>( \
        stgraphs, \
        stg1REFINE_INITIAL_DP/*fragments identified by DP*/, \
        FRAGREF_NMAXCONVIT/*#iterations until convergence*/, \
        streamproc, \
        maxnsteps, minfraglen, \
        nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs, \
        qycpx1len, dbcpx1len, qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad, \
        tmpdpdiagbuffers, tmpdpalnpossbuffer, \
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest, \
        wrkmemaux, wrkmem2, tfmmem);

void stagefrg3::stagefrg3_refinement_tfmaltconfig_complexes(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const int depth,
    const int nbranches,
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
    MYMSG("stagefrg3::stagefrg3_refinement_tfmaltconfig_complexes", 4);
    static const std::string preamb = "stagefrg3::stagefrg3_refinement_tfmaltconfig_complexes: ";


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


    //configurations to verfy alternatively:
    const int nconfigsections = 
        (depth <= CLOptions::csdDeep)? 3: ((depth <= CLOptions::csdHigh)? 2: 1);
    const int nconfigs = nbranches * nconfigsections;
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //process top N best-performing tfms found from the extensive 
    //application of spatial index:
    for(int rci = 0; rci < nconfigs; rci++)
    {
        //launch blocks along block diagonals to perform DP;
        //nblkdiags, total number of diagonals:
        for(uint d = 0; d < nblkdiags; d++)
        {
            ExecDPwBtck3264x
                <false/*ANCHORRGN*/,false/*BANDED*/,true/*GAP0*/,D02IND_SEARCH,
                 true/*ALTSCTMS*/,true/*COMPLEX*/>
                <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                    d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, rci/*stepnumber*/,
                    0.0f/*gap open cost*/,
                    wrkmemtmalt/*in*/, wrkmemaux,
                    tmpdpdiagbuffers, tmpdpbotbuffer, btckdata,
                    wrkmem2,
                    maxnstepsmem2/*assgmaxnsteps*/,
                    rci/*assgstepnumber*/);
            MYCUDACHECKLAST;
        }

        //process DP result;
        BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/,true/*COMPLEX*/>
            <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                ndbCstrs, ndbCposs, dbxpad, maxnsteps, rci/*stepnumber*/,
                btckdata, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
                frg3_cpx_complexstepnumber,
                maxnstepsmem2/*assgmaxnsteps*/,
                rci/*assgstepnumber*/);
        MYCUDACHECKLAST;

        //reformat match;
        ReformatMatched4Complexes<<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
                ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                maxnstepsmem2/*assgmaxnsteps*/,
                rci/*assgstepnumber*/,
                frg3_cpx_complexstepnumber,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer);
        MYCUDACHECKLAST;


        if(rci < 1) {
            STAGE1_REFINEFRAG_COMPLEXES(false/*true/CONDITIONAL */, SECONDARYUPDATE_UNCONDITIONAL);
        } else {
            STAGE1_REFINEFRAG_COMPLEXES(false, SECONDARYUPDATE_CONDITIONAL);
        }


        if(depth <= CLOptions::csdHigh)
        {   //one additional iteration of full DP sweep
            for(uint d = 0; d < nblkdiags; d++)
            {
                ExecDPwBtck3264x
                    <false/*ANCHORRGN*/,false/*BANDED*/,true/*GAP0*/,D02IND_SEARCH,
                     false/*ALTSCTMS*/,true/*COMPLEX*/>
                    <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                        d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                        0.0f/*gap open cost*/,
                        wrkmemtmibest/*in*/, wrkmemaux,
                        tmpdpdiagbuffers, tmpdpbotbuffer, btckdata,
                        wrkmem2,
                        maxnstepsmem2/*assgmaxnsteps*/,
                        rci/*assgstepnumber*/);
                MYCUDACHECKLAST;
            }

            //process DP result;
            BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/,true/*COMPLEX*/>
                <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                    ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                    btckdata, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
                    frg3_cpx_complexstepnumber,
                    maxnstepsmem2/*assgmaxnsteps*/,
                    rci/*assgstepnumber*/);
            MYCUDACHECKLAST;

            //reformat match;
            ReformatMatched4Complexes<<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
                    ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                    maxnstepsmem2/*assgmaxnsteps*/,
                    rci/*assgstepnumber*/,
                    frg3_cpx_complexstepnumber,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer);
            MYCUDACHECKLAST;

            if(rci < 1) {
                STAGE1_REFINEFRAG_COMPLEXES(false/*CONDITIONAL*/, SECONDARYUPDATE_UNCONDITIONAL);
            } else {
                STAGE1_REFINEFRAG_COMPLEXES(false, SECONDARYUPDATE_CONDITIONAL);
            }
        }
    }//rci
}
