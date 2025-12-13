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

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custage1/custage1.cuh"

#include "libmycu/custgfrg/linear_scoring.cuh"
#include "custage_frg.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stagefrg: search for superposition between multiple molecules 
// simultaneously by exhaustively matching their fragments for similarity;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefrg::run_stagefrg(
    std::map<CGKey,MyCuGraph>& /*stgraphs*/,
    cudaStream_t streamproc,
    const uint maxnsteps,
    float /*scorethld*/,
    uint nqystrs, uint ndbCstrs,
    uint nqyposs, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint qystrnlen, uint dbstrnlen,
    uint dbxpad,
    float* __restrict__ /*scores*/, 
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    uint* __restrict__ /*maxscoordsbuf*/,
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
    MYMSG("stagedp::run_stagedp", 4);

    stagefrg_extensive_frg_swift(
        streamproc,
        maxnsteps,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, btckdata,
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest, wrkmemaux, wrkmem2, tfmmem);
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagefrg_extensive_frg_swift: calculate tmscores and find most favorable 
// initial superposition based on fragment matching of multiple queries and 
// references;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefrg::stagefrg_extensive_frg_swift(
    cudaStream_t streamproc,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint /*qystrnlen*/, uint /*dbstrnlen*/,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ /*tmpdpbotbuffer*/,
    float* __restrict__ tmpdpalnpossbuffer,
    char* __restrict__ /*btckdata*/,
    float* __restrict__ wrkmem,
    float* __restrict__ /*wrkmemccd*/,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ /*tfmmem*/)
{
    MYMSG("stagedp::stagefrg_extensive_frg_swift", 4);
    static const std::string preamb = "stagedp::stagefrg_extensive_frg_swift: ";

    enum{nfrags = 2};//number of fragments of different length used
    static const int frags[nfrags] = {20, 100};
    int fraglen = GetNAlnPoss_frg(
            qystr1len, dbstr1len, 0/*qrypos,unsed*/, 0/*rfnpos,unused*/,
            0/*qryfragfct,unsed*/, 0/*rfnfragfct,unused*/, 0/*fraglen index*/);

    const int depth = CLOptions::GetC_DEPTH();
    const int fctdiv = (depth==CLOptions::csdShallow)? 5: 1;
    const int minnsteps = (depth==CLOptions::csdShallow)? 2: 10;
    const int rfnstepsize = (depth==CLOptions::csdShallow)? 5: 1;
    //set minimum #steps to 10 since length 150 leads to 150/15=10,
    //the largest among #steps for medium-sized structures:
    const int nstepsy = myhdmax(minnsteps, (int)qystr1len/(fctdiv * 45) + 1);
    const int nstepsx = ((int)dbstr1len - fraglen + rfnstepsize - 1) / rfnstepsize;

//     if(maxnsteps < nstepsx)
//         //maxnsteps computed for the minimum of #query and reference positions
//         //(step of 40<45 used by CuDeviceMemory)
//         throw MYRUNTIME_ERROR(preamb + "Number of steps exceeds the predetermined one.");

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

    //initialize memory for best scores;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory for query and reference positions;
    InitScores<INITOPT_QRYRFNPOS><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory for fragment specifications;
    InitScores<INITOPT_FRAGSPECS><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //reset convergence flag;
    InitScores<INITOPT_CONVFLAG_FRAGREF>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;


    for(int fragndx = 0; fragndx < nfrags; fragndx++)
    {
        //step number (<=maxnsteps) for efficiently launching numerous processing 
        //kernels and calculating scores on the query-reference dimensions:
        int stepnumber = 0;
        int ysndxproc = 0, xsndxproc = 0;//processed indices

        for(int ysndx = 0; ysndx < nstepsy; ysndx++)
        {
            //increase #step indices over query and reference structures;
            //maxnsteps is max allowed steps to be processed in parallel simultaneously
            for(int xsndx = 0; xsndx < nstepsx; xsndx++)
            {
                //#thread blocks to be launched
                //int nlocsteps = 
                //    (xsndx + (int)maxnsteps <= nstepsx)? maxnsteps: (nstepsx - xsndx);

                bool lastiteration = (nstepsy <= ysndx + 1) && (nstepsx <= xsndx + 1);

                stepnumber++;

                if(stepnumber < (int)maxnsteps && !lastiteration)
                    continue;

                //int nlocsteps = stepnumber;

                //execution configuration for tfm matrix initialization:
                //each block processes one query and CUS1_TBINITSP_TFMINIT_XFCT references:
                //dim3 nthrds_tfminitibest(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
                //dim3 nblcks_tfminitibest(
                //    (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
                //    nqystrs, nlocsteps);

                //NOTE: initialize memory for best transformation matrices;
                //InitTfmMatrices<<<nblcks_tfminitibest,nthrds_tfminitibest,0,streamproc>>>(
                //    ndbCstrs, maxnsteps, 0/*minfraglen(unused)*/, 0/*stepinit(unused)*/,
                //    false/*checkfragos*/,  wrkmemtm);
                //MYCUDACHECKLAST;

                //wrkmemtmibest contains best tfms over all DP recurses
                stagefrg_score_based_on_fragmatching3(
                    streamproc,
                    frags[fragndx],
                    ysndxproc, xsndxproc, fragndx,
                    maxnsteps, stepnumber,
                    qystr1len, dbstr1len,
                    nqystrs, ndbCstrs, ndbCposs, dbxpad,
                    tmpdpdiagbuffers, tmpdpalnpossbuffer,
                    wrkmem, wrkmemtmibest, wrkmemaux, wrkmem2, wrkmemtm
                );

                stepnumber = 0;
                ysndxproc = ysndx;
                xsndxproc = xsndx;

            }//reference positions
        }//query positions
    }//fragments

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUSF_TBSP_INDEX_SCORE_MAX_XDIM references:
    dim3 nthrds_linscmax(CUSF_TBSP_INDEX_SCORE_MAX_XDIM,CUSF_TBSP_INDEX_SCORE_MAX_YDIM,1);
    dim3 nblcks_linscmax(
        (ndbCstrs + CUSF_TBSP_INDEX_SCORE_MAX_XDIM - 1)/CUSF_TBSP_INDEX_SCORE_MAX_XDIM,
        nqystrs, 1);

    SaveBestScoreAndConfigAmongBestsLinear
        <<<nblcks_linscmax,nthrds_linscmax,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemaux);
    MYCUDACHECKLAST;

    //recalculate tfms wrt references from the best fragments obtained:
    stagefrg_get_tfms_from_best_fragments1(
        streamproc,
        frags[nfrags-1], maxnsteps,
        qystr1len, dbstr1len,  nqystrs, ndbCstrs,
        wrkmem, wrkmemaux, wrkmem2, wrkmemtmibest
    );
}





// -------------------------------------------------------------------------
// stagefrg_get_tfms_from_best_fragments1: recalculate transformation 
// matrices wrt references based on the superposition of best-scoring 
// fragments obtained;
//
void stagefrg::stagefrg_get_tfms_from_best_fragments1(
    cudaStream_t streamproc,
    const int maxfraglen,
    const uint maxnsteps,
    const uint qystr1len, const uint dbstr1len,
    const uint nqystrs, const uint ndbCstrs,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm)
{
    //max #aligned positions over the pairs in a chunk
    int minlenmax = myhdmin(qystr1len, dbstr1len);

    //execution configuration for CCM initialization:
    //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
    dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
    dim3 nblcks_init(
        (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
        nqystrs, 1/*actualnsteps*/);

    //execution configuration for reduction:
    //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
    dim3 nthrds_ccmtx_frg(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
    dim3 nblcks_ccmtx_frg(
        (maxfraglen + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
        ndbCstrs, nqystrs * 1/*actualnsteps*/);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
    dim3 nblcks_copyto(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs, 1/*actualnsteps*/);

    //execution configuration for calculating transformation matrices:
    //each block processes one query and CUS1_TBSP_TFM_N references:
    dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
    dim3 nblcks_tfm(
        (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
        nqystrs, 1/*actualnsteps*/);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
    dim3 nblcks_copyfrom(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs, 1/*actualnsteps*/);

    stagefrg_best_fragments1_subiter1(
        streamproc,
        maxnsteps,
        nqystrs, ndbCstrs,
        wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
        nblcks_init, nthrds_init,
        nblcks_ccmtx_frg, nthrds_ccmtx_frg,
        nblcks_copyto, nthrds_copyto,
        nblcks_copyfrom, nthrds_copyfrom,
        nblcks_tfm, nthrds_tfm);
}

// -------------------------------------------------------------------------
// stagefrg_score_based_on_fragmatching3: calculate superposition scores in 
// three iterations for massive number of variants obtained by fragment 
// matching between query and reference structures;
// maxnsteps, max #steps that can be executed in parallel for one 
// query-reference pair; it corresponds to different #variants for a pair;
// actualnsteps, actual #variants (steps);
//
void stagefrg::stagefrg_score_based_on_fragmatching3(
    cudaStream_t streamproc,
    const int maxfraglen,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint actualnsteps,
    const uint qystr1len, const uint dbstr1len,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ /*tmpdpalnpossbuffer*/,
    float* __restrict__ wrkmem,
    float* __restrict__ /*wrkmemtmibest*/,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm)
{
    //max #aligned positions over the pairs in a chunk
    int minlenmax = myhdmin(qystr1len, dbstr1len);

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

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, actualnsteps);



    //execution configuration for linearly calculating positional scores (no dp):
    //block processes CUSF_TBSP_INDEX_SCORE_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_linscos(CUSF_TBSP_INDEX_SCORE_XDIM,1,1);
    dim3 nblcks_linscos(
        (myhdmin((uint)CUSF_TBSP_INDEX_SCORE_POSLIMIT2, dbstr1len) +
            CUSF_TBSP_INDEX_SCORE_XDIMLGL - 1)/CUSF_TBSP_INDEX_SCORE_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //dynamically determined stack size for the linscos kernel:
    uint stacksize_linscos = 1;
    if(0 < qystr1len)
        stacksize_linscos = myhdmin((uint)17, (uint)ceilf(log2f(qystr1len)) + 1);

    //size of dynamically allocted smem for the linscos kernel:
    uint szdsmem_linscos = sizeof(float) * (
#if (CUSF_TBSP_INDEX_SCORE_XFCT > 1)
        nTTranformMatrix +
#endif
        CUSF_TBSP_INDEX_SCORE_XDIM * stacksize_linscos * nStks_);

    MYCUDACHECK(cudaFuncSetAttribute(
        PositionalScoresFromIndexLinear,
        cudaFuncAttributeMaxDynamicSharedMemorySize, szdsmem_linscos));


    //execution configuration for reducing positional scores (calculated before):
    //block processes CUSF_TBSP_INDEX_SCORE_REDUCE_XDIMLGL positions of one query-reference pair:
    dim3 nthrds_linreduce(CUSF_TBSP_INDEX_SCORE_REDUCE_XDIM,1,1);
    dim3 nblcks_linreduce(
        (myhdmin((uint)CUSF_TBSP_INDEX_SCORE_POSLIMIT2, dbstr1len) +
            CUSF_TBSP_INDEX_SCORE_REDUCE_XDIMLGL - 1)/CUSF_TBSP_INDEX_SCORE_REDUCE_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);



    //execution configuration for saving the configuration (score and fragment 
    //details) of best performing transformation matrices:
    //each block processes one query and CUSF_TBSP_INDEX_SCORE_SAVECFG_XDIM references:
    dim3 nthrds_savecfg(CUSF_TBSP_INDEX_SCORE_SAVECFG_XDIM,1,1);
    dim3 nblcks_savecfg(
        (ndbCstrs + CUSF_TBSP_INDEX_SCORE_SAVECFG_XDIM - 1)/CUSF_TBSP_INDEX_SCORE_SAVECFG_XDIM,
        nqystrs, actualnsteps);


    stagefrg_fragmatching3_subiter1(
        streamproc,
        qryfragfct, rfnfragfct, fragndx,
        maxnsteps,
        nqystrs, ndbCstrs,
        wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
        nblcks_init, nthrds_init,
        nblcks_ccmtx_frg, nthrds_ccmtx_frg,
        nblcks_copyto, nthrds_copyto,
        nblcks_copyfrom, nthrds_copyfrom,
        nblcks_tfm, nthrds_tfm);

    stagefrg_fragmatching3_subiter2(
        streamproc,
        qryfragfct, rfnfragfct, fragndx,
        maxnsteps,
        nqystrs, ndbCstrs, ndbCposs, dbxpad,
        tmpdpdiagbuffers, wrkmemaux, wrkmemtm,
        nblcks_scinit, nthrds_scinit,
        nblcks_linscos, nthrds_linscos, szdsmem_linscos, stacksize_linscos,
        nblcks_linreduce, nthrds_linreduce,
        nblcks_savecfg, nthrds_savecfg);
}


// -------------------------------------------------------------------------
// stagefrg_fragmatching3_subiter1: subiteration 1 of scoring structures by 
// fragment matching in three iterations;
//
inline
void stagefrg::stagefrg_fragmatching3_subiter1(
    cudaStream_t streamproc,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    float* __restrict__ wrkmem,
    float* __restrict__ /*wrkmemaux*/,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx_frg, const dim3& nthrds_ccmtx_frg,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    const int shallow = CLOptions::GetC_DEPTH() == CLOptions::csdShallow;

    //initialize memory for calculating cross covariance matrices
    InitCCData0_frg<<<nblcks_init,nthrds_init,0,streamproc>>>(
        (shallow),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,  wrkmem);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64_frg<<<nblcks_ccmtx_frg,nthrds_ccmtx_frg,0,streamproc>>>(
        (shallow),  nqystrs, ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,  wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable efficient 
    //structure-specific calculation; READNPOS_NOREAD, do not verify whether
    //#positions on which tfms are calculated has changed:
    CopyCCDataToWrkMem2_frg<<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            (shallow),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
            wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    //NOTE: calculate transformation matrices reversed wrt query-reference structure pair
    CalcTfmMatrices<TFMTX_REVERSE_TRUE><<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefrg_fragmatching3_subiter2: subiteration 2 of scoring structures by 
// fragment matching in three iterations;
//
inline
void stagefrg::stagefrg_fragmatching3_subiter2(
    cudaStream_t streamproc,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint /*dbxpad*/,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_linscos, const dim3& nthrds_linscos,
        const uint szdsmem_linscos, const uint stacksize_linscos,
    const dim3& nblcks_linreduce, const dim3& nthrds_linreduce,
    const dim3& nblcks_savecfg, const dim3& nthrds_savecfg)
{
    const int shallow = CLOptions::GetC_DEPTH() == CLOptions::csdShallow;

    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate positional score using index trees
    PositionalScoresFromIndexLinear
        <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
            (int)stacksize_linscos, (shallow),
            nqystrs, ndbCstrs, ndbCposs,  maxnsteps,
            qryfragfct, rfnfragfct, fragndx,
            wrkmemtm, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    //reduce positional score calculated before
    ReduceScoresLinear<<<nblcks_linreduce,nthrds_linreduce,0,streamproc>>>(
            (shallow), nqystrs, ndbCstrs, ndbCposs,  maxnsteps,
            qryfragfct, rfnfragfct, fragndx,
            tmpdpdiagbuffers, wrkmemaux);
    MYCUDACHECKLAST;

    //save scores and fragment config for tfms
    SaveBestScoreAndConfigLinear<<<nblcks_savecfg,nthrds_savecfg,0,streamproc>>>(
        (shallow), ndbCstrs,  maxnsteps,  qryfragfct, rfnfragfct, fragndx,  wrkmemaux);
    MYCUDACHECKLAST;
}



// -------------------------------------------------------------------------
// stagefrg_best_fragments1_subiter1: actual process of recalculating tfms 
// based on best fragment obtained; the recalculation is wrt references!
//
inline
void stagefrg::stagefrg_best_fragments1_subiter1(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx_frg, const dim3& nthrds_ccmtx_frg,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    // const int depth = CLOptions::GetC_DEPTH();

    //initialize memory for calculating cross covariance matrices
    InitCCData0_frgbest<<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64_frgbest<<<nblcks_ccmtx_frg,nthrds_ccmtx_frg,0,streamproc>>>(
        nqystrs, ndbCstrs,  maxnsteps,  wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable efficient 
    //structure-specific calculation; READNPOS_NOREAD, do not verify whether
    //#positions on which tfms are calculated has changed:
    CopyCCDataToWrkMem2_frgbest<<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    //calculate transformation matrices wrt reference structures (no reverse):
    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;
}
