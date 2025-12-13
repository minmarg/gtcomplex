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

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_dp_scan.cuh"
#include "libmycu/cudp/dpw_score.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "custage_dp.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stagedp: search for superposition between multiple molecules 
// simultaneoulsy by exhaustively applying DP using transformation matrices 
// obtained on all fragments separated by a constant step;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagedp::run_stagedp(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const uint maxnsteps,
    float scorethld,
    uint nqystrs, uint ndbCstrs,
    uint nqyposs, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint qystrnlen, uint dbstrnlen,
    uint dbxpad,
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
    MYMSG("stagedp::run_stagedp", 4);
    constexpr bool completedpapp = false;//true;

    if(completedpapp)
        stagedp_extensive_dp_complete(
            streamproc,
            maxnsteps,
            nqystrs, ndbCstrs,
            nqyposs, ndbCposs,
            qystr1len, dbstr1len,
            qystrnlen, dbstrnlen,
            dbxpad,
            tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, btckdata,
            wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest, wrkmemaux, wrkmem2, tfmmem);
    else
        stagedp_extensive_dp_swift(
            streamproc,
            maxnsteps,
            nqystrs, ndbCstrs,
            nqyposs, ndbCposs,
            qystr1len, dbstr1len,
            qystrnlen, dbstrnlen,
            dbxpad,
            tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, btckdata,
            wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest, wrkmemaux, wrkmem2, tfmmem);

#if 0
    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs,1);

    //reset query and reference positions identified when processed full 
    //sequences in the first stage;
    InitScores<INITOPT_QRYRFNPOS>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, wrkmemaux);
    MYCUDACHECKLAST;

    //reset fragment specifications which are to be identified during the 
    //refinement of the fragment identified by DP using SS information;
    InitScores<INITOPT_FRAGSPECS>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, wrkmemaux);
    MYCUDACHECKLAST;

    //reset the number of aligned positions before conducting DP;
    InitScores<INITOPT_NALNPOSS>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, wrkmemaux);
    MYCUDACHECKLAST;

    //reset score convergence flag;
    InitScores<INITOPT_CONVFLAG_SCOREDP>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, wrkmemaux);
    MYCUDACHECKLAST;

    //reset the flag of no progressing tmscore;
    InitScores<INITOPT_CONVFLAG_NOTMPRG>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, wrkmemaux);
    MYCUDACHECKLAST;


    //identify alignment by DP using secondary structure information
    stage2_dpss_align(
        streamproc,
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
        wrkmemaux);

    //refine alignment boundaries to improve scores
    stage1::stage1_refinefrag(
        stgraphs,
        stg1REFINE_INITIAL_DP/*fragments identified by DP*/,
        FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
        streamproc,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers,
        tmpdpalnpossbuffer,
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem);

    //verify whether the last score obtained is large enough to
    //apply stage1_dprefine
    CheckScoreProgression<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, 0.2f/*maxscorefct*/, wrkmemaux);
    MYCUDACHECKLAST;

    //refine alignment boundaries identified in the previous 
    //substage by applying DP
    stage1::stage1_dprefine(
        stgraphs,
        streamproc,
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
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem);
#endif
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagedp_extensive_dp_complete: apply DP with all fragment-induced 
// transformation matrices obtained by traversing queries and references 
// with a constant step and calculate tm-scores for all alignments;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagedp::stagedp_extensive_dp_complete(
    cudaStream_t streamproc,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint /*qystrnlen*/, uint /*dbstrnlen*/,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    MYMSG("stagedp::stagedp_extensive_dp_complete", 4);
    static const std::string preamb = "stagedp::stagedp_extensive_dp_complete: ";
    const int depth = CLOptions::GetC_DEPTH();
    const int fctdiv = (depth==CLOptions::csdShallow)? 5: 1;
    const int minnsteps = (depth==CLOptions::csdShallow)? 2: 10;
    //set minimum #steps to 10 since length 150 leads to 150/15=10,
    //the largest among #steps for medium-sized structures:
    const int nstepsy = myhdmax(minnsteps, (int)qystr1len/(fctdiv * 45) + 1);
    const int nstepsx = myhdmax(minnsteps, (int)dbstr1len/(fctdiv * 45) + 1);
    constexpr int nfrags = 2;//number of fragments of different length used
    static const int frags[nfrags] = {20, 100};
    static const float gcost = 0.0f;

    //NOTE: using anchor (for banded or not DP) adds several uncoalesced 
    //NOTE: reads and additional inexpensive arithmetics
    constexpr bool vANCHORRGN = false;//using anchor region
    constexpr bool vBANDED = false;//banded alignment

//     if(maxnsteps < nstepsx)
//         //maxnsteps computed for the minimum of #query and reference positions
//         //(step of 40<45 used by CuDeviceMemory)
//         throw MYRUNTIME_ERROR(preamb + "Number of steps exceeds the predetermined one.");

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //CUDP_2DCACHE_DIM_D x CUDP_2DCACHE_DIM_X;
    //NOTE: using block diagonals, where blocks share a common point 
    //NOTE: (corner) with a neighbour in a diagonal;
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    //then, this number d is divided by the length of the block diagonal;
    //uint nblkdiags =
    // ((dbstr1len+qystr1len-1)+CUDP_2DCACHE_DIM_X-1)/CUDP_2DCACHE_DIM_X;
    //REVISION: due to the positioning of the first block, the first 
    // 1-position diagonal of the first diagonal block is out of bounds: remove -1
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    //----
    //NOTE: now use block DIAGONALS, whe    re blocks share a COMMON POINT 
    //NOTE: (corner, instead of edge) with a neighbour in a diagonal;
    //the number of series of such block diagonals equals 
    // #regular block diagonals (above) + {(l-1)/w}, 
    // where l is query length (y coord.), w is block width (dimension), and
    // {} denotes floor rounding; {(l-1)/w} is #interior divisions;
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;


    //execution configuration for extracting matched positions
    //identified during DP:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);


    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

    //initialize memory for best scores only;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //reset convergence flag;
    InitScores<INITOPT_CONVFLAG_FRAGREF>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;


    //step number (<=maxnsteps) for collecting aligned positions from a series of 
    //DP application in order to efficiently launch numerous processing kernels:
    int stepnumber = 0;

    for(int ysndx = 0; ysndx < nstepsy; ysndx++)
    {
        //increase #step indices over reference structures by maxnsteps,
        //which is max allowed steps to be processed in parallel simultaneously
        for(int xsndx = 0; xsndx < nstepsx; xsndx += maxnsteps)
        {
            //reduce #thread blocks to be launched if maxnsteps implies exceeding the limit
            int nlocsteps = 
                (xsndx + (int)maxnsteps <= nstepsx)? maxnsteps: (nstepsx - xsndx);


            //execution configuration for tfm matrix initialization:
            //each block processes one query and CUS1_TBINITSP_TFMINIT_XFCT references:
            //dim3 nthrds_tfminitibest(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
            //dim3 nblcks_tfminitibest(
            //    (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
            //    nqystrs, nlocsteps);

            //NOTE: initialize memory for transformation matrices;
            //InitTfmMatrices<<<nblcks_tfminitibest,nthrds_tfminitibest,0,streamproc>>>(
            //    ndbCstrs, maxnsteps, 0/*minfraglen(unused)*/, 0/*stepinit(unused)*/,
            //    false/*checkfragos*/,  wrkmemtm);
            //MYCUDACHECKLAST;

            //execution configuration for initialization:
            //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
            dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
            dim3 nblcks_init(
                (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
                nqystrs, nlocsteps);

            //execution configuration for reformatting data:
            //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
            dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
            dim3 nblcks_copyto(
                (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
                nqystrs, nlocsteps);

            //execution configuration for calculating transformation matrices:
            //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
            dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
            dim3 nblcks_tfm(
                (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
                nqystrs, nlocsteps);

            //execution configuration for reformatting data:
            //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
            dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
            dim3 nblcks_copyfrom(
                (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
                nqystrs, nlocsteps);


            for(int fragndx = 0; fragndx < nfrags; fragndx++)
            {
                //execution configuration for reduction:
                //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
                //NOTE: ndbCstrs and nqystrs * maxnsteps cannot be greater than 65535:
                //TODO: ensured by JobDispatcher and CuDeviceMemory
                dim3 nthrds_ccmtx_var(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
                dim3 nblcks_ccmtx_var(
                    (frags[fragndx] + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
                    ndbCstrs, nqystrs * nlocsteps);

                //reuse memory are wrkmemccd for resulting tfms instead of wrkmemtm,
                //which is employed during the three-iteration tmscore calculation;
                //NOTE: wrkmemccd is larger as the size of CCD data > that of tfms;
                stagedp_tfms_from_fragments(
                    streamproc,
                    ysndx, xsndx, fragndx, maxnsteps,
                    nqystrs, ndbCstrs,
                    wrkmem, wrkmemaux, wrkmem2, wrkmemccd,
                    //
                    nblcks_init, nthrds_init,
                    nblcks_ccmtx_var, nthrds_ccmtx_var,
                    nblcks_copyto, nthrds_copyto,
                    nblcks_copyfrom, nthrds_copyfrom,
                    nblcks_tfm, nthrds_tfm);


                for(int lsi = 0; lsi < nlocsteps; lsi++)
                {
                    //perform complete DP with backtracking: launch blocks along data block anti-diagonals;
                    //nblkdiags, total number of diagonals;
                    //NOTE: if vANCHORRGN==true, one additional kernel for setting tawmvQRYpos, 
                    //NOTE: tawmvRFNpos, and fragment length tawmvSubFragNdx!
                    for(uint d = 0; d < nblkdiags; d++)
                    {
                        ExecDPwBtck3264x<vANCHORRGN,vBANDED,true/*GAP0*/,D02IND_DPSCAN>
                            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                                lsi/*ndx for wrkmemccd*/, gcost,
                                wrkmemccd/*as wrkmemtm*/, wrkmemaux,
                                tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
                        MYCUDACHECKLAST;
                    }

                    //get the alignments written to tmpdpalnpossbuffer
                    BtckToMatched32x<vANCHORRGN,vBANDED>
                        <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                            ndbCstrs, ndbCposs, dbxpad,
                            maxnsteps, stepnumber/*ndx for tmpdpalnpossbuffer*/,
                            btckdata, wrkmemaux, tmpdpalnpossbuffer);
                    MYCUDACHECKLAST;

                    bool lastiteration =
                        (nstepsy <= ysndx + 1) &&
                        (nstepsx <= xsndx + (int)maxnsteps) &&
                        (nfrags <= fragndx + 1) && (nlocsteps <= lsi + 1);

                    stepnumber++;

                    if((int)maxnsteps <= stepnumber || lastiteration)
                    {
                        //wrkmemtmibest contains best tfms over all DP recurses
                        stagedp_score_alignment3(
                            streamproc,
                            maxnsteps, stepnumber,
                            qystr1len, dbstr1len,
                            nqystrs, ndbCstrs, ndbCposs, dbxpad,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,
                            wrkmem, wrkmemtmibest, wrkmemaux, wrkmem2, wrkmemtm
                        );

                        stepnumber = 0;
                    }
                }//nlocsteps
            }//fragments

        }//reference positions
    }//query positions

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    SaveBestScoreAndTMAmongBests<false/*WRITEFRAGINFO*/>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemtmibest, tfmmem, wrkmemaux);
    MYCUDACHECKLAST;
}



// -------------------------------------------------------------------------
// stagedp_extensive_dp_swift: apply DP with all fragment-induced 
// transformation matrices obtained by traversing queries and references 
// with a constant step and calculate tm-scores only for the best scoring 
// alignments;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagedp::stagedp_extensive_dp_swift(
    cudaStream_t streamproc,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint /*qystrnlen*/, uint /*dbstrnlen*/,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    MYMSG("stagedp::stagedp_extensive_dp_swift", 4);
    static const std::string preamb = "stagedp::stagedp_extensive_dp_swift: ";
    const int depth = CLOptions::GetC_DEPTH();
    const int fctdiv = (depth==CLOptions::csdShallow)? 5: 1;
    const int minnsteps = (depth==CLOptions::csdShallow)? 2: 10;
    //set minimum #steps to 10 since length 150 leads to 150/15=10,
    //the largest among #steps for medium-sized structures:
    const int nstepsy = myhdmax(minnsteps, (int)qystr1len/(fctdiv * 45) + 1);
    const int nstepsx = myhdmax(minnsteps, (int)dbstr1len/(fctdiv * 45) + 1);
    constexpr int nfrags = 2;//number of fragments of different length used
    static const int frags[nfrags] = {20, 100};
    static const float gcost = 0.0f;

    //NOTE: using anchor (for banded or not DP) adds several uncoalesced 
    //NOTE: reads, additional inexpensive arithmetics, but most 
    //NOTE: importantly, increases #registers leading to reduced occupancy 
    constexpr bool vANCHORRGNswft = false;//using anchor region
    constexpr bool vBANDEDswft = false;//banded alignment
    constexpr bool vANCHORRGN = false;//using anchor region
    constexpr bool vBANDED = false;//banded alignment

//     if(maxnsteps < nstepsx)
//         //maxnsteps computed for the minimum of #query and reference positions
//         //(step of 40<45 used by CuDeviceMemory)
//         throw MYRUNTIME_ERROR(preamb + "Number of steps exceeds the predetermined one.");

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //CUDP_2DCACHE_DIM_D x CUDP_2DCACHE_DIM_X;
    //NOTE: using block diagonals, where blocks share a common point 
    //NOTE: (corner) with a neighbour in a diagonal;
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    //then, this number d is divided by the length of the block diagonal;
    //uint nblkdiags =
    // ((dbstr1len+qystr1len-1)+CUDP_2DCACHE_DIM_X-1)/CUDP_2DCACHE_DIM_X;
    //REVISION: due to the positioning of the first block, the first 
    // 1-position diagonal of the first diagonal block is out of bounds: remove -1
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    //----
    //NOTE: now use block DIAGONALS, where blocks share a COMMON POINT 
    //NOTE: (corner, instead of edge) with a neighbour in a diagonal;
    //the number of series of such block diagonals equals 
    // #regular block diagonals (above) + {(l-1)/w}, 
    // where l is query length (y coord.), w is block width (dimension), and
    // {} denotes floor rounding; {(l-1)/w} is #interior divisions;
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;


    //configuration for swift DP; same considerations hold:
    const uint maxblkdiagelems_swft = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_SWFT_2DCACHE_DIM_D, CUDP_SWFT_2DCACHE_DIM_X);
    dim3 nthrds_dp_swft(CUDP_SWFT_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp_swft(maxblkdiagelems_swft, ndbCstrs, nqystrs * nstepsx);
    uint nblkdiags_swft = (uint)
        (((dbstr1len + qystr1len) + CUDP_SWFT_2DCACHE_DIM_X-1) / CUDP_SWFT_2DCACHE_DIM_X);
    nblkdiags_swft += (uint)(qystr1len - 1) / CUDP_SWFT_2DCACHE_DIM_D;


    //execution configuration for finding the maximum among dp scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_dpscmax(CUS1_TBSP_DPSCORE_MAX_XDIM,CUS1_TBSP_DPSCORE_MAX_YDIM,1);
    dim3 nblcks_dpscmax(
        (ndbCstrs + CUS1_TBSP_DPSCORE_MAX_XDIM - 1)/CUS1_TBSP_DPSCORE_MAX_XDIM,
        nqystrs, 1);


    //execution configuration for extracting matched positions
    //identified during DP:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);


    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

    //initialize memory for best scores only;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //reset convergence flag;
    InitScores<INITOPT_CONVFLAG_FRAGREF>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;


    //step number (<=maxnsteps) for collecting aligned positions from a series of 
    //DP application in order to efficiently launch numerous processing kernels:
    int stepnumber = 0;

    for(int ysndx = 0; ysndx < nstepsy; ysndx++)
    {
        //increase #step indices over reference structures by maxnsteps,
        //which is max allowed steps to be processed in parallel simultaneously
        for(int xsndx = 0; xsndx < nstepsx; xsndx += maxnsteps)
        {
            //reduce #thread blocks to be launched if maxnsteps implies exceeding the limit
            int nlocsteps = 
                (xsndx + (int)maxnsteps <= nstepsx)? maxnsteps: (nstepsx - xsndx);


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

            nthrds_scinit = dim3(CUS1_TBSP_SCORE_SET_XDIM,1,1);
            nblcks_scinit = dim3(
                (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
                nqystrs, nlocsteps);

            //if(xsndx == 0) //initialize memory for best scores:
                InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
                    ndbCstrs, maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/, wrkmemaux);
            MYCUDACHECKLAST;

            //execution configuration for initialization:
            //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
            dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
            dim3 nblcks_init(
                (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
                nqystrs, nlocsteps);

            //execution configuration for reformatting data:
            //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
            dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
            dim3 nblcks_copyto(
                (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
                nqystrs, nlocsteps);

            //execution configuration for calculating transformation matrices:
            //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
            dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
            dim3 nblcks_tfm(
                (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
                nqystrs, nlocsteps);

            //execution configuration for reformatting data:
            //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
            dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
            dim3 nblcks_copyfrom(
                (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
                nqystrs, nlocsteps);


            nthrds_dp_swft = dim3(CUDP_SWFT_2DCACHE_DIM_D,1,1);
            nblcks_dp_swft = dim3(maxblkdiagelems_swft, ndbCstrs, nqystrs * nlocsteps);


            for(int fragndx = 0; fragndx < nfrags; fragndx++)
            {
                //execution configuration for reduction:
                //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
                //NOTE: ndbCstrs and nqystrs * maxnsteps cannot be greater than 65535:
                //TODO: ensured by JobDispatcher and CuDeviceMemory
                dim3 nthrds_ccmtx_var(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
                dim3 nblcks_ccmtx_var(
                    (frags[fragndx] + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
                    ndbCstrs, nqystrs * nlocsteps);

                stagedp_tfms_from_fragments(
                    streamproc,
                    ysndx, xsndx, fragndx, maxnsteps,
                    nqystrs, ndbCstrs,
                    wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
                    //
                    nblcks_init, nthrds_init,
                    nblcks_ccmtx_var, nthrds_ccmtx_var,
                    nblcks_copyto, nthrds_copyto,
                    nblcks_copyfrom, nthrds_copyfrom,
                    nblcks_tfm, nthrds_tfm);

                //launch blocks along block diagonals to perform DP;
                //nblkdiags_swft, total number of diagonals:
                for(uint d = 0; d < nblkdiags_swft; d++)
                {
                    ExecDPScore3264x<vANCHORRGNswft,vBANDEDswft,true/*GAP0*/,false/*CHECKCONV*/>
                        <<<nblcks_dp_swft,nthrds_dp_swft,0,streamproc>>>(
                            d, nqystrs, ndbCstrs, ndbCposs, dbxpad, maxnsteps, gcost,
                            wrkmemtm, wrkmemaux,
                            tmpdpdiagbuffers, tmpdpbotbuffer);
                    MYCUDACHECKLAST;
                }//swift_dp

                //reuse memory wrkmemccd for best tfms (over references) written at slot 0;
                //NOTE: wrkmemccd is larger as the size of CCD data > that of tfms;
                SaveBestDPscoreAndTMAmongDPswifts
                    <<<nblcks_dpscmax,nthrds_dpscmax,0,streamproc>>>(
                        false/*WRITEFRAGINFO*/, (fragndx/* || xsndx*/)/*READSCORE*/, (speed>0)/*STEPx5*/,
                        ndbCstrs, ndbCposs, dbxpad,
                        maxnsteps, nlocsteps,  ysndx, xsndx, fragndx,
                        tmpdpdiagbuffers, wrkmemtm, wrkmemccd/*target tfms*/, wrkmemaux);
                MYCUDACHECKLAST;
            }//fragments

            //perform complete DP with backtracking: launch blocks along data block anti-diagonals;
            //using tfms temporarily written in wrkmemccd;
            //nblkdiags, total number of diagonals:
            for(uint d = 0; d < nblkdiags; d++)
            {
                ExecDPwBtck3264x<vANCHORRGN,vBANDED,true/*GAP0*/,D02IND_DPSCAN>
                    <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                        d, ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                        0/*ndx for wrkmemccd*/, gcost,
                        wrkmemccd/*as wrkmemtm*/, wrkmemaux,
                        tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
                MYCUDACHECKLAST;
            }

            //get the alignments written to tmpdpalnpossbuffer
            BtckToMatched32x<vANCHORRGN,vBANDED>
                <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                    ndbCstrs, ndbCposs, dbxpad,
                    maxnsteps, stepnumber/*ndx for tmpdpalnpossbuffer*/,
                    btckdata, wrkmemaux, tmpdpalnpossbuffer);
            MYCUDACHECKLAST;

            bool lastiteration = (nstepsy <= ysndx + 1) && (nstepsx <= xsndx + (int)maxnsteps);

            stepnumber++;

            if((int)maxnsteps <= stepnumber || lastiteration)
            {
                //wrkmemtmibest contains best tfms over all DP recurses
                stagedp_score_alignment3(
                    streamproc,
                    maxnsteps, stepnumber,
                    qystr1len, dbstr1len,
                    nqystrs, ndbCstrs, ndbCposs, dbxpad,
                    tmpdpdiagbuffers, tmpdpalnpossbuffer,
                    wrkmem, wrkmemtmibest, wrkmemaux, wrkmem2, wrkmemtm
                );

                stepnumber = 0;
            }

        }//reference positions
    }//query positions

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    SaveBestScoreAndTMAmongBests<false/*WRITEFRAGINFO*/>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemtmibest, tfmmem, wrkmemaux);
    MYCUDACHECKLAST;
}





// -------------------------------------------------------------------------
// stagedp_tfms_from_fragments: find transformation matrices from gapless 
// fragments in query and reference structures
//
inline
void stagedp::stagedp_tfms_from_fragments(
    cudaStream_t streamproc,
    int qryfragfct, int rfnfragfct, int fragndx,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    float* __restrict__ wrkmem,
    float* __restrict__ /*wrkmemaux*/,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx_var, const dim3& nthrds_ccmtx_var,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    const int depth = CLOptions::GetC_DEPTH();

    if(depth == CLOptions::csdShallow) {
    //initialize memory for calculating cross covariance matrices
        // (required for each iteration);
        //NOTE: initialization only for participating pairs may be not
        // advantageous: requires additional 2*CUS1_TBINITSP_CCDINIT_XFCT 
        // reads per block and arithmetics;
        InitCCData0_var<true/*STEPx5*/><<<nblcks_init,nthrds_init,0,streamproc>>>(
            ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,  wrkmem);
        MYCUDACHECKLAST;

        //calculate cross-covariance matrices with unrolling
        CalcCCMatrices64_var<true/*STEPx5*/>
            <<<nblcks_ccmtx_var,nthrds_ccmtx_var,0,streamproc>>>(
                nqystrs,  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,  wrkmem);
        MYCUDACHECKLAST;

        //copy CC data to section 2 of working memory to enable
        // efficient structure-specific calculation
        CopyCCDataToWrkMem2_var<READNPOS_NOREAD,true/*STEPx5*/>
            <<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
                ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
                wrkmem/*in*/, wrkmem2/*out*/);
        MYCUDACHECKLAST;
    }
    else {
        InitCCData0_var<false><<<nblcks_init,nthrds_init,0,streamproc>>>(
            ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,  wrkmem);
        MYCUDACHECKLAST;

        CalcCCMatrices64_var<false>
            <<<nblcks_ccmtx_var,nthrds_ccmtx_var,0,streamproc>>>(
                nqystrs,  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,  wrkmem);
        MYCUDACHECKLAST;

        CopyCCDataToWrkMem2_var<READNPOS_NOREAD,false>
            <<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
                ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
                wrkmem/*in*/, wrkmem2/*out*/);
        MYCUDACHECKLAST;
    }

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;
}



// -------------------------------------------------------------------------
// stagedp_score_alignment3: calculate superposition score for given 
// alignments between query and reference structures in three iterations;
// maxnsteps, max #steps that can be executed in parallel for one 
// query-reference pair; it corresponds to #different alignments for a pair;
// actualnsteps, actual #steps (alignments) performed before this call;
//
void stagedp::stagedp_score_alignment3(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint actualnsteps,
    const uint qystr1len, const uint dbstr1len,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemtmibest,
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


    stagedp_scorealn3_subiter1(
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

    stagedp_scorealn3_subiter2(
        streamproc,
        maxnsteps,
        nqystrs, ndbCstrs, ndbCposs, dbxpad,
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
        nblcks_tfm, nthrds_tfm);

    stagedp_scorealn3_subiter3(
        streamproc,
        maxnsteps,
        nqystrs, ndbCstrs, ndbCposs, dbxpad,
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
        nblcks_tfm, nthrds_tfm);
}

// -------------------------------------------------------------------------
// stagedp_scorealn3_subiter1: subiteration 1 of scoring alignments in three
// iterations;
//
inline
void stagedp::stagedp_scorealn3_subiter1(
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
    //initialize memory for calculating cross covariance matrices
    InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, maxnsteps,  wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64_DPscan<<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
        wrkmemaux, tmpdpalnpossbuffer, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable efficient 
    //structure-specific calculation; READNPOS_NOREAD, do not verify whether
    //#positions on which tfms are calculated has changed:
    CopyCCDataToWrkMem2_DPscan<READNPOS_NOREAD>
        <<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagedp_scorealn3_subiter2: subiteration 2 of scoring alignments in three
// iterations;
//
inline
void stagedp::stagedp_scorealn3_subiter2(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_findd2, const dim3& nthrds_findd2,
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_scores, const dim3& nthrds_scores,
    const dim3& nblcks_savetm, const dim3& nthrds_savetm,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores and temporarily save distances
    CalcScoresUnrl_DPscan<SAVEPOS_SAVE,CHCKALNLEN_NOCHECK>
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


    //find required minimum distances for adjusting rotation
    FindD02ThresholdsCCM_DPscan<READCNST_CALC>
        <<<nblcks_findd2,nthrds_findd2,0,streamproc>>>(
            ndbCstrs, ndbCposs,  maxnsteps,
            tmpdpdiagbuffers, wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory before calculating cross-covariance matrices
    InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, maxnsteps,  wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling;
    //include only positions within given distances
    CalcCCMatrices64_DPscanExtended<READCNST_CALC>
        <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
            tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory for efficient 
    //structure-specific calculation
    CopyCCDataToWrkMem2_DPscan<READNPOS_READ>
        <<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagedp_scorealn3_subiter2: subiteration 3 of scoring alignments in three
// iterations;
//
inline
void stagedp::stagedp_scorealn3_subiter3(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_findd2, const dim3& nthrds_findd2,
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_scores, const dim3& nthrds_scores,
    const dim3& nblcks_savetm, const dim3& nthrds_savetm,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores and temporarily save distances
    CalcScoresUnrl_DPscan<SAVEPOS_SAVE,CHCKALNLEN_CHECK>
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


    //find required minimum distances for adjusting rotation
    FindD02ThresholdsCCM_DPscan<READCNST_CALC2>
        <<<nblcks_findd2,nthrds_findd2,0,streamproc>>>(
            ndbCstrs, ndbCposs,  maxnsteps,
            tmpdpdiagbuffers, wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory before calculating cross-covariance matrices
    InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, maxnsteps,  wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling;
    //include only positions within given distances
    CalcCCMatrices64_DPscanExtended<READCNST_CALC2>
        <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
            tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory for efficient 
    //structure-specific calculation
    CopyCCDataToWrkMem2_DPscan<READNPOS_READ>
        <<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;

    //final calculation of scores:
    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores; no saving of distances:
    CalcScoresUnrl_DPscan<SAVEPOS_NOSAVE,CHCKALNLEN_CHECK>
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
