/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/scoring.cuh"
#include "custage1.cuh"

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
void stage1::run_stage1(
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
    uint* __restrict__ globvarsbuf)
{
//     //blockIdx.x is the refn. structure serial number;
//     //blockIdx.y is the query serial number;
// 
//     INTYPE qrylen, dbstrlen;//lengths
//     LNTYPE qrydst, dbstrdst;//distances in positions to respective structures (beginnings)
//     uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;

//     if(dbstrndx < ndbCstrs) {
//         dbstrlen = ((INTYPE*)(dc_pm2dvfields_[ndx_dbs_dc_pm2dvfields_*pmv2DTotFlds+pps2DLen]))[dbstrndx];
//         dbstrdst = ((LNTYPE*)(dc_pm2dvfields_[ndx_dbs_dc_pm2dvfields_*pmv2DTotFlds+pps2DDist]))[dbstrndx];
//         qrylen = ((INTYPE*)(dc_pm2dvfields_[ndx_qrs_dc_pm2dvfields_*pmv2DTotFlds+pps2DLen]))[blockIdx.y];
//         qrydst = ((LNTYPE*)(dc_pm2dvfields_[ndx_qrs_dc_pm2dvfields_*pmv2DTotFlds+pps2DDist]))[blockIdx.y];
//     }


//     //{{parameter initilization based on the structure lengths
//     int minlen = myhdmin(qrylen, dbstrlen);//NOTE: assumed >=3
//     float d0_min = 0.5f;
//     float dcu0 = 4.25f;
//     float d0 = 0.168f;
//     float lnorm = minlen;
// 
//     if(lnorm > 19.f)
//         //d0 = 1.24f * powf(lnorm - 15.f, 1.f/3.f) - 1.8f;
//         d0 = 1.24f * cbrtf(lnorm - 15.f) - 1.8f;
//     d0_min = d0 + 0.8f;
//     d0 = d0_min;
// 
//     float d0_search = myhdmin(d0, 8.f);
//     d0_search = myhdmax(d0_search, 4.5f);
// 
//     //float score_d8 = 1.5f * powf(lnorm, 0.3f) + 3.5f;
//     float score_d8 = 1.5f * expf(0.3f * logf(lnorm)) + 3.5f;
//     //}}

    //NOTE: minlen, minimum of the largest structures to compare, assumed >=3
//     int minlen = myhdmin(qystr1len, dbstr1len);
//     int minaln = myhdmax(minlen >> 1, 5);
//     int n1 = minaln - dbstr1len;
//     int n2 = qystr1len - minaln;

    //minimum length among largest
    int minlenmax = myhdmin(qystr1len, dbstr1len);
    //minimum length among smallest
    int minlenmin = myhdmin(qystrnlen, dbstrnlen);
    int minalnmin = myhdmax(minlenmin >> 1, 5);
    int n1 = minalnmin - dbstr1len;
    int n2 = qystr1len - minalnmin;


    //execution configuration for tfm matrix initialization:
    //each block processes one query and CUS1_TBINITSP_TFMINIT_XFCT references:
    dim3 nthrds_tfminit(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    dim3 nblcks_tfminit(
        (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
        nqystrs,1);

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs,1);

    //initialize memory for transformation matrices (once in the stage);
    InitTfmMatrices<<<nblcks_tfminit,nthrds_tfminit,0,streamproc>>>(
        ndbCstrs,
        tfmmem);
    MYCUDACHECKLAST;

    //initialize memory for scores (best, once in the stage);
    InitScores<INITOPT_ALL><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,
        wrkmemaux);
    MYCUDACHECKLAST;


    //execution configuration for initialization:
    //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
    dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
    dim3 nblcks_init(
        (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
        nqystrs,1);

    //execution configuration for reduction:
    //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_ccmtx(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
    dim3 nblcks_ccmtx(
        (minlenmax + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
        ndbCstrs,nqystrs);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
    dim3 nblcks_copyto(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs,1);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
    dim3 nblcks_copyfrom(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs,1);

    //execution configuration for calculating transformation matrices:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
    dim3 nblcks_tfm(
        (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
        nqystrs,1);

    //execution configuration for calculating scores (reduction):
    //block processes CUS1_TBSP_SCORE_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_scores(CUS1_TBSP_SCORE_XDIM,1,1);
    dim3 nblcks_scores(
        (minlenmax + CUS1_TBSP_SCORE_XDIMLGL - 1)/CUS1_TBSP_SCORE_XDIMLGL,
        ndbCstrs,nqystrs);


    for(; n1 <= n2; n1 += stepinit) {
        stage1_subiter1(
            streamproc,
            ndbCstrs,
            wrkmem, wrkmem2, tfmmem,
            myhdmax(0,n1)/*qrypos*/,
            myhdmax(-n1,0)/*refpos*/,
            nblcks_init, nthrds_init,
            nblcks_ccmtx, nthrds_ccmtx,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);

        stage1_subiter2(
            streamproc,
            ndbCstrs,
            wrkmem, wrkmemaux, wrkmem2, tfmmem,
            myhdmax(0,n1)/*qrypos*/,
            myhdmax(-n1,0)/*refpos*/,
            nblcks_init, nthrds_init,
            nblcks_ccmtx,nthrds_ccmtx,
            nblcks_scinit, nthrds_scinit,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);

        stage1_subiter3(
            streamproc,
            ndbCstrs,
            wrkmem, wrkmemaux, wrkmem2, tfmmem,
            myhdmax(0,n1)/*qrypos*/,
            myhdmax(-n1,0)/*refpos*/,
            nblcks_init, nthrds_init,
            nblcks_ccmtx, nthrds_ccmtx,
            nblcks_scinit, nthrds_scinit,
            nblcks_scores, nthrds_scores,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);
    }
}

// -------------------------------------------------------------------------
// stage1_subiter1: subiteration 1 of stage1: calculate cross-covariances 
// and rotation matrices
//
inline
void stage1::stage1_subiter1(
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
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //initialize memory for calculating cross covariance matrices
    // (required for each iteration);
    //NOTE: initialization only for participating pairs may be not
    // advantageous: requires additional 2*CUS1_TBINITSP_CCDINIT_XFCT 
    // reads per block and arithmetics;
    InitCCData<<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, wrkmem);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64<<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable
    // efficient structure-specific calculation
    CopyCCDataToWrkMem2<READNPOS_NOREAD><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs, wrkmem2/*in*/, tfmmem/*out*/);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage1_subiter2: subiteration 2 of stage1: unconditionally calculate 
// and save scores, calculate cross-covariances for alignment positions 
// within given distances, and optionally calculate rotation matrices
//
inline
void stage1::stage1_subiter2(
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
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //initialize memory before calculating cross-covariances
    InitCCData<<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, wrkmem);
    MYCUDACHECKLAST;

    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    Transform_CalcCCM_Score64<READCNST_CALC><<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, 0,  tfmmem, wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    SaveBestScore<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, wrkmemaux);
    MYCUDACHECKLAST;

    for(int i = 0; i < N_ITER_DST_INCREASE; i++) {
        InitCCData<<<nblcks_init,nthrds_init,0,streamproc>>>(
            ndbCstrs, wrkmem);
        MYCUDACHECKLAST;

        InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, wrkmemaux);
        MYCUDACHECKLAST;

        Transform_CalcCCM_Score64<READCNST_READ_CHCK><<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
            ndbCstrs, qrypos, rfnpos, i,  tfmmem, wrkmem, wrkmemaux);
        MYCUDACHECKLAST;
    }

    //copy CC data to section 2 of working memory to enable
    // efficient structure-specific calculation
    CopyCCDataToWrkMem2<READNPOS_READ><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs, wrkmem2/*in*/, tfmmem/*out*/);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage1_subiter3: subiteration 3 of stage1: unconditionally calculate 
// and save scores, calculate cross-covariances for alignment positions 
// within given distances, optionally calculate rotation matrices, and
// calculate and save scores only 
//
inline
void stage1::stage1_subiter3(
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
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //initialize memory before calculating cross-covariances
    InitCCData<<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs,
        wrkmem);
    MYCUDACHECKLAST;

    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, wrkmemaux);
    MYCUDACHECKLAST;

    Transform_CalcCCM_Score64<READCNST_CALC2><<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, 0,  tfmmem, wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    SaveBestScore<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, wrkmemaux);
    MYCUDACHECKLAST;

    for(int i = 0; i < N_ITER_DST_INCREASE2; i++) {
        InitCCData<<<nblcks_init,nthrds_init,0,streamproc>>>(
            ndbCstrs,
            wrkmem);
        MYCUDACHECKLAST;

        InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, wrkmemaux);
        MYCUDACHECKLAST;

        Transform_CalcCCM_Score64<READCNST_READ_CHCK2><<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
            ndbCstrs, qrypos, rfnpos, i,  tfmmem, wrkmem, wrkmemaux);
        MYCUDACHECKLAST;
    }

    //copy CC data to section 2 of working memory to enable
    // efficient structure-specific calculation
    CopyCCDataToWrkMem2<READNPOS_READ><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs, wrkmem2/*in*/, tfmmem/*out*/);
    MYCUDACHECKLAST;

    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, wrkmemaux);
    MYCUDACHECKLAST;

    CalcScoresUnrl<<<nblcks_scores,nthrds_scores,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, tfmmem, wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    SaveBestScore<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, qrypos, rfnpos, wrkmemaux);
    MYCUDACHECKLAST;
}
