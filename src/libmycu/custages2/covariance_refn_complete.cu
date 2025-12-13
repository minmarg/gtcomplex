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
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/covariance_dp_refn.cuh"
#include "libmycu/custages2/covariance_complete.cuh"
#include "covariance_refn_complete.cuh"

// =========================================================================
// FragmentBasedAlignmentRefinement: refine alignment and its boundaries 
// within the single kernel's actions to obtain favorable superposition;
// WRITEFRAGINFO, template parameter, flag of whether refined fragment 
// boundaries should be saved;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// MAX_NCHAINS, max #complex chains for this kernel to continue computation;
// nmaxconvit, maximum number of superposition iterations;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// nmaxsubfrags, total number of fragment lengths to consider;
// maxnsteps, total number of steps accommodated for reference structures;
// effnsteps, total logical (required) number of steps;
// sfragstep, step size to traverse subfragments;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<
    bool WRITEFRAGINFO,
    bool TFM_DINV,
    int MAX_NCHAINS>
__global__ 
void FragmentBasedAlignmentRefinement(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint nmaxsubfrags,
    const uint maxnsteps,
    const uint effnsteps,
    const int sfragstep,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    const uint dbstrndx = blockIdx.x;//reference serial number
    const uint sfragfctxndx = blockIdx.y;//fragment factor x fragment length index
    const uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds+1};
    __shared__ float ccmCache[
        smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM + twmvEndOfCCDataExt * 2 + nTTranformMatrix];
//     __shared__ float ccmLast[twmvEndOfCCDataExt];
//     __shared__ float tfmCache[twmvEndOfCCDataExt];//twmvEndOfCCDataExt>nTTranformMatrix
    float* ccmLast = ccmCache + smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM;
    float* tfmCache = ccmLast + twmvEndOfCCDataExt;//twmvEndOfCCDataExt>nTTranformMatrix
    float* tfmBest = tfmCache + twmvEndOfCCDataExt;//of size nTTranformMatrix

    int qrylen, dbstrlen;//query and reference lengths
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos, rfnpos;
    int sfragpos, fraglen;


    if(MAX_NCHAINS > 0) {
        if(threadIdx.x == 0) {
            int qrycpxndx = GetQueryStrField<INTYPE,pps2DCpxI>(qryndx);
            int dbcpxndx = GetDbStrField<INTYPE,pps2DCpxI>(dbstrndx);
            int qrycpxN = GetQueryStrField<INTYPE,pcx2DN>(qrycpxndx);//#chains
            int dbcpxN = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);//#chains
            ccmCache[4] = (MAX_NCHAINS < qrycpxN) && (MAX_NCHAINS < dbcpxN);
        }
        __syncthreads();
        //NOTE: ccmCache[4] cannot be overwritten until the next sync!
        if(ccmCache[4]) return;//block exits
    }


    //convergence check for different molecule types:
    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    //(NOTE:any type of convergence applies);
    if(ccmCache[6]) return;
    //NOTE: no sync as long ccmCache[6] not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        GetQueryLenDst(qryndx, (int*)ccmCache + 2);
    }

    if(threadIdx.x == tawmvQRYpos + 8 || threadIdx.x == tawmvRFNpos + 8) {
        //NOTE: reuse ccmCache to read positions;
        //NOTE: written at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-8) * ndbCstrs + dbstrndx];
    }

    __syncthreads();


    dbstrlen = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylen = ((int*)ccmCache)[2]; qrydst = ((int*)ccmCache)[3];
    qrypos = ccmCache[tawmvQRYpos+8]; rfnpos = ccmCache[tawmvRFNpos+8];

    __syncthreads();


    if(qrylen <= qrypos || dbstrlen <= rfnpos)
        return;//all threads in the block exit

    //threshold calculated for the original lengths
    const float d0 = GetD0(qrylen, dbstrlen);
    const float d02 = SQRD(d0);
    const float d82 = GetD82(qrylen, dbstrlen);
    float best = 0.0f;//best score obtained

    for(uint fctxndx = sfragfctxndx; fctxndx < effnsteps; fctxndx += maxnsteps)
    {
        const uint sfragfct = fctxndx / nmaxsubfrags;//fragment factor
        const uint sfragndx = fctxndx - sfragfct * nmaxsubfrags;//fragment length index
        sfragpos = sfragfct * sfragstep;

        fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
        if(fraglen < 1) continue;

        if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
           dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen) continue;


        float dst32 = CP_LARGEDST;

        CalcCCMatrices64Refined_Complete<smidim,neffds>(
            qrydst, dbstrdst, fraglen,
            qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
            ccmCache);


        for(int cit = 0; cit < nmaxconvit + 2; cit++)
        {
            if(0 < cit) {
                CalcCCMatrices64RefinedExtended_Complete<smidim,neffds>(
                    (cit < 2)? READCNST_CALC: READCNST_CALC2,
                    qryndx, ndbCposs, maxnsteps, sfragfctxndx, qrydst, dbstrdst,
                    qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                    tmpdpdiagbuffers, ccmCache);

                CheckConvergence64Refined_Complete(ccmCache, ccmLast);
                if(ccmLast[0]) break;//converged
                __syncthreads();//prevent overwriting ccmLast[0]
            }

            //NOTE: synced above and below before ccmCache gets updated;
            if(ccmCache[twmvNalnposs] < 1.0f) break;

            SaveCCMData_Complete(ccmCache, tfmCache, ccmLast);
            //NOTE: tfmCache updated by the first warp; 
            //NOTE: CalcTfmMatrices_Complete uses only the first warp;
            //NOTE: ccmLast not used until the first syncthreads below;
            __syncwarp();

            CalcTfmMatrices_Complete<TFM_DINV>(tfmCache, qrylen, dbstrlen);
            //all threads synced and see the tfm

            CalcScoresUnrlRefined_Complete(
                (cit < 1)? READCNST_CALC: READCNST_CALC2,
                qryndx, ndbCposs, maxnsteps, sfragfctxndx, qrydst, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
                tmpdpdiagbuffers, tfmCache, ccmCache+1);

            //distance threshold for at least three aligned pairs:
            dst32 = ccmCache[2];

            //NOTE: no sync inside:
            SaveLocalBestScoreAndTM(best, ccmCache[1]/*score*/, tfmCache, tfmBest);

            //sync all threads to see dst32 (and prevent overwriting the cache):
            __syncthreads();
        }

    }//for(;fctxndx < effnsteps;)

    //NOTE: synced either after the last cit or convergence check:
    SaveBestScoreAndTM_Complete<WRITEFRAGINFO,false/*CONDITIONAL*/>(
        best,  qryndx, dbstrndx, ndbCstrs, 
        maxnsteps, sfragfctxndx, 0/*sfragndx, unused*/, 0/*sfragpos, unused*/,
        tfmBest, wrkmemtmibest, wrkmemaux);
}

// -------------------------------------------------------------------------
// Instantiations
// 
#define INSTANTIATE_FragmentBasedAlignmentRefinement( \
        tpWRITEFRAGINFO,tpTFM_DINV,tpMAX_NCHAINS) \
    template __global__ void FragmentBasedAlignmentRefinement \
        <tpWRITEFRAGINFO,tpTFM_DINV,tpMAX_NCHAINS>( \
            const int nmaxconvit, /*const uint nqystrs,*/ \
            const uint ndbCstrs, const uint ndbCposs, \
            const uint nmaxsubfrags, const uint maxnsteps, const uint effnsteps, \
            const int sfragstep, \
            float* __restrict__ tmpdpdiagbuffers, \
            float* __restrict__ wrkmemtmibest, \
            float* __restrict__ wrkmemaux);

INSTANTIATE_FragmentBasedAlignmentRefinement(false/* true */,false,0);
INSTANTIATE_FragmentBasedAlignmentRefinement(false/* true */,true,0);

INSTANTIATE_FragmentBasedAlignmentRefinement(false/* true */,false,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);
INSTANTIATE_FragmentBasedAlignmentRefinement(false/* true */,true,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);

// =========================================================================

// =========================================================================
// FragmentBasedAlignmentRefinementC2C: refine chain-to-chain alignments to
// obtain favorable superposition;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// MAX_NCHAINS, max #complex chains for this kernel to continue computation;
// nmaxconvit, maximum number of superposition iterations;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// nmaxsubfrags, total number of fragment lengths to consider;
// maxnsteps, total number of steps that should be performed for each reference structure;
// effnsteps, total logical (required) number of steps;
// sfragstep, step size to traverse subfragments;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<
    bool TFM_DINV,
    int MAX_NCHAINS>
__global__ 
void FragmentBasedAlignmentRefinementC2C(
    const int nmaxconvit,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint nmaxsubfrags,
    const uint maxnsteps,
    const uint effnsteps,
    const int sfragstep,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    const uint dbstrndx = blockIdx.x;//reference serial number
    const uint sfragfctxndx = blockIdx.y;//fragment factor x fragment length index
    const uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds+1};
    __shared__ float ccmCache[
        smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM + twmvEndOfCCDataExt * 2 + nTTranformMatrix];
//     __shared__ float ccmLast[twmvEndOfCCDataExt];
//     __shared__ float tfmCache[twmvEndOfCCDataExt];//twmvEndOfCCDataExt>nTTranformMatrix
    float* ccmLast = ccmCache + smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM;
    float* tfmCache = ccmLast + twmvEndOfCCDataExt;//twmvEndOfCCDataExt>nTTranformMatrix
    float* tfmBest = tfmCache + twmvEndOfCCDataExt;//of size nTTranformMatrix

    int qrylen, dbstrlen;//query and reference lengths
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos, rfnpos;
    int sfragpos, fraglen;


    if(MAX_NCHAINS > 0) {
        if(threadIdx.x == 0) {
            int qrycpxndx = GetQueryStrField<INTYPE,pps2DCpxI>(qryndx);
            int dbcpxndx = GetDbStrField<INTYPE,pps2DCpxI>(dbstrndx);
            int qrycpxN = GetQueryStrField<INTYPE,pcx2DN>(qrycpxndx);//#chains
            int dbcpxN = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);//#chains
            ccmCache[4] = (MAX_NCHAINS < qrycpxN) && (MAX_NCHAINS < dbcpxN);
        }
        __syncthreads();
        //NOTE: ccmCache[4] cannot be overwritten until the next sync!
        if(ccmCache[4]) return;//block exits
    }


    //convergence check for different molecule types:
    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    //(NOTE:any type of convergence applies);
    if(ccmCache[6]) return;
    //NOTE: no sync as long ccmCache[6] not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        GetQueryLenDst(qryndx, (int*)ccmCache + 2);
    }

    if(threadIdx.x == tawmvQRYpos + 8 || threadIdx.x == tawmvRFNpos + 8) {
        //NOTE: reuse ccmCache to read positions;
        //NOTE: written at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-8) * ndbCstrs + dbstrndx];
    }

    __syncthreads();


    dbstrlen = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylen = ((int*)ccmCache)[2]; qrydst = ((int*)ccmCache)[3];
    qrypos = ccmCache[tawmvQRYpos+8]; rfnpos = ccmCache[tawmvRFNpos+8];

    __syncthreads();


    if(qrylen <= qrypos || dbstrlen <= rfnpos)
        return;//all threads in the block exit

    //threshold calculated for the original lengths
    const float d0 = GetD0(qrylen, dbstrlen);
    const float d02 = SQRD(d0);
    const float d82 = GetD82(qrylen, dbstrlen);
    float best = 0.0f;//best score obtained


    for(uint fctxndx = sfragfctxndx; fctxndx < effnsteps; fctxndx += maxnsteps)
    {
        uint sfragfct = fctxndx / nmaxsubfrags;//fragment factor
        uint sfragndx = fctxndx - sfragfct * nmaxsubfrags;//fragment length index
        sfragpos = sfragfct * sfragstep;

        fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
        if(fraglen < 1) continue;

        if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
           dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen) continue;


        float dst32 = CP_LARGEDST;

        CalcCCMatrices64Refined_Complete<smidim,neffds>(
            qrydst, dbstrdst, fraglen,
            qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
            ccmCache);


        for(int cit = 0; cit < nmaxconvit + 2; cit++)
        {
            if(0 < cit) {
                CalcCCMatrices64RefinedExtended_Complete<smidim,neffds>(
                    (cit < 2)? READCNST_CALC: READCNST_CALC2,
                    qryndx, ndbCposs, maxnsteps, sfragfctxndx, qrydst, dbstrdst,
                    qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                    tmpdpdiagbuffers, ccmCache);

                CheckConvergence64Refined_Complete(ccmCache, ccmLast);
                if(ccmLast[0]) break;//converged
                __syncthreads();//prevent overwriting ccmLast[0]
            }

            //NOTE: synced above and below before ccmCache gets updated;
            if(ccmCache[twmvNalnposs] < 1.0f) break;

            SaveCCMData_Complete(ccmCache, tfmCache, ccmLast);
            //NOTE: tfmCache updated by the first warp; 
            //NOTE: CalcTfmMatrices_Complete uses only the first warp;
            //NOTE: ccmLast not used until the first syncthreads below;
            __syncwarp();

            CalcTfmMatrices_Complete<TFM_DINV>(tfmCache, qrylen, dbstrlen);
            //all threads synced and see the tfm

            CalcScoresUnrlRefined_Complete(
                (cit < 1)? READCNST_CALC: READCNST_CALC2,
                qryndx, ndbCposs, maxnsteps, sfragfctxndx, qrydst, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
                tmpdpdiagbuffers, tfmCache, ccmCache+1);

            //distance threshold for at least three aligned pairs:
            dst32 = ccmCache[2];

            //NOTE: no sync inside:
            SaveLocalBestScoreAndTM(best, ccmCache[1]/*score*/, tfmCache, tfmBest);

            //sync all threads to see dst32 (and prevent overwriting the cache):
            __syncthreads();
        }

    }//for(;fctxndx < effnsteps;)

    //NOTE: synced either after the last cit or convergence check:
    SaveBestScoreAndTM_Complete<false/*WRITEFRAGINFO*/,false/*CONDITIONAL*/>(
        best,  qryndx, dbstrndx, ndbCstrs, 
        maxnsteps, sfragfctxndx, 0/*sfragndx, unused*/, sfragpos,
        tfmBest, wrkmemtmibest, wrkmemaux);
}

// -------------------------------------------------------------------------
// Instantiations
// 
#define INSTANTIATE_FragmentBasedAlignmentRefinementC2C( \
        tpTFM_DINV,tpMAX_NCHAINS) \
    template __global__ void FragmentBasedAlignmentRefinementC2C \
        <tpTFM_DINV,tpMAX_NCHAINS>( \
            const int nmaxconvit, const uint ndbCstrs, const uint ndbCposs, \
            const uint nmaxsubfrags, const uint maxnsteps, const uint effnsteps, \
            const int sfragstep, \
            float* __restrict__ tmpdpdiagbuffers, \
            float* __restrict__ wrkmemtmibest, \
            float* __restrict__ wrkmemaux);

INSTANTIATE_FragmentBasedAlignmentRefinementC2C(false,0);
INSTANTIATE_FragmentBasedAlignmentRefinementC2C(true,0);

INSTANTIATE_FragmentBasedAlignmentRefinementC2C(false,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);
INSTANTIATE_FragmentBasedAlignmentRefinementC2C(true,CUS1_TBSP_CPXSCORE_MAX_NCHAINS);

// =========================================================================
