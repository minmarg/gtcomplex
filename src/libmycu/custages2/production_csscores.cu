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
#include "libmycu/cucom/cutemplates.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/cudp/dpw_complex_com.cuh"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/custages2/covariance_production_dp_refn_complete.cuh"
#include "production_csscores.cuh"

// -------------------------------------------------------------------------
// ProductionCSScores: calculate chain-specific TM-scores & RMSDs and write
// them to memory;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, total number of steps performed for each reference structure;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// tfmmem, memory for transformation matrices;
// alndatamem, memory for full alignment information, including scores;
// wrkmem2, working memory for chain scores and assignments;
// complexstepnumber, step number corresponding to memory section for 
// matched coordinates written;
// assgmaxnsteps, max number of steps for complex chain assignments;
// assgstepnumber, step number for complex chain assignments;
// 
__global__ 
void ProductionCSScores(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tfmmem,
    float* __restrict__ alndatamem,
    const float* __restrict__ wrkmem2,
    const uint complexstepnumber,
    const uint assgmaxnsteps,
    const uint assgstepnumber)
{
    enum {
        lXDIM = CUDP_PRODUCTION_CS_SCORES_DIM_X,
        neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = twmvEndOfCCDataExtPlus
    };
    uint dbstrndx = blockIdx.x;//reference serial number
    uint qryndx = blockIdx.y;//query serial number
    uint dbcpxndx = dbstrndx;//reference complex serial number
    uint qrycpxndx = qryndx;//query complex serial number
    //cache for scores: add 1 to avoid bank conflicts:
    __shared__ float tfmCache[nTTranformMatrix + 1 + lXDIM * smidim];
    float* scvCache = tfmCache + nTTranformMatrix + 1;

    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    enum {qrypos = 0, rfnpos = 0, tmpcslot = tawmvEndOfCCMDat/*<16-28*/};


    if(threadIdx.x ==  0) scvCache[4] = GetQueryStrField<INTYPE,pps2DCpxI>(qryndx);
    if(threadIdx.x == 32) scvCache[5] = GetDbStrField<INTYPE,pps2DCpxI>(dbstrndx);
    __syncthreads();
    //NOTE: scvCache[4-5] cannot be overwritten until the next sync!
    qrycpxndx = scvCache[4];//uint<-float
    dbcpxndx = scvCache[5];//uint<-float

    if(threadIdx.x == 0) {
        uint mloc = ((qrycpxndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        scvCache[6] = wrkmemaux[mloc + dbcpxndx];
    }

    __syncthreads();

    if(((int)(scvCache[6])) & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as cache cell for convergence is not overwritten;


    if(ChainsWithinAssignment<lXDIM/*xdim*/,1/*ydim*/>(
            qrycpxndx, dbcpxndx,  qryndx, dbstrndx,  ndbCstrs,
            assgmaxnsteps, assgstepnumber,
            wrkmem2, scvCache) == 0)
        //all threads exit if chains are not mutually assigned;
        return;

    //reuse cache
    if(threadIdx.x == 0) {
        ((int*)scvCache)[tmpcslot+0] = GetDbStrLength(dbstrndx);
        ((int*)scvCache)[tmpcslot+1] = GetDbStrDst(dbstrndx);
    }

    //NOTE: read chain-specific number of aligned residues
    if(threadIdx.x == tawmvNAlnPoss) {
        //NOTE: reuse cache to read values written at sfragfct==0:
        uint mloc0 = ((qrycpxndx * maxnsteps + complexstepnumber) * nTAuxWorkingMemoryVars) * ndbCstrs;
        scvCache[threadIdx.x] = wrkmemaux[mloc0 + threadIdx.x * ndbCstrs + dbstrndx];
    }

#if (lXDIM >= 128)
    if(64 <= threadIdx.x && threadIdx.x < nTTranformMatrix + 64) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qrycpxndx * ndbCstrs + dbcpxndx) * nTTranformMatrix;
        tfmCache[threadIdx.x-64] = tfmmem[mloc0 + threadIdx.x-64];
    }
    if(threadIdx.x == 96)
        ((int*)scvCache)[tmpcslot+2] = GetQueryLength(qryndx);
    if(threadIdx.x == 32) {
        ((int*)scvCache)[tmpcslot+3] = qrydst = GetQueryDst(qryndx);
        ((int*)scvCache)[tmpcslot+4] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(qrydst);
    }
#elif (lXDIM >= 64)
    if(32 <= threadIdx.x && threadIdx.x < nTTranformMatrix + 32) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qrycpxndx * ndbCstrs + dbcpxndx) * nTTranformMatrix;
        tfmCache[threadIdx.x-32] = tfmmem[mloc0 + threadIdx.x-32];
    }
    if(threadIdx.x == 32) {
        ((int*)scvCache)[tmpcslot+2] = GetQueryLength(qryndx);
        ((int*)scvCache)[tmpcslot+3] = qrydst = GetQueryDst(qryndx);
        ((int*)scvCache)[tmpcslot+4] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(qrydst);
    }
#else
    if(threadIdx.x < nTTranformMatrix) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qrycpxndx * ndbCstrs + dbcpxndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = tfmmem[mloc0 + threadIdx.x];
    }
    if(threadIdx.x == 0) {
        ((int*)scvCache)[tmpcslot+2] = GetQueryLength(qryndx);
        ((int*)scvCache)[tmpcslot+3] = qrydst = GetQueryDst(qryndx);
        ((int*)scvCache)[tmpcslot+4] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(qrydst);
    }
#endif

    __syncthreads();


    dbstrdst = ((int*)scvCache)[tmpcslot+1];
    qrydst = ((int*)scvCache)[tmpcslot+3];
    qrylen = dbstrlen = scvCache[tawmvNAlnPoss];
    dbstrlenorg = ((int*)scvCache)[tmpcslot+0];
    qrylenorg = ((int*)scvCache)[tmpcslot+2];
    //NOTE: type determined by one chain, as chains of different type not aligned!
    int moltype = GetMoleculeType(((int*)scvCache)[tmpcslot+4]);

    __syncthreads();


    CalcExtCCMatrices64_DPRefined_Complete<smidim,neffds>(
        qrycpxndx,  ndbCposs, dbxpad,  maxnsteps, complexstepnumber/*sfragfctxndx*/,
        dbstrdst, dbstrlen/*fraglen*/,  qrylen, dbstrlen,  qrypos, rfnpos,
        tmpdpalnpossbuffer, scvCache);
    //synced above; rmsd valid only for thread 0:
    float rmsd = CalcRMSD_Complete(scvCache);
    //no sync, as 1st warp synced above and the threads of the remaining
    //warps write only in their locations until the next sync;


    //threshold calculated for the original lengths
    const float d0 = GetD0fin(qrylenorg, dbstrlenorg, moltype);
    const float d02 = SQRD(d0);

    CalcCSscoresUnrl_Complete(
        qrycpxndx, ndbCposs, dbxpad, maxnsteps,
        qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos, d02,
        complexstepnumber,
        tmpdpalnpossbuffer, tfmCache, scvCache);

    const float best = scvCache[0];//score: synced inside the above function;

    //sync to not overwrite scvCache[0]:
    __syncthreads();


    //calculate the score for the larger structure of the two:
    //threshold calculated for the greater length
    const int greaterlen = myhdmax(qrylenorg, dbstrlenorg);
    const float g0 = GetD0fin(greaterlen, greaterlen, moltype);
    const float g02 = SQRD(g0);
    float gbest = best;//score calculated for the other structure

    if(qrylenorg != dbstrlenorg) {
        CalcCSscoresUnrl_Complete(
            qrycpxndx, ndbCposs, dbxpad, maxnsteps,
            qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos, g02,
            complexstepnumber,
            tmpdpalnpossbuffer, tfmCache, scvCache);
        gbest = scvCache[0];
    }


    //NOTE: scvCache not overwritten any more;
    //NOTE: write directly to production output memory:
    SaveBestQRCSscores_Complete(
        best, gbest,  d0, g0, rmsd,
        qryndx, dbstrndx, ndbCstrs, qrylenorg, dbstrlenorg,
        alndatamem);
}

// =========================================================================
// -------------------------------------------------------------------------
