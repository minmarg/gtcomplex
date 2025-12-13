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
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/cudp/dpw_complex_com.cuh"
#include "production_2tmscore.cuh"

// -------------------------------------------------------------------------
// Production2TMscores: calculate secondary TM-scores, 2TM-scores, and write
// them to memory;
// For complexes, chain-specific 2TM-scores are calculated and
// summed up in an appropriate memory location pre-initialized
// before starting calculations for each new chunk;
// COMPLEX, template parameter, flag of processing full complexes;
// NOTE: CPXTYPENUMBERSCT, memory step number section of alignment-determined complex types;
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
// complexstepnumber2, step number corresponding to memory section of 
// matched POSITIONS written;
// assgmaxnsteps, max number of steps for complex chain assignments;
// assgstepnumber, step number for complex chain assignments;
// 
template<int COMPLEX, int CPXTYPENUMBERSCT>
__global__ 
void Production2TMscores(
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
    const uint complexstepnumber2,
    const uint assgmaxnsteps,
    const uint assgstepnumber)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    uint qryndx = blockIdx.y;//query serial number
    uint dbcpxndx = dbstrndx;//reference complex serial number
    uint qrycpxndx = qryndx;//query complex serial number
    //cache for scores: add 1 to avoid bank conflicts:
    __shared__ float tfmCache[nTTranformMatrix + 1 + CUDP_PRODUCTION_2TMSCORE_DIM_X];
    float* scvCache = tfmCache + nTTranformMatrix + 1;

    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    enum {qrypos = 0, rfnpos = 0, tmpcslot = tawmvEndOfCCMDat/*<16-28*/};


    if(COMPLEX) {
        if(threadIdx.x ==  0) scvCache[4] = GetQueryStrField<INTYPE,pps2DCpxI>(qryndx);
        if(threadIdx.x == 32) scvCache[5] = GetDbStrField<INTYPE,pps2DCpxI>(dbstrndx);
        __syncthreads();
        //NOTE: scvCache[4-5] cannot be overwritten until the next sync!
        qrycpxndx = scvCache[4];//uint<-float
        dbcpxndx = scvCache[5];//uint<-float
    }

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


    if(COMPLEX &&
       ChainsWithinAssignment<CUDP_PRODUCTION_2TMSCORE_DIM_X/*xdim*/,1/*ydim*/>(
            qrycpxndx, dbcpxndx,  qryndx, dbstrndx,  ndbCstrs,
            assgmaxnsteps, assgstepnumber,
            wrkmem2, scvCache) == 0)
        //all threads exit if chains are not mutually assigned;
        return;

    if(COMPLEX == ctpvCOMPLEX) {
        if(threadIdx.x == 0) {
            //get complex length and CHAIN distance
            GetDbComplexAddressLength(dbcpxndx, ((int*)scvCache)[tmpcslot+1], ((int*)scvCache)[tmpcslot+0]);
            ((int*)scvCache)[tmpcslot+1] = GetDbStrDst(dbstrndx);
        }
#if (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 64)
        if(threadIdx.x == 32) {
#else
        if(threadIdx.x == 0) {
#endif
            //get complex length and CHAIN distance
            GetQueryComplexAddressLength(qrycpxndx, ((int*)scvCache)[tmpcslot+3], ((int*)scvCache)[tmpcslot+2]);
            ((int*)scvCache)[tmpcslot+3] = GetQueryDst(qryndx);
        }
#if (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 128)
        if(threadIdx.x == 96) {
#elif (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 64)
        if(threadIdx.x == 32) {
#else
        if(threadIdx.x == 0) {
#endif
            uint mloct = ((qrycpxndx * maxnsteps + CPXTYPENUMBERSCT) * nTAuxWorkingMemoryVars) * ndbCstrs;
            ((int*)scvCache)[tmpcslot+4] = wrkmemaux[mloct + tawmvNAlnPoss * ndbCstrs + dbcpxndx];
        }

    } else {
        if(threadIdx.x == 0) {
            ((int*)scvCache)[tmpcslot+0] = GetDbStrLength(dbstrndx);
            ((int*)scvCache)[tmpcslot+1] = GetDbStrDst(dbstrndx);
        }
#if (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 128)
        if(threadIdx.x == 96)
#elif (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 64)
        if(threadIdx.x == 32)
#else
        if(threadIdx.x == 0)
#endif
            ((int*)scvCache)[tmpcslot+2] = GetQueryLength(qryndx);
#if (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 64)
        if(threadIdx.x == 32) {
#else
        if(threadIdx.x == 0) {
#endif
            ((int*)scvCache)[tmpcslot+3] = qrydst = GetQueryDst(qryndx);
            //NOTE: type determined by one chain, as chains of different type not aligned!
            ((int*)scvCache)[tmpcslot+4] = GetMoleculeType(GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(qrydst));
        }
    }

    //NOTE: use a different warp for structure-specific-formatted data;
    //NOTE: read chain-specific number of aligned residues
#if (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 64)
    if(threadIdx.x == tawmvNAlnPoss + 32) {
#else
    if(threadIdx.x == tawmvNAlnPoss) {
#endif
        uint mloc0 = ((qrycpxndx * maxnsteps + complexstepnumber) * nTAuxWorkingMemoryVars) * ndbCstrs;
        scvCache[tawmvNAlnPoss] = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs + dbstrndx];
    }

#if (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 128)
    if(64 <= threadIdx.x && threadIdx.x < nTTranformMatrix + 64) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qrycpxndx * ndbCstrs + dbcpxndx) * nTTranformMatrix;
        tfmCache[threadIdx.x-64] = tfmmem[mloc0 + threadIdx.x-64];
    }
#elif (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 64)
    if(32 <= threadIdx.x && threadIdx.x < nTTranformMatrix + 32) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qrycpxndx * ndbCstrs + dbcpxndx) * nTTranformMatrix;
        tfmCache[threadIdx.x-32] = tfmmem[mloc0 + threadIdx.x-32];
    }
#else
    if(threadIdx.x < nTTranformMatrix) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qrycpxndx * ndbCstrs + dbcpxndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = tfmmem[mloc0 + threadIdx.x];
    }
#endif

    __syncthreads();


    dbstrdst = ((int*)scvCache)[tmpcslot+1];
    qrydst = ((int*)scvCache)[tmpcslot+3];
    qrylen = dbstrlen = scvCache[tawmvNAlnPoss];
    dbstrlenorg = ((int*)scvCache)[tmpcslot+0];
    qrylenorg = ((int*)scvCache)[tmpcslot+2];
    int moltype = (((int*)scvCache)[tmpcslot+4]);

    __syncthreads();

    //threshold calculated for the original lengths
    const float d0 = GetD0fin(qrylenorg, dbstrlenorg, moltype);
    const float d02 = SQRD(d0);


    Calc2TMscoresUnrl_Complete(
        qrycpxndx, ndbCposs, dbxpad, maxnsteps,
        qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos, d02,
        complexstepnumber, complexstepnumber2,
        tmpdpalnpossbuffer, tfmCache, scvCache);

    const float best = scvCache[0];//score: synced inside the above function;

    //sync for scvCache[0] not to be overwritten by other warps:
    __syncthreads();


    //calculate the score for the larger structure of the two:
    //threshold calculated for the greater length
    const int greaterlen = myhdmax(qrylenorg, dbstrlenorg);
    const float g0 = GetD0fin(greaterlen, greaterlen, moltype);
    const float g02 = SQRD(g0);
    float gbest = best;//score calculated for the other structure

    if(qrylenorg != dbstrlenorg) {
        Calc2TMscoresUnrl_Complete(
            qrycpxndx, ndbCposs, dbxpad, maxnsteps,
            qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos, g02,
            complexstepnumber, complexstepnumber2,
            tmpdpalnpossbuffer, tfmCache, scvCache);
        gbest = scvCache[0];
    }


    //NOTE: scvCache not overwritten any more;
    //NOTE: write directly to production output memory:
    //NOTE: for complexes, qrycpxndx and dbcpxndx determine location;
    if(COMPLEX == ctpvNO_COMPLEX || COMPLEX == ctpvCOMPLEX)
        SaveBestQR2TMscores_Complete<dp2oad2ScoreQ,dp2oad2ScoreR>(
            best, gbest, qrycpxndx, dbcpxndx, ndbCstrs, qrylenorg, dbstrlenorg,
            alndatamem);
    else if(COMPLEX == ctpvCOMPLEX_CS)
        SaveBestQR2TMscores_Complete<dp2oad2ScoreQ_C,dp2oad2ScoreR_C>(
            best, gbest, qryndx, dbstrndx, ndbCstrs, qrylenorg, dbstrlenorg,
            alndatamem);
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_Production2TMscores(tpCOMPLEX,tpCPXTYPENUMBERSCT) \
    template \
    __global__ void Production2TMscores<tpCOMPLEX,tpCPXTYPENUMBERSCT>( \
        const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        const float* __restrict__ wrkmemaux, \
        const float* __restrict__ tfmmem, \
        float* __restrict__ alndatamem, const float* __restrict__ wrkmem2, \
        const uint complexstepnumber, const uint complexstepnumber2, \
        const uint assgmaxnsteps, const uint assgstepnumber);

INSTANTIATE_Production2TMscores(ctpvNO_COMPLEX,sfin_cpx_complextypenumber2);
INSTANTIATE_Production2TMscores(ctpvCOMPLEX,sfin_cpx_complextypenumber2);
INSTANTIATE_Production2TMscores(ctpvCOMPLEX_CS,sfin_cpx_complextypenumber2);
// -------------------------------------------------------------------------
