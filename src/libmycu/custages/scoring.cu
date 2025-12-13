/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmymp/mpdp/mpdpbase.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cucom/mysort.cuh"
#include "libmycu/cumath/cumath.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/transform.cuh"
#include "scoring.cuh"

// -------------------------------------------------------------------------
// SetCurrentFragSpecs: set the specifications of the current fragment 
// under process;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// sfragndx, index defining fragment length;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SetCurrentFragSpecs(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragndx,
    float* __restrict__ wrkmemaux)
{
    //index of the reference structure:
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor

    if(ndbCstrs <= dbstrndx)
        //no sync below: exit
        return;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx] = sfragndx;
}

// -------------------------------------------------------------------------
// SetConvergenceForUnmatchedTypes: set convergence flag for chain pairs of
// inconsistent molecule types;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (configurations) for each pair;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SetConvergenceForUnmatchedTypes(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    //blockDim.x == CUS1_TBSP_SCORE_SET_XDIM
    const uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number

    if(ndbCstrs <= dbstrndx) return;//exit unless sync below!

    uint mloc0 = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;

    int qrydst = GetQueryDst(qryndx);
    int qrytpf = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(qrydst);
    int dbstrdst = GetDbStrDst(dbstrndx);
    int rfntpf = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst);

    if(GetMoleculeType(qrytpf) != GetMoleculeType(rfntpf)) {
        wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx] =
            (float)(CONVERGED_LOWTMSC_bitval);
    }
}

// -------------------------------------------------------------------------
// SetConvergenceForUnmatchedTypesComplex: set convergence flag for complex
// pairs of inconsistent by type;
// ndbCcpxs, total number of reference complexes in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (configurations) for each pair;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SetConvergenceForUnmatchedTypesComplex(
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    //blockDim.x == CUS1_TBSP_SCORE_SET_XDIM
    const uint dbcpxndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number

    if(ndbCcpxs <= dbcpxndx) return;//exit unless sync below!

    uint mloc0 = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;

    int qrylen, qryaddr;
    int rfnlen, rfnaddr;
    GetQueryComplexAddressLength(qryndx, qryaddr, qrylen);
    int qrytype = GetQueryStrField<INTYPE,pcx2DType>(qryndx);

    if(GetComplexTypeLength(qrytype) < qrylen) return;

    GetDbComplexAddressLength(dbcpxndx, rfnaddr, rfnlen);
    int rfntype = GetDbStrField<INTYPE,pcx2DType>(dbcpxndx);

    if(GetComplexTypeLength(rfntype) < rfnlen) return;

    if(GetMoleculeType(qrytype) != GetMoleculeType(rfntype)) {
        wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbcpxndx] =
            (float)(CONVERGED_LOWTMSC_bitval);
    }
}

// -------------------------------------------------------------------------
// SetLowScoreConvergenceFlag: set the appropriate convergence flag for 
// the pairs for which the score is below the threshold;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
template<bool COMPLEX>
__global__ void SetLowScoreConvergenceFlag(
    const float scorethld,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns)
{
    //index of the reference structure:
    //blockDim.x == CUS1_TBSP_SCORE_SET_XDIM
    const uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint sfragfct = 0;//fragment factor
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;

    //exit without sync as long as no sync below:
    if(ndbCstrs <= dbstrndx) return;

    uint mloc0 = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCtrgs;

    int convflag = wrkmemaux[mloc0 + tawmvConverged * ndbCtrgs + dbstrndx];//float->int

    //exit without sync as long as no sync below:
    if(convflag & CONVERGED_LOWTMSC_bitval) return;

    //grand scores fragment factor position 0:
    float grand = wrkmemaux[mloc0 + tawmvGrandBest * ndbCtrgs + dbstrndx];

    int qrylen, dbstrlen;

    if(COMPLEX) {
        int qaddr, raddr;
        GetQueryComplexAddressLength(qryndx, qaddr, qrylen);
        GetDbComplexAddressLength(dbstrndx, raddr, dbstrlen);
    } else {
        qrylen = GetQueryLength(qryndx);
        dbstrlen = GetDbStrLength(dbstrndx);
    }

    //check for low scores:
    if(grand < scorethld * (float)myhdmin(qrylen, dbstrlen))
        wrkmemaux[mloc0 + tawmvConverged * ndbCtrgs + dbstrndx] =
            (float)(convflag | CONVERGED_LOWTMSC_bitval);
}

// Instantiations:
// 
#define INSTANTIATE_SetLowScoreConvergenceFlag(tpCOMPLEX) \
    template __global__ void SetLowScoreConvergenceFlag<tpCOMPLEX>( \
        const float scorethld, const uint ndbCstrs, const uint maxnsteps, \
        float* __restrict__ wrkmemaux, const uint ndbCchns);

INSTANTIATE_SetLowScoreConvergenceFlag(false);
INSTANTIATE_SetLowScoreConvergenceFlag(true);

// -------------------------------------------------------------------------
// SetLowScoreConvergenceFlagComplexInitial: set the convergence flag for 
// complex pairs for which the initial score originating from individual
// chain processing is below the threshold;
// MAX_NCHAINS, max #complex chains for this kernel to proceed;
// scorethld, score threshold;
// nqrycpxs, total number of query complexes in the chunk;
// ndbCcpxs, total number of reference complexes in the chunk;
// maxnsteps, max number of steps/configurations performed;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// tmpdpdiagbuffers, memory section for temporary data;
// ndbCchns, total number of reference chains in the chunk;
// 
template<int MAX_NCHAINS>
__global__ void SetLowScoreConvergenceFlagComplexInitial(
    const float scorethld,
    const uint nqrycpxs,
    const uint ndbCcpxs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    const uint ndbCchns)
{
    enum {
        lXDIM = CUS1_CHOR_CONV_SCORE_XDIM
    };

    __shared__ int cnvCache[lXDIM + 1];//cache of convergence flags

    //first, check and write convergence flags in the same logical memory space;
    //second, rewrite convergence flags for the following spatial indexing stage!
    //this avoids overwriting memory regions that are read first;
    for(int qryndx = 0; qryndx < nqrycpxs; qryndx++)
    {
        int qrycpxN;

        __syncthreads();

        if(MAX_NCHAINS > 0 && threadIdx.x == 0)
            cnvCache[lXDIM] = GetQueryStrField<INTYPE,pcx2DN>(qryndx);//#chains

        __syncthreads();

        qrycpxN = cnvCache[lXDIM];

        uint mloc0 = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCchns;
        uint mtmpt = (qryndx * nTAuxWorkingMemoryVars + tawmvGrandBest) * ndbCchns;

        for(int dbcpxndx = threadIdx.x; dbcpxndx < ndbCcpxs; dbcpxndx += blockDim.x)
        {
            float grand = atomicExch(&wrkmemaux[mloc0 + tawmvGrandBest * ndbCchns + dbcpxndx], 0.0f);

            if(MAX_NCHAINS > 0 && MAX_NCHAINS < qrycpxN) {
                int dbcpxN = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);//#chains
                if(MAX_NCHAINS < dbcpxN) grand = -1.0f;
            }

            if(0.0f < grand) {
                int qrylen;
                int dbcpxlen;
                int qaddr, raddr;
                GetQueryComplexAddressLength(qryndx, qaddr, qrylen);
                GetDbComplexAddressLength(dbcpxndx, raddr, dbcpxlen);
                //check for low scores:
                if(grand < scorethld * (float)myhdmin(qrylen, dbcpxlen)) grand = 0.0f;
            }

            tmpdpdiagbuffers[mtmpt + dbcpxndx] = grand;
        }
    }

    __syncthreads();

    //2nd phase: copy convergence flags to appropriate regions:
    for(int qryndx = 0; qryndx < nqrycpxs; qryndx++)
    {
        uint mtmpt = (qryndx * nTAuxWorkingMemoryVars + tawmvGrandBest) * ndbCchns;
        uint mloct = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCcpxs;

        for(int dbcpxndx = threadIdx.x; dbcpxndx < ndbCcpxs; dbcpxndx += blockDim.x)
        {
            float grand = tmpdpdiagbuffers[mtmpt + dbcpxndx];

            wrkmemaux[mloct + tawmvGrandBest * ndbCcpxs + dbcpxndx] = myhdmax(0.0f, grand);
            wrkmemaux[mloct + tawmvConverged * ndbCcpxs + dbcpxndx] = (float)(grand? 0: CONVERGED_LOWTMSC_bitval);
        }
    }
}

// Instantiations:
// 
#define INSTANTIATE_SetLowScoreConvergenceFlagComplexInitial(tpMAX_NCHAINS) \
    template __global__ void SetLowScoreConvergenceFlagComplexInitial<tpMAX_NCHAINS>( \
        const float scorethld, const uint nqrycpxs, const uint ndbCcpxs, const uint maxnsteps, \
        float* __restrict__ wrkmemaux, float* __restrict__ tmpdpdiagbuffers, const uint ndbCchns);

INSTANTIATE_SetLowScoreConvergenceFlagComplexInitial(CUS1_TBSP_CPXSCORE_MAX_NCHAINS_FILTER);

// -------------------------------------------------------------------------
// SetLowScoreConvergenceFlagComplexIntermediate: set the convergence 
// flag for complex pairs for which intermediate provisional score is 
// below the threshold;
// MIN_NCHAINS, lower bound of #complex chains above which this kernel takes effect;
// ndbCcpxs, total number of reference complexes in the chunk;
// ndbCstrs, total number of reference chains in the chunk;
// maxnsteps, max number of steps/configurations performed;
// nconfigs, number of configurations used (for wrkmem2);
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// wrkmem2, working memory of chain scores and assignments;
// 
template<int MIN_NCHAINS>
__global__ void SetLowScoreConvergenceFlagComplexIntermediate(
    const float scorethld,
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint nconfigs,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2)
{
    enum {
        lLARGE = 99999,
        lYDIM = CUS1_TBSP_DPSCORE_MAX_YDIM,
        lXDIM = CUS1_TBSP_DPSCORE_MAX_XDIM
    };
    //index of complex structure:
    //blockDim.x == CUS1_TBSP_SCORE_SET_XDIM
    const uint dbcpxndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query complex serial number
    __shared__ float scvCache[lYDIM][lXDIM+1];
    __shared__ int chnCache[lXDIM+1];

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    chnCache[threadIdx.x] = 0;

    if(MIN_NCHAINS > 0 && threadIdx.y == 0 && threadIdx.x == 0)
        chnCache[lXDIM] = GetQueryStrField<INTYPE,pcx2DN>(qryndx);//#chains

    if(MIN_NCHAINS > 0) __syncthreads();

    if(MIN_NCHAINS > 0 && chnCache[lXDIM] <= MIN_NCHAINS) return;

    if(MIN_NCHAINS > 0 && threadIdx.y == 0 && dbcpxndx < ndbCcpxs) {
        int dbcpxN = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);//#chains
        if(dbcpxN <= MIN_NCHAINS) chnCache[threadIdx.x] = lLARGE;
    }

    __syncthreads();

    for(uint sfragfct = threadIdx.y; sfragfct < nconfigs && sfragfct < maxnsteps; sfragfct += blockDim.y)
    {
        const uint mloc2 = ((qryndx * nconfigs + sfragfct) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
        float dpscore = (float)chnCache[threadIdx.x];
        if(dbcpxndx < ndbCcpxs && ! dpscore)
            dpscore = wrkmem2[mloc2 + twm2aCpxScore * ndbCstrs + dbcpxndx];
        if(scvCache[threadIdx.y][threadIdx.x] < dpscore)
            scvCache[threadIdx.y][threadIdx.x] = dpscore;
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (blockDim.y>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y + ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y + ydim][threadIdx.x];
        }
        __syncthreads();
    }

    //scvCache[0][...] now contain maximums

    if(threadIdx.y == 0 && dbcpxndx < ndbCcpxs) {
        int qrylen, dbcpxlen, qaddr, raddr;
        float dpscore = scvCache[0][threadIdx.x];
        GetQueryComplexAddressLength(qryndx, qaddr, qrylen);
        GetDbComplexAddressLength(dbcpxndx, raddr, dbcpxlen);
        //check for low scores:
        chnCache[threadIdx.x] = (dpscore < scorethld * (float)myhdmin(qrylen, dbcpxlen));
    }

    __syncthreads();

    for(uint sfragfct = threadIdx.y; sfragfct < nconfigs && sfragfct < maxnsteps; sfragfct += blockDim.y)
    {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbcpxndx < ndbCcpxs && chnCache[threadIdx.x])
            wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbcpxndx] = (float)(CONVERGED_LOWTMSC_bitval);
    }
}

// Instantiations:
// 
#define INSTANTIATE_SetLowScoreConvergenceFlagComplexIntermediate(tpMIN_NCHAINS) \
    template __global__ void SetLowScoreConvergenceFlagComplexIntermediate<tpMIN_NCHAINS>( \
        const float scorethld, const uint ndbCcpxs, const uint ndbCstrs, const uint maxnsteps, \
        const uint nconfigs, float* __restrict__ wrkmemaux, float* __restrict__ wrkmem2);

INSTANTIATE_SetLowScoreConvergenceFlagComplexIntermediate(0);
INSTANTIATE_SetLowScoreConvergenceFlagComplexIntermediate(CUS1_TBSP_CPXSCORE_MAX_NCHAINS_FILTER);



// -------------------------------------------------------------------------
// InitWrkmem2: initialize the slot of the wrkmem2 memory used
// temporarily for convergence;
// ndbCstrs, total number of reference structures in the chunk;
// maxnstepsmem2, max number of steps for assignments in wrkmem2;
// wrkmem2, working memory for chain scores and assignments;
// value, value to populate;
// 
__global__
void InitWrkmem2(
    const uint ndbCstrs,
    const uint maxnstepsmem2,
    float* __restrict__ wrkmem2,
    const float value)
{
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;//structure index
    const uint qryndx = blockIdx.y;//query serial number
    const uint mloc2 = ((qryndx * maxnstepsmem2 + 0) * nTWorkingMemory2VarsAssignment) * ndbCstrs;

    if(dbstrndx < ndbCstrs)
        wrkmem2[mloc2 + twm2aTmpConvergence * ndbCstrs + dbstrndx] = value;
}

// -------------------------------------------------------------------------
// InitScores: initialize best and current scores to 0;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// minfraglen, minimum fragment length for which maxnsteps is calculated;
// checkfragos, check whether calculated fragment position is within boundaries;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
template<int INITOPT>
__global__ void InitScores(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint /* minfraglen */,
    const bool checkfragos,
    float* __restrict__ wrkmemaux)
{
    //index of the reference structure:
    //blockDim.x == CUS1_TBSP_SCORE_SET_XDIM
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor


//     //{{TODO: to be removed as two reads per pair is more expensive than one write
//     if(checkfragos) {
//         uint sfragpos = sfragfct * FRAGREF_SFRAGSTEP;//fragment position
//         int dbstrlen = 0;
//         __shared__ int qrylen;
// 
//         qrylen = 0;
// 
//         if(dbstrndx < ndbCstrs) {
//             dbstrlen = GetDbStrLength(dbstrndx);
//             if(threadIdx.x == 0) qrylen = GetQueryLength(qryndx);
//         }
// 
//         __syncthreads();
// 
//         if(dbstrndx < ndbCstrs) {
//             uint maxalnlen = myhdmin(dbstrlen, qrylen);
//             dbstrlen = FragPosWithinAlnBoundaries(maxalnlen, FRAGREF_SFRAGSTEP, sfragpos, minfraglen);
//         }
// 
//         //__syncthreads();//no sync as each thread accesses own data
// 
//         if(dbstrlen == 0) return;//no sync below: exit
//     }
//     //}}TODO


    if(ndbCstrs <= dbstrndx)
        //no sync below: exit
        return;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(INITOPT & INITOPT_ALL) {
        #pragma unroll
        for(int f = 0; f < nTAuxWorkingMemoryVars; f++)
            wrkmemaux[mloc + f * ndbCstrs + dbstrndx] = 0.0f;
    }

    //NOTE: one read and write are more efficient than two reads plus additional 
    // calculation (for bounds) and these memory instructions (done when checking 
    // whether a pair is out of bounds)
    if(INITOPT & INITOPT_BEST)
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = 0.0f;

    if(INITOPT & INITOPT_CURRENT) 
        wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx] = 0.0f;


    if(INITOPT & INITOPT_QRYRFNPOS) {
        wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx] = 0.0f;
        wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx] = 0.0f;
    }


    if(INITOPT & INITOPT_FRAGSPECS) {
        wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx] = 0.0f;
        wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + dbstrndx] = 0.0f;
    }


    if(INITOPT & INITOPT_NALNPOSS) 
        wrkmemaux[mloc + tawmvNAlnPoss * ndbCstrs + dbstrndx] = 0.0f;


    if(INITOPT & INITOPT_CONVFLAG_ALL) 
        wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx] = 0.0f;


    if(INITOPT &
        (INITOPT_CONVFLAG_FRAGREF | INITOPT_CONVFLAG_SCOREDP |
        INITOPT_CONVFLAG_NOTMPRG | INITOPT_CONVFLAG_LOWTMSC |
        INITOPT_CONVFLAG_LOWTMSC_SET))
    {
        mloc += tawmvConverged * ndbCstrs + dbstrndx;
        int convflag = wrkmemaux[mloc];//float->int

        //NOTE: do not set any value if the process for this pair is to terminate
        if(convflag & CONVERGED_LOWTMSC_bitval) return;

        if(INITOPT & INITOPT_CONVFLAG_FRAGREF)
            if(convflag & CONVERGED_FRAGREF_bitval)
                convflag = convflag & (~CONVERGED_FRAGREF_bitval);

        if(INITOPT & INITOPT_CONVFLAG_SCOREDP)
            if(convflag & CONVERGED_SCOREDP_bitval)
                convflag = convflag & (~CONVERGED_SCOREDP_bitval);

        if(INITOPT & INITOPT_CONVFLAG_NOTMPRG)
            if(convflag & CONVERGED_NOTMPRG_bitval)
                convflag = convflag & (~CONVERGED_NOTMPRG_bitval);

        if(INITOPT & INITOPT_CONVFLAG_LOWTMSC)
            if(convflag & CONVERGED_LOWTMSC_bitval)
                convflag = convflag & (~CONVERGED_LOWTMSC_bitval);

        if(INITOPT & INITOPT_CONVFLAG_LOWTMSC_SET)
            convflag = convflag | (CONVERGED_LOWTMSC_bitval);

        wrkmemaux[mloc] = (float)convflag;//int->float
    }
}

// Instantiations:
// 
#define INSTANTIATE_InitScores(tpINITOPT) \
    template __global__ void InitScores<tpINITOPT>( \
        const uint ndbCstrs, \
        const uint maxnsteps, const uint minfraglen, \
        const bool checkfragos, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_InitScores(INITOPT_ALL);
INSTANTIATE_InitScores(INITOPT_BEST);
INSTANTIATE_InitScores(INITOPT_CURRENT);
INSTANTIATE_InitScores(INITOPT_QRYRFNPOS);
INSTANTIATE_InitScores(INITOPT_FRAGSPECS);
INSTANTIATE_InitScores(INITOPT_NALNPOSS);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_ALL);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_FRAGREF);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_SCOREDP);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_NOTMPRG);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_LOWTMSC);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_LOWTMSC_SET|INITOPT_ALL);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_FRAGREF|INITOPT_CONVFLAG_SCOREDP);
INSTANTIATE_InitScores(INITOPT_NALNPOSS|INITOPT_CONVFLAG_SCOREDP);
INSTANTIATE_InitScores(INITOPT_BEST|INITOPT_CONVFLAG_ALL);
INSTANTIATE_InitScores(INITOPT_BEST|INITOPT_CURRENT|INITOPT_QRYRFNPOS|INITOPT_FRAGSPECS|INITOPT_NALNPOSS);

// -------------------------------------------------------------------------
// SaveLastScore: save the last best score at the position corresponding to 
// fragment factor 0;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SaveLastScore0(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns)
{
    //index of the structure to start with (blockIdx.x, refn. serial number):
    const uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;

    if(ndbCstrs <= dbstrndx)
        //no sync below: exit
        return;

    uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCtrgs;

    int converged = wrkmemaux[mloc0 + tawmvConverged * ndbCtrgs + dbstrndx];//float->int

    //score convergence applies:
    if(converged & (CONVERGED_SCOREDP_bitval | CONVERGED_LOWTMSC_bitval)) return;

    wrkmemaux[mloc0 + tawmvBest0 * ndbCtrgs + dbstrndx] =
        wrkmemaux[mloc0 + tawmvBestScore * ndbCtrgs + dbstrndx];
}

// -------------------------------------------------------------------------
// SaveBestScore: save best score along with query and reference 
// positions for which this score is observed;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// n1, starting position that determines positions in query and reference;
// step, step size in positions used to traverse query and reference 
// ungapped alignments;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SaveBestScore(
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    float* __restrict__ wrkmemaux)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    n1 += sfragfct * step;
    int qrypos = myhdmax(0,n1);//starting query position
    int rfnpos = myhdmax(-n1,0);//starting reference position

    //no sync below, threads process independently: exit
    if(ndbCstrs <= dbstrndx) return;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    float bestscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
    float score = wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx];

    //NOTE: two reads are more efficient than two structure length reads plus 
    // additional calculation (for bounds) (done when checking whether a 
    // pair is out of bounds);
    // the if clause will not be true for an out-of-bounds case
    if(bestscore < score) {
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = score;
        wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
        wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
    }
}

// -------------------------------------------------------------------------
// SaveBestScoreAmongBests: save best score along with query and reference 
// positions by considering all partial best scores calculated over all 
// fragment factors; write it to the location of fragment factor 0;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// effnsteps, effective (actual maximum) number of steps (blockIdx.z);
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SaveBestScoreAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    float* __restrict__ wrkmemaux)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < effnsteps; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
        if(scvCache[threadIdx.y][threadIdx.x] < bscore) {
            scvCache[threadIdx.y][threadIdx.x] = bscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUS1_TBSP_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    if(threadIdx.y == 0) {
        uint sfragfct = ndxCache[0][threadIdx.x];
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        float bscore = scvCache[0][threadIdx.x];
        if(sfragfct != 0 && dbstrndx < ndbCstrs) {
            float qrypos = wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx];
            float rfnpos = wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx];
            //coalesced WRITE for multiple references
            wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs + dbstrndx] = bscore;
            wrkmemaux[mloc0 + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
            wrkmemaux[mloc0 + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
        }
        if(dbstrndx < ndbCstrs)
            wrkmemaux[mloc0 + tawmvInitialBest * ndbCstrs + dbstrndx] = bscore;
    }
}

// -------------------------------------------------------------------------
// CheckScoreConvergence: check whether the score of the last two 
// procedures converged, i.e., the difference is small;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void CheckScoreConvergence(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns)
{
    //index of the structure to start with (blockIdx.x, refn. serial number):
    const uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;
    const int sfragfct = blockIdx.z;//fragment factor

    if(ndbCstrs <= dbstrndx)
        //no sync below, threads process independently: exit
        return;

    uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCtrgs;
    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCtrgs;

    int converged = wrkmemaux[mloc + tawmvConverged * ndbCtrgs + dbstrndx];//float->int

    //score convergence applies:
    if(converged & (CONVERGED_SCOREDP_bitval | CONVERGED_LOWTMSC_bitval)) return;

    //best scores are recorded at a fragment factor position of 0
    float prevbest0 = wrkmemaux[mloc0 + tawmvBest0 * ndbCtrgs + dbstrndx];
    float best0 = wrkmemaux[mloc0 + tawmvBestScore * ndbCtrgs + dbstrndx];

    //check score convergence; populate convergence flag over all fragment factors
    if(fabsf(best0-prevbest0) < SCORE_CONVEPSILON)
        wrkmemaux[mloc + tawmvConverged * ndbCtrgs + dbstrndx] =
            (float)(converged | CONVERGED_SCOREDP_bitval);
}

#if 0
// -------------------------------------------------------------------------
// CheckScoreProgression: check whether the difference between the maximum 
// score and the score of the last procedure is large enough; if not, set 
// the appropriate convergence flag;
// ndbCstrs, total number of reference structures in the chunk;
// maxscorefct, factor for the maximum score;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void CheckScoreProgression(
    uint ndbCstrs,
    float maxscorefct,
    float* __restrict__ wrkmemaux)
{
    //index of the structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number

    if(ndbCstrs <= dbstrndx)
        //no sync below, threads process independently: exit
        return;

    int converged = wrkmemaux[
        (qryndx * nTAuxWorkingMemoryVars + tawmvConverged) *
        ndbCstrs + dbstrndx];//float->int

    //score convergence applies:
    if(converged & (CONVERGED_NOTMPRG_bitval | CONVERGED_LOWTMSC_bitval)) return;

    float grand = wrkmemaux[
        (qryndx * nTAuxWorkingMemoryVars + tawmvGrandBest) *
        ndbCstrs + dbstrndx];
    float best = wrkmemaux[
        (qryndx * nTAuxWorkingMemoryVars + tawmvBestScore) *
        ndbCstrs + dbstrndx];

    //check score progression
    if(best <= grand * maxscorefct)
        wrkmemaux[
            (qryndx * nTAuxWorkingMemoryVars + tawmvConverged) *
            ndbCstrs + dbstrndx] =
                (float)(converged | CONVERGED_NOTMPRG_bitval);
}
#endif

// -------------------------------------------------------------------------
// SaveBestScoreAndTM: save best scores along with transformation matrices;
// save fragment indices and starting positions too;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// sfragstep, step size to traverse subfragments;
// sfragndx, index defining fragment length;
// sfragpos, starting position within fragment;
// NOTE: memory pointers should be aligned!
// wrkmemtm, working memory for transformation matrices;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// tfmmem, memory for transformation matrices;
// wrkmemaux, auxiliary working memory;
// NOTE: unroll by a factor of CUS1_TBINITSP_TMSAVE_XFCT: this number of 
// structures processed by a thread block
// 
template<bool WRITEFRAGINFO>
__global__ void SaveBestScoreAndTM(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * CUS1_TBINITSP_TMSAVE_XFCT;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    uint ndx = 0;//relative reference index < CUS1_TBINITSP_TMSAVE_XFCT
    __shared__ float scvCache[CUS1_TBINITSP_TMSAVE_XFCT];


    if(threadIdx.x < CUS1_TBINITSP_TMSAVE_XFCT)
        scvCache[threadIdx.x] = 0.0f;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(threadIdx.x < CUS1_TBINITSP_TMSAVE_XFCT && dbstrndx + threadIdx.x < ndbCstrs) {
        scvCache[threadIdx.x] = wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx + threadIdx.x];
        float best = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx + threadIdx.x];
        if(scvCache[threadIdx.x] <= best) scvCache[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for(int i = 1; i < CUS1_TBINITSP_TMSAVE_XFCT; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //save scores first
    if(threadIdx.x < CUS1_TBINITSP_TMSAVE_XFCT && scvCache[threadIdx.x]) {
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx + threadIdx.x] = scvCache[threadIdx.x];

        if(WRITEFRAGINFO) {
            //NOTE: using this function with WRITEFRAGINFO==true will not be correct and may 
            //lead to inconsistent results in the final stage of the best alignment refinement!
            // wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx + threadIdx.x] =
            //     wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx + threadIdx.x];

            wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + dbstrndx + threadIdx.x] =
                sfragfct * sfragstep;
        }
    }

    //save transformation matrices next
    if(threadIdx.x < nTTranformMatrix * (ndx+1) && scvCache[ndx]) {
        mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        float value = wrkmemtm[mloc];
        wrkmemtmibest[mloc] = value;
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
#define INSTANTIATE_SaveBestScoreAndTM(tpWRITEFRAGINFO) \
    template __global__ void SaveBestScoreAndTM<tpWRITEFRAGINFO>( \
        const uint ndbCstrs, const uint maxnsteps, const int sfragstep, \
        const float* __restrict__ wrkmemtm, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_SaveBestScoreAndTM(false);
INSTANTIATE_SaveBestScoreAndTM(true);

// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// SaveBestScoreAndTMAmongBests: save best scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the 
// location of fragment factor 0;
// WRITEFRAGINFO, write fragment information if the best score is obtained;
// GRANDUPDATE, >=0: update the grand best score if the best score is obtained;
// FORCEWRITEFRAGINFO, force writing frag info for the best score obtained
// among the bests;
// SECONDARYUPDATE, indication of whether and how secondary update of best scores is done;
// GRANDSCOREMEMSECTION, memory section for grand scores;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// tfmmem, memory for transformation matrices;
// wrkmemaux, auxiliary working memory;
// wrkmemtmibest2nd, secondary working memory for iteration-best transformation 
// matrices (only slot 0 is used but indexing involves maxnsteps);
// 
template<
    bool WRITEFRAGINFO,
    int GRANDUPDATE,
    bool FORCEWRITEFRAGINFO,
    int SECONDARYUPDATE>
__global__
void SaveBestScoreAndTMAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ tfmmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemtmibest2nd,
    const uint ndbCchns)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < /*maxnsteps*/effnsteps; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCtrgs + dbstrndx];
        if(scvCache[threadIdx.y][threadIdx.x] < bscore) {
            scvCache[threadIdx.y][threadIdx.x] = bscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUS1_TBSP_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    uint sfragfct = ndxCache[0][threadIdx.x];
    bool wrtgrand = 0;
    bool wrt2ndry = (SECONDARYUPDATE == SECONDARYUPDATE_UNCONDITIONAL);

    //write scores first
    if(threadIdx.y == 0) {
        ndxCache[1][threadIdx.x] = 0;
        ndxCache[2][threadIdx.x] = wrt2ndry;
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        if(sfragfct != 0 && dbstrndx < ndbCstrs) {
            float bscore = scvCache[0][threadIdx.x];
            //coalesced WRITE for multiple references
            wrkmemaux[mloc0 + tawmvBestScore * ndbCtrgs + dbstrndx] = bscore;
        }
        //update the grand best scores
        if(dbstrndx < ndbCstrs) {
            float bscore2nd;
            float bscore = scvCache[0][threadIdx.x];
            float grand = 0.0f;
            if(GRANDUPDATE >= 0)
                grand = wrkmemaux[mloc0 + GRANDUPDATE * ndbCtrgs + dbstrndx];
            {
                uint mloc20 = ((qryndx * maxnsteps + 0) * ndbCtrgs + ndbCtrgs) * nTTranformMatrix;
                if(SECONDARYUPDATE == SECONDARYUPDATE_CONDITIONAL) {
                    //NOTE: 2nd'ry scores written immediately following tfms
                    bscore2nd = wrkmemtmibest2nd[mloc20 + dbstrndx];
                    ndxCache[2][threadIdx.x] = wrt2ndry = (bscore2nd < bscore);//reuse cache
                }
                if(wrt2ndry) wrkmemtmibest2nd[mloc20 + dbstrndx] = bscore;
            }
            if(GRANDUPDATE >= 0)
                ndxCache[1][threadIdx.x] = wrtgrand = (grand < bscore);//reuse cache
            //coalesced WRITE for multiple references
            if(wrtgrand)
                wrkmemaux[mloc0 + GRANDUPDATE * ndbCtrgs + dbstrndx] = bscore;
            if(WRITEFRAGINFO && (FORCEWRITEFRAGINFO || wrtgrand)) {
                float frgndx = wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCtrgs + dbstrndx];
                float frgpos = wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCtrgs + dbstrndx];
                wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCtrgs + dbstrndx] = frgndx;
                wrkmemaux[mloc0 + tawmvSubFragPos * ndbCtrgs + dbstrndx] = frgpos;
            }
        }
    }

    __syncthreads();

    //NOTE: change indexing so that threadIdx.y refers to a different reference
    sfragfct = ndxCache[0][threadIdx.y];
    wrtgrand = ndxCache[1][threadIdx.y];
    wrt2ndry = ndxCache[2][threadIdx.y];

    __syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    //READ and WRITE iteration-best transformation matrices
    if(threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * ndbCtrgs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCtrgs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        float value = 0.0f;
        if(sfragfct != 0 || wrtgrand || wrt2ndry) value = wrkmemtmibest[mloc];//READ from gmem
        if(sfragfct != 0) wrkmemtmibest[mloc0] = value;//WRITE to gmem
        //save this transformation matrix for tfmmem below
        scvCache[threadIdx.y][threadIdx.x] = value;
    }

    __syncthreads();

    //WRITE the transformation matrix with the currently grand best score
    if(wrtgrand && threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint tfmloc = (qryndx * ndbCtrgs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        tfmmem[tfmloc] = scvCache[threadIdx.y][threadIdx.x];
    }

    //WRITE tfm corresponding to the best-performing over all wrkmemtmibest of multiple passes
    if(wrt2ndry && threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * ndbCtrgs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        wrkmemtmibest2nd[mloc0] = scvCache[threadIdx.y][threadIdx.x];
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
#define INSTANTIATE_SaveBestScoreAndTMAmongBests( \
    tpWRITEFRAGINFO,tpGRANDUPDATE,tpFORCEWRITEFRAGINFO,tpSECONDARYUPDATE) \
    template __global__ void SaveBestScoreAndTMAmongBests \
        <tpWRITEFRAGINFO,tpGRANDUPDATE,tpFORCEWRITEFRAGINFO,tpSECONDARYUPDATE>( \
            const uint ndbCstrs, const uint maxnsteps, const uint effnsteps, \
            float* __restrict__ wrkmemtmibest, \
            float* __restrict__ tfmmem, \
            float* __restrict__ wrkmemaux, \
            float* __restrict__ wrkmemtmibest2nd, \
            const uint ndbCchns);

INSTANTIATE_SaveBestScoreAndTMAmongBests(false,tawmvGrandBest,false,SECONDARYUPDATE_NOUPDATE);
INSTANTIATE_SaveBestScoreAndTMAmongBests(false,tawmvGrandBest,false,SECONDARYUPDATE_UNCONDITIONAL);
INSTANTIATE_SaveBestScoreAndTMAmongBests(false,tawmvGrandBest,false,SECONDARYUPDATE_CONDITIONAL);
INSTANTIATE_SaveBestScoreAndTMAmongBests(true,tawmvGrandBest,false,SECONDARYUPDATE_NOUPDATE);
INSTANTIATE_SaveBestScoreAndTMAmongBests(true,tawmvGrandBest,true,SECONDARYUPDATE_NOUPDATE);
//
INSTANTIATE_SaveBestScoreAndTMAmongBests(false,tawmvScore,false,SECONDARYUPDATE_NOUPDATE);

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// ProductionSaveBestScoresAndTMAmongBests: save best scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; production version;
// WRITEFRAGINFO, template parameter, whether to save a fragment length 
// index and position for the best score;
// CONDITIONAL, template parameter, flag of whether the grand best score is
// compared with the current best before writing;
// COMPLEX, template parameter, flag of processing complexes;
// NOTE: CPXTYPENUMBERSCT, memory step number section of alignment-determined complex types;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemaux, auxiliary working memory;
// alndatamem, memory for full alignment information, including scores;
// tfmmem, memory for transformation matrices;
// 
template<bool WRITEFRAGINFO, bool CONDITIONAL, bool COMPLEX, int CPXTYPENUMBERSCT>
__global__
void ProductionSaveBestScoresAndTMAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem,
    const uint ndbCchns)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;
    enum {
        //array index for query
        QRYX = CUS1_TBSP_SCORE_MAX_XDIM,
        //#alignment data entries to write
        nADT = (nTDP2OutputAlnDataPart2 - dp2oadScoreQ)
    };
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ float adtCache[CUS1_TBSP_SCORE_MAX_XDIM][nADT];
    __shared__ int lenCache[CUS1_TBSP_SCORE_MAX_XDIM+1];//lengths
    __shared__ int ctpCache[CUS1_TBSP_SCORE_MAX_XDIM+1];//types

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    if(threadIdx.y < nADT && dbstrndx < ndbCstrs)
        adtCache[threadIdx.x][threadIdx.y] = 0.0f;

    if(threadIdx.y == 1 && threadIdx.x == 0) {
        int len, addr;
        if(COMPLEX) GetQueryComplexAddressLength(qryndx, addr, len);
        else len = GetQueryLength(qryndx);
        lenCache[QRYX] = len;
    }

    if(threadIdx.y == 0 && dbstrndx < ndbCstrs) {
        int len, addr;
        if(COMPLEX) GetDbComplexAddressLength(dbstrndx, addr, len);
        else len = GetDbStrLength(dbstrndx);
        lenCache[threadIdx.x] = len;
    }

    if(threadIdx.y == 2 && dbstrndx < ndbCstrs) {
        int type;
        uint mloct = ((qryndx * maxnsteps + CPXTYPENUMBERSCT) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        if(COMPLEX) type = wrkmemaux[mloct + tawmvNAlnPoss * ndbCtrgs + dbstrndx];
        else {
            uint dbstrdst = GetDbStrDst(dbstrndx);
            type = GetMoleculeType(GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst));
        }
        ctpCache[threadIdx.x] = type;
    }

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < /* maxnsteps */effnsteps; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCtrgs + dbstrndx];
        if(scvCache[threadIdx.y][threadIdx.x] < bscore) {
            scvCache[threadIdx.y][threadIdx.x] = bscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUS1_TBSP_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    uint sfragfct = ndxCache[0][threadIdx.x];
    bool wrtgrand = 0;

    //write scores first
    if(threadIdx.y == 0) {
        ndxCache[1][threadIdx.x] = 0;
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        //update the grand best scores
        if(dbstrndx < ndbCstrs) {
            float bscore = scvCache[0][threadIdx.x];
            float grand = 0.0f;
            if(CONDITIONAL)
                grand = wrkmemaux[mloc0 + tawmvGrandBest * ndbCtrgs + dbstrndx];
            ndxCache[1][threadIdx.x] = wrtgrand = (grand < bscore);//reuse cache
            //coalesced WRITE for multiple references
            if(wrtgrand) {
                wrkmemaux[mloc0 + tawmvGrandBest * ndbCtrgs + dbstrndx] = bscore;
                //score calculated for the longer structure:
                float gbest = wrkmemaux[mloc + tawmvBest0 * ndbCtrgs + dbstrndx];
                const float d0Q = GetD0fin(lenCache[QRYX], lenCache[QRYX], ctpCache[threadIdx.x]);//threshold for query
                const float d0R = GetD0fin(lenCache[threadIdx.x], lenCache[threadIdx.x], ctpCache[threadIdx.x]);//for reference
                //make bscore (should not be used below associated clauses) represent the query score:
                if(lenCache[threadIdx.x] < lenCache[QRYX]) myhdswap(bscore, gbest);
                //write alignment information in cache:
                adtCache[threadIdx.x][(dp2oadScoreQ-dp2oadScoreQ)] = __fdividef(bscore, lenCache[QRYX]);
                adtCache[threadIdx.x][(dp2oadScoreR-dp2oadScoreQ)] = __fdividef(gbest, lenCache[threadIdx.x]);
                adtCache[threadIdx.x][(dp2oadD0Q-dp2oadScoreQ)] = d0Q;
                adtCache[threadIdx.x][(dp2oadD0R-dp2oadScoreQ)] = d0R;
                if(WRITEFRAGINFO) {
                    float frgndx = wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCtrgs + dbstrndx];
                    float frgpos = wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCtrgs + dbstrndx];
                    wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCtrgs + dbstrndx] = frgndx;
                    wrkmemaux[mloc0 + tawmvSubFragPos * ndbCtrgs + dbstrndx] = frgpos;
                }
            }
        }
    }

    __syncthreads();

    //NOTE: change indexing so that threadIdx.y refers to a different reference
    sfragfct = ndxCache[0][threadIdx.y];
    wrtgrand = ndxCache[1][threadIdx.y];

    __syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    //READ iteration-best transformation matrices
    if(threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        //uint mloc0 = ((qryndx * maxnsteps + 0) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCtrgs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        float value = 0.0f;
        if(wrtgrand) value = wrkmemtmibest[mloc];//READ from gmem
        //save this transformation matrix for tfmmem below
        scvCache[threadIdx.y][threadIdx.x] = value;
    }

    __syncthreads();

    //WRITE the transformation matrix with the currently grand best score
    if(wrtgrand && threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint tfmloc = (qryndx * ndbCtrgs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        tfmmem[tfmloc] = scvCache[threadIdx.y][threadIdx.x];
    }

    //WRITE the transformation matrix with the currently grand best score
    if(wrtgrand && threadIdx.x < nADT && dbstrndx < ndbCstrs) {
        uint mloc = (qryndx * ndbCtrgs + dbstrndx) * nTDP2OutputAlnData;
        alndatamem[mloc + dp2oadScoreQ + threadIdx.x] = adtCache[threadIdx.y][threadIdx.x];
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
#define INSTANTIATE_ProductionSaveBestScoresAndTMAmongBests(tpWRITEFRAGINFO,tpCONDITIONAL,tpCOMPLEX,tpCPXTYPENUMBERSCT) \
    template __global__ void ProductionSaveBestScoresAndTMAmongBests<tpWRITEFRAGINFO,tpCONDITIONAL,tpCOMPLEX,tpCPXTYPENUMBERSCT>( \
        const uint ndbCstrs, const uint maxnsteps, const uint effnsteps, \
        const float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ alndatamem, \
        float* __restrict__ tfmmem, \
        const uint ndbCchns);

INSTANTIATE_ProductionSaveBestScoresAndTMAmongBests(true,false,false,sfin_cpx_complextypenumber2);
INSTANTIATE_ProductionSaveBestScoresAndTMAmongBests(false,true,false,sfin_cpx_complextypenumber2);

INSTANTIATE_ProductionSaveBestScoresAndTMAmongBests(true,false,true,sfin_cpx_complextypenumber2);
INSTANTIATE_ProductionSaveBestScoresAndTMAmongBests(false,true,true,sfin_cpx_complextypenumber2);
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// SaveTopNScoresAndTMsAmongSecondaryBests: save secondary top N scores and 
// respective transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the first
// N locations of fragment factors;
// depth, superposition depth for calculating query and reference position factors;
// firstit, flag of the first iteration;
// twoconfs, process two configurations of secondary bests scores (with varying pace);
// rfnfragfctinit, initial fragment factor for reference to calculate query and
// reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemtm, working memory for selected transformation matrices;
// wrkmemaux, auxiliary working memory;
// 
// __launch_bounds__(1024,1)//for tests
template<bool COMPLEX>
__global__
void SaveTopNScoresAndTMsAmongSecondaryBests(
    const int lTOPN,
    const int depth,
    const bool firstit,
    const bool twoconfs,
    const int rfnfragfctinit,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns,
    const int seedapproachstruct)
{
    enum {
        //length threshold for triggering the analysis of secondary best scores:
        lLENTHR = 150,
        lYDIM = CUS1_TBSP_SCORE_MAX_YDIM,
        lXDIM = CUS1_TBSP_SCORE_MAX_XDIM,
        lQRYNDX = lXDIM,
        // lTOPN = CUS1_TBSP_DPSCORE_TOP_N,
        lMAXS = CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS
        // lTOPNxMAXS = lTOPN * lMAXS,
        // lTOPN2 = lTOPN * 2
    };
    const int lTOPNxMAXS = lTOPN * lMAXS;
    const int lTOPN2 = lTOPN * 2;
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;
    __shared__ float scvCache2[lYDIM][lXDIM+1];
    __shared__ float scvCache3[lYDIM][lXDIM+1];
    __shared__ int ndxCache2[lYDIM][lXDIM+1];
    __shared__ int ndxCache3[lYDIM][lXDIM+1];
    __shared__ int dbstrlen[lXDIM+1];

    //coalesced reads of lengths of multiple reference structures:
    if(threadIdx.y == 0 && dbstrndx < ndbCstrs) {
        int len, addr;
        if(COMPLEX) GetDbComplexAddressLength(dbstrndx, addr, len);
        else len = GetDbStrLength(dbstrndx);
        dbstrlen[threadIdx.x] = len;
    }

    if(threadIdx.y == 1 && threadIdx.x == 0) {
        int len, addr;
        if(COMPLEX) GetQueryComplexAddressLength(qryndx, addr, len);
        else len = GetQueryLength(qryndx);
        dbstrlen[lQRYNDX] = len;
    }

    scvCache2[threadIdx.x][threadIdx.y] = 0.0f;
    ndxCache2[threadIdx.x][threadIdx.y] = -1;
    // if(threadIdx.y == 0) ndxCache2[threadIdx.x][lYDIM] = 0;//counter
    if(twoconfs) {
        scvCache3[threadIdx.x][threadIdx.y] = 0.0f;
        ndxCache3[threadIdx.x][threadIdx.y] = -1;
        // if(threadIdx.y == 0) ndxCache3[threadIdx.x][lYDIM] = 0;//counter
    }

    //sync for lengths;
    __syncthreads();
 
    if(!firstit) {
        //read previously saved scores (NOTE) in wrkmemtm memory, which is assumed to be large enough!
        uint mloc = ((qryndx * maxnsteps + lTOPNxMAXS) * ndbCtrgs) * nTTranformMatrix;
        if(dbstrndx < ndbCstrs && (lLENTHR < dbstrlen[threadIdx.x] || lLENTHR < dbstrlen[lQRYNDX]))
        {
            scvCache2[threadIdx.x][threadIdx.y] =
                wrkmemtm[mloc + threadIdx.y * ndbCtrgs + dbstrndx];
            if(twoconfs)
                scvCache3[threadIdx.x][threadIdx.y] =
                    wrkmemtm[mloc + (lTOPN + threadIdx.y) * ndbCtrgs + dbstrndx];
        }
    }

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < effnsteps; sfragfct += blockDim.y)
    {
        if(dbstrlen[threadIdx.x] <= lLENTHR && dbstrlen[lQRYNDX] <= lLENTHR)
            continue;//no sync below in this block

        float bscore = 0.0f;
        int qryfragfct, rfnfragfct;
        int qrystepsz, rfnstepsz;

        if(dbstrndx < ndbCstrs)
            GetQryRfnFct_frg2(
                depth, &qryfragfct, &rfnfragfct,
                dbstrlen[lQRYNDX], dbstrlen[threadIdx.x],
                sfragfct, rfnfragfctinit,
                &qrystepsz, &rfnstepsz,
                seedapproachstruct);

        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

        bool sec2met =
            (dbstrndx < ndbCstrs) &&
            //NOTE: out-of-bounds condition check for consistency!
            (qryfragfct * qrystepsz < dbstrlen[lQRYNDX]) &&
            (rfnfragfct * rfnstepsz < dbstrlen[threadIdx.x]) &&
            ((dbstrlen[lQRYNDX] <= lLENTHR) || ((qryfragfct & 1) == 0)) && 
            ((dbstrlen[threadIdx.x] <= lLENTHR) || ((rfnfragfct & 1) == 0));

        bool sec3met =
            twoconfs && (dbstrndx < ndbCstrs) &&
            //NOTE: out-of-bounds condition check for consistency!
            (qryfragfct * qrystepsz < dbstrlen[lQRYNDX]) &&
            (rfnfragfct * rfnstepsz < dbstrlen[threadIdx.x]) &&
            ((dbstrlen[lQRYNDX] <= lLENTHR) || (myfastmod3(qryfragfct) == 0)) && 
            ((dbstrlen[threadIdx.x] <= lLENTHR) || (myfastmod3(rfnfragfct) == 0));

        if(dbstrndx < ndbCstrs && (sec2met || sec3met))//coalesced READ
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];

        if(sec2met) {
            // int locndx = atomicAdd(&ndxCache2[threadIdx.x][lYDIM], 1);
            // ndxCache2[threadIdx.x][lYDIM] = ndxCache2[threadIdx.x][lYDIM] & (lYDIM-1);
            int locndx = threadIdx.y;
            if(scvCache2[threadIdx.x][locndx] < bscore) {
                scvCache2[threadIdx.x][locndx] = bscore;
                ndxCache2[threadIdx.x][locndx] = sfragfct;
            }
        }

        if(sec3met) {
            // int locndx = atomicAdd(&ndxCache3[threadIdx.x][lYDIM], 1);
            // ndxCache3[threadIdx.x][lYDIM] = ndxCache3[threadIdx.x][lYDIM] & (lYDIM-1);
            int locndx = threadIdx.y;
            if(scvCache3[threadIdx.x][locndx] < bscore) {
                scvCache3[threadIdx.x][locndx] = bscore;
                ndxCache3[threadIdx.x][locndx] = sfragfct;
            }
        }

        //no sync, every thread works in its own space
    }

    __syncthreads();

    //NOTE: do not sort.
    // //sort scores and accompanying indices:
    // BatcherSortYDIMparallel<lYDIM,false/*descending*/>(
    //     lYDIM, scvCache2[threadIdx.x], ndxCache2[threadIdx.x]);
    // //no sync: sync'ed in BatcherSortYDIMparallel

    // if(twoconfs) {
    //     //synced inside
    //     BatcherSortYDIMparallel<lYDIM,false/*descending*/>(
    //         lYDIM, scvCache3[threadIdx.x], ndxCache3[threadIdx.x]);
    // }

    //scvCache[...][0..lTOPN-1]now contain top N scores
    //write scores first
    if(threadIdx.y < lTOPN && threadIdx.y < maxnsteps) {
        uint mloc = ((qryndx * maxnsteps + lTOPNxMAXS) * ndbCtrgs) * nTTranformMatrix;
        if(dbstrndx < ndbCstrs) {
            //int sfragfct2 = ndxCache2[threadIdx.x][threadIdx.y];//can be <0
            wrkmemtm[mloc + threadIdx.y * ndbCtrgs + dbstrndx] =
                scvCache2[threadIdx.x][threadIdx.y];
            if(twoconfs) {
                //int sfragfct3 = ndxCache3[threadIdx.x][threadIdx.y];//can be <0
                wrkmemtm[mloc + (lTOPN + threadIdx.y) * ndbCtrgs + dbstrndx] =
                    scvCache3[threadIdx.x][threadIdx.y];
            }
        }
    }

    //NOTE: no sync as long as caches are not overwritten from the last sync:
    //__syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    constexpr int nmtxs = lXDIM / nTTranformMatrix;
    int ndx = 0;//relative reference index
    for(int i = 1; i < nmtxs; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //READ and WRITE iteration-best transformation matrices;
    //rearrange lTOPN best performing mtxs at the first slots (sfragfct indices)
    for(int sx = 0; sx < lTOPN; sx += nmtxs) {
        if(threadIdx.x < nTTranformMatrix * nmtxs && 
           threadIdx.x < nTTranformMatrix * (ndx+1) && dbstrndx < ndbCstrs) {
            uint tid = threadIdx.x - nTTranformMatrix * ndx;
            uint sxx = sx + ndx;
            if(sxx < lTOPN && sxx < maxnsteps) {
                //NOTE: indexing changed so that threadIdx.y refers to a different reference
                int sfragfct = ndxCache2[threadIdx.y][sxx];
                uint mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                uint mloct = ((qryndx * maxnsteps + (lTOPN + sxx)) * ndbCtrgs + dbstrndx) * nTTranformMatrix + tid;
                if(0 <= sfragfct)
                    wrkmemtm[mloct] = wrkmemtmibest[mlocs];//GMEM READ/WRITE
                else if(firstit) {
                    //initialize otherwise; GMEM READ/WRITE
                    wrkmemtm[mloct] = (tid==tfmmRot_0_0 || tid==tfmmRot_1_1 || tid==tfmmRot_2_2)? 1.0f: 0.0f;
                }
                if(twoconfs) {
                    sfragfct = ndxCache3[threadIdx.y][sxx];
                    mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                    mloct = ((qryndx * maxnsteps + (lTOPN2 + sxx)) * ndbCtrgs + dbstrndx) * nTTranformMatrix + tid;
                    if(0 <= sfragfct)
                        wrkmemtm[mloct] = wrkmemtmibest[mlocs];//GMEM READ/WRITE
                    else if(firstit) {
                        //initialize otherwise; GMEM READ/WRITE
                        wrkmemtm[mloct] = (tid==tfmmRot_0_0 || tid==tfmmRot_1_1 || tid==tfmmRot_2_2)? 1.0f: 0.0f;
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
#define INSTANTIATE_SaveTopNScoresAndTMsAmongSecondaryBests(tpCOMPLEX) \
    template __global__ void SaveTopNScoresAndTMsAmongSecondaryBests<tpCOMPLEX>( \
        const int lTOPN, const int depth, const bool firstit, const bool twoconfs, \
        const int rfnfragfctinit, const uint ndbCstrs, const uint maxnsteps, const uint effnsteps, \
        const float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemtm, \
        float* __restrict__ wrkmemaux, \
        const uint ndbCchns, const int seedapproachstruct);

INSTANTIATE_SaveTopNScoresAndTMsAmongSecondaryBests(false);
INSTANTIATE_SaveTopNScoresAndTMsAmongSecondaryBests(true);
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// SaveTopNScoresAndTMsAmongBests: save top N scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the first
// N locations of fragment factors;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemtm, working memory for selected transformation matrices;
// wrkmemaux, auxiliary working memory;
// ndbCchns, total number of reference chains for complexes;
// 
__global__
void SaveTopNScoresAndTMsAmongBests(
    const uint lTOPN,
    const uint ndbCstrs,
    const uint maxnsteps,
//     const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    const uint ndbCchns,
    const int windowsize,
    const float scorethreshold)
{
    enum {
        lXDIM = CUS1_TBSP_SCORE_MAX_XDIM,
        lYDIM = CUS1_TBSP_SCORE_MAX_YDIM
    };

    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;
    __shared__ float scvCache[lYDIM][lXDIM+1];
    __shared__ uint ndxCache[lYDIM][lXDIM+1];

    scvCache[threadIdx.x][threadIdx.y] = 0.0f;
    ndxCache[threadIdx.x][threadIdx.y] = 0;

    if(threadIdx.y == 0 && threadIdx.x == 0 && windowsize && scorethreshold) {
        int addr, length;
        GetQueryComplexAddressLength(qryndx, addr, length);
        ndxCache[0][lXDIM] = (uint)length;
    }

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < maxnsteps/*effnsteps*/; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
        if(scvCache[threadIdx.x][threadIdx.y] < bscore) {
            scvCache[threadIdx.x][threadIdx.y] = bscore;
            ndxCache[threadIdx.x][threadIdx.y] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //sort scores and accompanying indices:
    BatcherSortYDIMparallel<lYDIM,false/*descending*/>(
        lYDIM, scvCache[threadIdx.x], ndxCache[threadIdx.x]);
    //no sync: sync'ed in BatcherSortYDIMparallel

    //scvCache[...][0..lTOPN-1]now contain top N scores
    uint sfragfct = ndxCache[threadIdx.x][threadIdx.y];

    //write scores first
    if(threadIdx.y < lTOPN && threadIdx.y < maxnsteps) {
        uint mlocs = ((qryndx * maxnsteps + threadIdx.y) * nTAuxWorkingMemoryVars) * ndbCstrs;
        // uint mloct = ((qryndx * maxnsteps + threadIdx.y) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        if(dbstrndx < ndbCstrs) {
            float bscore = scvCache[threadIdx.x][threadIdx.y];
            int convflag = 0;
            if(threadIdx.y == 0)
                convflag = wrkmemaux[mlocs + tawmvConverged * ndbCstrs + dbstrndx];
            if(0.0f < bscore) convflag = convflag & (~CONVERGED_SCOREDP_bitval);//reset
            else convflag = convflag | CONVERGED_SCOREDP_bitval;//set
            //NOTE: COMPLEXES: cannot adjust global/local convergence here:
            //NOTE: COMPLEXES: writing in the same address space!
            //NOTE: COMPLEXES: use another kernel (ConditionalInitFromComplex)!
            wrkmemaux[mlocs + tawmvConverged * ndbCstrs + dbstrndx] = convflag;
            //NOTE: no WRITE as scores to be recomputed by DP!
            //NOTE: for complexes, it can overwrite other data!
            // if(sfragfct != threadIdx.y)
            //     wrkmemaux[mloc + tawmvBestScore * ndbCtrgs + dbstrndx] = bscore;
            if(tmpdpdiagbuffers) {
                uint mtmpt = ((qryndx * lTOPN + threadIdx.y) * nTAuxWorkingMemoryVars) * ndbCstrs;
                float grand = wrkmemaux[mlocs + tawmvGrandBest * ndbCstrs + dbstrndx];
                if(windowsize && scorethreshold) {
                    int addr, length;
                    GetDbComplexAddressLength(dbstrndx, addr, length);
                    length = myhdmin((int)ndxCache[0][lXDIM], length);
                    length = myhdmin(windowsize, length);
                    if(bscore < scorethreshold * (float)length)
                        convflag = convflag | CONVERGED_SCOREDP_bitval;
                }
                tmpdpdiagbuffers[mtmpt + tawmvConverged * ndbCstrs + dbstrndx] = convflag;
                if(threadIdx.y < 1)
                    tmpdpdiagbuffers[mtmpt + tawmvGrandBest * ndbCstrs + dbstrndx] = grand;
            }
        }
    }

    //NOTE: no sync as long as caches are not overwritten from the last sync:
    //__syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    constexpr int nmtxs = lXDIM / nTTranformMatrix;
    int ndx = 0;//relative reference index
    for(int i = 1; i < nmtxs; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //READ and WRITE iteration-best transformation matrices;
    //rearrange lTOPN best performing mtxs at the first slots (sfragfct indices)
    for(int sx = 0; sx < (int)lTOPN; sx += nmtxs) {
        if(threadIdx.x < nTTranformMatrix * nmtxs && 
           threadIdx.x < nTTranformMatrix * (ndx+1) && dbstrndx < ndbCstrs) {
            uint tid = threadIdx.x - nTTranformMatrix * ndx;
            uint sxx = sx + ndx;
            if(sxx < lTOPN && sxx < maxnsteps) {
                //NOTE: indexing changed so that threadIdx.y refers to a different reference
                sfragfct = ndxCache[threadIdx.y][sxx];
                //float bscore = scvCache[threadIdx.y][sxx];
                //NOTE: always copy tfms for consistent results, as these are 
                //NOTE: used unconditionally in calculating swift scores!
                if(1/* 0.0f < bscore */) {
                    uint mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                    uint mloct = ((qryndx * maxnsteps + sxx) * ndbCtrgs + dbstrndx) * nTTranformMatrix + tid;
                    wrkmemtm[mloct] = wrkmemtmibest[mlocs];//GMEM READ/WRITE
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// ConditionalInitToComplex: save and initialize aux memory variables after 
// finishing the spatial index-based stage;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__
void ConditionalInitToComplex(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    const uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint sfragfct = blockIdx.z;//fragment factor

    const uint mlocs = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(dbstrndx < ndbCstrs) {
        wrkmemaux[mlocs + tawmvGrandBest * ndbCstrs + dbstrndx] = 0.0f;
        wrkmemaux[mlocs + tawmvBestScore * ndbCstrs + dbstrndx] = 0.0f;
        wrkmemaux[mlocs + tawmvScore * ndbCstrs + dbstrndx] = 0.0f;
        wrkmemaux[mlocs + tawmvConverged * ndbCstrs + dbstrndx] = 0.0f;
    }
}

// -------------------------------------------------------------------------
// ConditionalInitFromComplex: copy back aux memory variables after
// finishing the spatial index-based stage;
// grandupdate, flag of updating grand scores;
// ndbCcpxs, total number of reference complexes in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, memory section for temporary data;
// wrkmemaux, auxiliary working memory;
// 
__global__
void ConditionalInitFromComplex(
    const uint lTOPN,
    const int grandupdate,
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux)
{
    // enum {lTOPN = CUS1_TBSP_DPSCORE_TOP_N};

    const uint dbcpxndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint sfragfct = blockIdx.z;//fragment factor

    const uint mlocs = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
    const uint mtmpt = ((qryndx * lTOPN + sfragfct) * nTAuxWorkingMemoryVars) * ndbCcpxs;

    if(dbcpxndx < ndbCcpxs) {
        float convflag = tmpdpdiagbuffers[mtmpt + tawmvConverged * ndbCcpxs + dbcpxndx];
        wrkmemaux[mlocs + tawmvConverged * ndbCstrs + dbcpxndx] = convflag;
        if(grandupdate && sfragfct < 1) {
            float grand = tmpdpdiagbuffers[mtmpt + tawmvGrandBest * ndbCcpxs + dbcpxndx];
            wrkmemaux[mlocs + tawmvGrandBest * ndbCstrs + dbcpxndx] = grand;
        }
    }
}

// // -------------------------------------------------------------------------
// // ReformatTopNFlagsComplex: populate flags of top N scores for the next
// // stage of complex processing;
// // nqrycpxs, total number of query complexes in the chunk;
// // ndbCcpxs, total number of reference complexes in the chunk;
// // maxnsteps, max number of steps performed for each reference;
// // NOTE: memory pointers should be aligned!
// // wrkmemaux, auxiliary working memory;
// // ndbCchns, total number of reference complex chains in the chunk;
// // 
// __global__
// void ReformatTopNFlagsComplex(
//     const uint nqrycpxs,
//     const uint ndbCcpxs,
//     const uint maxnsteps,
//     float* __restrict__ wrkmemaux,
//     const uint ndbCchns)
// {
//     enum {
//         lXDIM = CUS1_TBSP_SCORE_MAX_XDIM,
//         lYDIM = CUS1_TBSP_DPSCORE_TOP_N
//     };

//     __shared__ float scvCache[lYDIM][lXDIM + 1];

//     scvCache[threadIdx.y][threadIdx.x] = 0.0f;

//     //iterate over complexes and populate convergence flags so that they conform to
//     //final complex format (avoiding overwriting complex-specific flags):
//     for(int qryndx = nqrycpxs - 1; 0 <= qryndx; qryndx--)
//     {
//         for(int dbcpxndx0 = ndbCcpxs - 1; 0 <= dbcpxndx0; dbcpxndx0 -= blockDim.x)
//         {
//             int dbcpxndx = dbcpxndx0 - (int)threadIdx.x;//complex index

//             uint mlocs = ((qryndx * maxnsteps + threadIdx.y) * nTAuxWorkingMemoryVars) * ndbCcpxs;
//             uint mloct = ((qryndx * maxnsteps + threadIdx.y) * nTAuxWorkingMemoryVars) * ndbCchns;

//             //read and then write after sync at most blockDim.x flags:
//             if(0 <= dbcpxndx && mlocs != mloct)
//                 scvCache[threadIdx.y][threadIdx.x] =
//                     atomicExch(&wrkmemaux[mlocs + tawmvConverged * ndbCcpxs + dbcpxndx], 0.0f);

//             __syncthreads();

//             if(0 <= dbcpxndx && mlocs != mloct)
//                 wrkmemaux[mloct + tawmvConverged * ndbCchns + dbcpxndx] =
//                     scvCache[threadIdx.y][threadIdx.x];
//         }
//     }
// }

// -------------------------------------------------------------------------



#if 0
// -------------------------------------------------------------------------
// SaveBestDPscoreAndTMAmongDPswifts: save best DP scores and respective 
// transformation matrices by considering all partial DP swift scores 
// calculated over all fragment factors; write the information to the 
// location of fragment factor 0;
// WRITEFRAGINFO, template parameter, write fragment specifications too;
// READSCORE, read previously written DP swift score;
// STEPx5, template parameter, multiply the step by 5 when calculating 
// query and reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// qryfragfct, fragment factor for query (to be multiplied by step dependent 
// upon lengths);
// rfnfragfct, fragment factor for reference;
// fragndx, fragment index determining the fragment length;
// NOTE: memory pointers should be aligned!
// wrkmemtmtarget, working memory for iteration-best (target) transformation matrices;
// tfmmem, memory for transformation matrices;
// wrkmemaux, auxiliary working memory;
// 
// template<bool WRITEFRAGINFO, bool READSCORE, bool STEPx5>
__global__
void SaveBestDPscoreAndTMAmongDPswifts(
    bool WRITEFRAGINFO, bool READSCORE,bool STEPx5,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint effnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    //distances in positions to the beginnings of the reference structures:
    __shared__ uint dbstrdst[CUS1_TBSP_SCORE_MAX_XDIM];
    //lengths of references; the last index is for the query
    __shared__ int dbstrlen[CUS1_TBSP_SCORE_MAX_XDIM+1];
    int qrylen;//query length

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    //coalesced reads of lengths an addresses of multiple reference structures:
    if(dbstrndx < ndbCstrs) {
        if(threadIdx.y == 0) dbstrlen[threadIdx.x] = GetDbStrLength(dbstrndx);
        if(threadIdx.y == 1) dbstrdst[threadIdx.x] = GetDbStrDst(dbstrndx);
    }

    if(threadIdx.y == 2 && threadIdx.x == 0)
        dbstrlen[CUS1_TBSP_SCORE_MAX_XDIM] = GetQueryLength(qryndx);

    __syncthreads();

    qrylen = dbstrlen[CUS1_TBSP_SCORE_MAX_XDIM];

    //no sync; dbstrlen cache will not be overwritten below!

    for(uint sfragfct = threadIdx.y; sfragfct < effnsteps; sfragfct += blockDim.y) {
        float dpscore = 0.0f;
        int dblen = ndbCposs + dbxpad;
        int yofff = (qryndx * maxnsteps + sfragfct) * dblen;
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;

        if(dbstrndx < ndbCstrs)
            //READ; uncoalesced for multiple references, but rare kernel call and 
            //compact thread block packaging counterbalance this inefficiency;
            //NOTE: last score is considered: assumed no banded alignment;
            dpscore = tmpdpdiagbuffers[doffs + dbstrdst[threadIdx.x] + dbstrlen[threadIdx.x]-1];

        if(scvCache[threadIdx.y][threadIdx.x] < dpscore) {
            scvCache[threadIdx.y][threadIdx.x] = dpscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUS1_TBSP_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    uint sfragfct = ndxCache[0][threadIdx.x];
    bool wrtscore = 0;

    //write scores first
    if(threadIdx.y == 0) {
        ndxCache[1][threadIdx.x] = 0;
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        //uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs) {
            float prevdpscore = -1.0f;
            float dpscore = scvCache[0][threadIdx.x];
            //coalesced READ/WRITE for multiple references
            if(READSCORE) prevdpscore = wrkmemaux[mloc0 + tawmvScore * ndbCstrs + dbstrndx];
            ndxCache[1][threadIdx.x] = wrtscore = (prevdpscore < dpscore);//reuse cache
            if(wrtscore) {
                wrkmemaux[mloc0 + tawmvScore * ndbCstrs + dbstrndx] = dpscore;
                if(WRITEFRAGINFO) {
                    int qrypos, rfnpos;
                    GetQryRfnPos_varP(STEPx5, qrypos, rfnpos,
                        qrylen, dbstrlen[threadIdx.x], sfragfct, qryfragfct, rfnfragfct, fragndx);
                    float frgndx = fragndx;//frag length to be determined by GetNAlnPoss_var
                    float frgpos = 0.0f;//fragpos is 0 for this particular calculation
                    wrkmemaux[mloc0 + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
                    wrkmemaux[mloc0 + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
                    wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCstrs + dbstrndx] = frgndx;
                    wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs + dbstrndx] = frgpos;
                }
            }
        }
    }

    __syncthreads();

    //NOTE: change indexing so that threadIdx.y refers to a different reference
    sfragfct = ndxCache[0][threadIdx.y];
    wrtscore = ndxCache[1][threadIdx.y];

    __syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    //READ and WRITE iteration-best transformation matrices
    if(threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        if(wrtscore) wrkmemtmtarget[mloc0] = wrkmemtm[mloc];//READ/WRITE to gmem
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
// #define INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(tpWRITEFRAGINFO,tpREADSCORE,tpSTEPx5) 
//     template __global__ void SaveBestDPscoreAndTMAmongDPswifts<tpWRITEFRAGINFO,tpREADSCORE,tpSTEPx5>( 
//         const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, 
//         const uint maxnsteps, const uint effnsteps, 
//         int qryfragfct, int rfnfragfct, int fragndx, 
//         const float* __restrict__ tmpdpdiagbuffers, 
//         const float* __restrict__ wrkmemtm, 
//         float* __restrict__ wrkmemtmtarget, 
//         float* __restrict__ wrkmemaux);
// 
// INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(false,false,false);
// INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(false,false,true);
// INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(false,true,false);
// INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(false,true,true);

// -------------------------------------------------------------------------
#endif



// -------------------------------------------------------------------------
// SortBestDPscoresAndTMsAmongDPswifts: sort best DP scores and then save 
// them along with respective transformation matrices by considering all 
// partial DP swift scores calculated over all fragment factors; write the 
// information to the first fragment factor locations;
// COMPLEX, template parameter, complex configuration (processing complexes);
// nbranches, #final superposition-stage branches for further exploration
// (CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT);
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, memory section of DP scores;
// wrkmemtm, input working memory of calculated transformation matrices;
// wrkmemtmtarget, working memory for iteration-best (target) transformation matrices;
// wrkmemaux, auxiliary working memory;
// ndbCchns, total #reference chains in the chunk (when COMPLEX set);
// 
template<bool COMPLEX>
__global__
void SortBestDPscoresAndTMsAmongDPswifts(
    const uint lTOPN,
    const uint nbranches,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux,
    const uint ndbCchns)
{
    enum {
        lYDIM = CUS1_TBSP_DPSCORE_MAX_YDIM,
        lXDIM = CUS1_TBSP_DPSCORE_MAX_XDIM,
        // lTOPN = CUS1_TBSP_DPSCORE_TOP_N,
        lREFN = CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT,
        lMAXS = CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS,
        lREFNxMAXS = lREFN * lMAXS
        // lTOPNxMAXS = CUS1_TBSP_DPSCORE_TOP_N_CONFIGS
    };
    const uint lTOPNxMAXS = lTOPN * lMAXS;
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint secndx = blockIdx.z;//score section index
    const uint ndbCtrgs = ndbCchns? ndbCchns: ndbCstrs;
    __shared__ float scvCache[lYDIM][lXDIM+1];
    __shared__ uint ndxCache[lYDIM][lXDIM+1];
    //distances in positions to the beginnings of the reference structures:
    __shared__ uint dbstrdst[lXDIM+1];
    //lengths of references; the last index is for the query
    __shared__ int dbstrlen[lXDIM+1];

    scvCache[threadIdx.x][threadIdx.y] = 0.0f;
    ndxCache[threadIdx.x][threadIdx.y] = 0;

    if(!COMPLEX) {
        //coalesced reads of lengths an addresses of multiple reference structures:
        if(dbstrndx < ndbCstrs) {
            if(threadIdx.y == 0) dbstrlen[threadIdx.x] = GetDbStrLength(dbstrndx);
            if(threadIdx.y == 1) dbstrdst[threadIdx.x] = GetDbStrDst(dbstrndx);
        }
    }

    __syncthreads();

    for(uint sfragfct = secndx * lTOPN + threadIdx.y; 
        sfragfct < (secndx + 1) * lTOPN && sfragfct < maxnsteps; 
        sfragfct += blockDim.y)
    {
        float dpscore = 0.0f;
        int dblen = ndbCposs + dbxpad;
        int yofff = (qryndx * maxnsteps + sfragfct) * dblen;
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;

        int convflag = CONVERGED_SCOREDP_bitval;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        if(dbstrndx < ndbCstrs)
            convflag = wrkmemaux[mloc + tawmvConverged * ndbCtrgs + dbstrndx];

        if(convflag == 0/* dbstrndx < ndbCstrs */) {
            if(COMPLEX) dpscore = wrkmemaux[mloc + tawmvBestScore * ndbCtrgs + dbstrndx];
            else 
                //READ; uncoalesced for multiple references, but rare kernel call and 
                //compact thread block packaging counterbalance this inefficiency;
                //NOTE: last score is considered: assumed no banded alignment;
                dpscore = tmpdpdiagbuffers[doffs + dbstrdst[threadIdx.x] + dbstrlen[threadIdx.x]-1];
        }

        if(scvCache[threadIdx.x][threadIdx.y] < dpscore) {
            scvCache[threadIdx.x][threadIdx.y] = dpscore;
            ndxCache[threadIdx.x][threadIdx.y] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //sort scores and accompanying indices:
    BatcherSortYDIMparallel<lYDIM,false/*descending*/>(
        lYDIM, scvCache[threadIdx.x], ndxCache[threadIdx.x]);
    //no sync: sync'ed in BatcherSortYDIMparallel

    //scvCache[...][0..lTOPN-1] now contain s
    //uint sfragfct = ndxCache[threadIdx.x][threadIdx.y];

    //write scores first (NOTE: scores required only for testing):
    if(threadIdx.y < lTOPN && (secndx * lTOPN + threadIdx.y) < maxnsteps) {
        uint mloc = ((qryndx * maxnsteps + (secndx * lTOPN + threadIdx.y)) * nTAuxWorkingMemoryVars) * ndbCtrgs;
        if(dbstrndx < ndbCstrs) {
            float dpscore = scvCache[threadIdx.x][threadIdx.y];
            //NOTE: convergence flags have to be set/reset again after new sorting.
            int convflag = 0;
            if(threadIdx.y == 0 && secndx == 0)
                convflag = wrkmemaux[mloc + tawmvConverged * ndbCtrgs + dbstrndx];
            if(0.0f < dpscore) convflag = convflag & (~CONVERGED_SCOREDP_bitval);//reset
            else convflag = convflag | CONVERGED_SCOREDP_bitval;//set
            //adjust global/local convergence
            wrkmemaux[mloc + tawmvConverged * ndbCtrgs + dbstrndx] = convflag;
            //coalesced WRITE for multiple references (for test)
            // wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx] = dpscore;
            // reset score/may not be reinitialized during refinement (bugfix: 231110/v0.13):
            wrkmemaux[mloc + tawmvBestScore * ndbCtrgs + dbstrndx] = 0.0f;
        }
    }

    //NOTE: no sync as long as caches are not overwritten from the last sync:
    //__syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    constexpr int nmtxs = lXDIM / nTTranformMatrix;
    int ndx = 0;//relative reference index
    for(int i = 1; i < nmtxs; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //READ and WRITE iteration-best transformation matrices;
    //rearrange lREFN best performing mtxs at the first slots (sfragfct indices)
    for(int sx = 0; sx < (int)nbranches; sx += nmtxs) {
        if(threadIdx.x < nTTranformMatrix * nmtxs && 
           threadIdx.x < nTTranformMatrix * (ndx+1) && dbstrndx < ndbCstrs) {
            uint tid = threadIdx.x - nTTranformMatrix * ndx;
            uint sxx = sx + ndx;
            if(sxx < nbranches && sxx < maxnsteps) {
                //NOTE: indexing changed so that threadIdx.y refers to a different reference
                uint sfragfct = ndxCache[threadIdx.y][sxx];
                float dpscore = scvCache[threadIdx.y][sxx];
                ////NOTE: copy tfms irrespective of scores for consistency!
                if(0.0f < dpscore) {
                    uint mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCtrgs + dbstrndx) * nTTranformMatrix + tid;
                    //NOTE: lREFN for target tms!
                    uint mloct = ((qryndx * lREFNxMAXS + (secndx * nbranches + sxx)) * ndbCtrgs + dbstrndx) *
                            nTTranformMatrix + tid;
                    wrkmemtmtarget[mloct] = wrkmemtm[mlocs];//GMEM READ/WRITE
                }
            }
        }
    }

    //READ and WRITE iteration-best chain assignments;
    //rearrange best performing assignments at the first slots (sfragfct indices)
    if(COMPLEX) {
        //NOTE: operating on complex indices and numers!
        //dbstrndx is a complex sn; ndbCstrs, #complexes;
        int dbcpxN = 0, dbcpxdstN = 0;
        int qrycpxN = 0;//, qrycpxdstN = 0;
        if(threadIdx.x == 0) {
            qrycpxN = GetQueryStrField<INTYPE,pcx2DN>(qryndx);
            //query complex's 1st chain index:
            // qrycpxdstN = GetQueryStrField<LNTYPE,pcx2DDstN>(qryndx);
        }
        if(threadIdx.x == 0 && dbstrndx < ndbCstrs) {
            dbcpxN = GetDbStrField<INTYPE,pcx2DN>(dbstrndx);
            dbcpxdstN = GetDbStrField<LNTYPE,pcx2DDstN>(dbstrndx);
        }
        //NOTE: valid only when blockDim.x <= warpsize!
        //NOTE: warp syncs across blockDim.y dimension (complexes):
        qrycpxN = __shfl_sync(0xffffffff, qrycpxN, 0/*srcLane*/);
        dbcpxN = __shfl_sync(0xffffffff, dbcpxN, 0/*srcLane*/);
        // qrycpxdstN = __shfl_sync(0xffffffff, qrycpxdstN, 0/*srcLane*/);
        dbcpxdstN = __shfl_sync(0xffffffff, dbcpxdstN, 0/*srcLane*/);
        int Q = myhdmin(qrycpxN, dbcpxN);

        for(uint sx = 0; sx < nbranches && sx < maxnsteps; sx++) {
            const uint sfragfct = ndxCache[threadIdx.y][sx];
            //const float dpscore = scvCache[threadIdx.y][sx];
            //NOTE: copy assignments irrespective of dpscore:
            //NOTE: assignment checks only the CONVERGED_LOWTMSC_bitval flag at sfragfct 0!
            if(/* 0.0f < dpscore &&  */dbstrndx < ndbCstrs) {
                //write assignment (chain indices) to gmem;
                //NOTE: Q < dbcpxN always
                for(int q = threadIdx.x; q < Q; q += lXDIM) {
                    const uint locqryndx = qryndx;//complex's sn;//qrycpxDstN;//query complex's 1st chain index
                    //reference complex's 1st chain index + running index:
                    const uint locdbstrndx = (uint)(dbcpxdstN + q);
                    //NOTE: assignments corresponding to sfragfct are written beyond the lTOPNxMAXS margin:
                    //NOTE: (copying from the same address space!)
                    const uint mlocs = ((locqryndx * maxnsteps + (lTOPNxMAXS + sfragfct)) *
                        nTAuxWorkingMemoryVars) * ndbCtrgs;
                    const uint mloct =
                        ((locqryndx * maxnsteps + (secndx * nbranches + sx)) * nTAuxWorkingMemoryVars) * ndbCtrgs;

                    wrkmemaux[mloct + tawmvInitialBest * ndbCtrgs + locdbstrndx] = //GMEM READ/WRITE
                        wrkmemaux[mlocs + tawmvInitialBest * ndbCtrgs + locdbstrndx];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
//Instantiations:
#define INSTANTIATE_SortBestDPscoresAndTMsAmongDPswifts(tpCOMPLEX) \
    template __global__ void SortBestDPscoresAndTMsAmongDPswifts<tpCOMPLEX>( \
        const uint lTOPN, const uint nbranches, const uint ndbCstrs, \
        const uint ndbCposs, const uint dbxpad, const uint maxnsteps, \
        const float* __restrict__ tmpdpdiagbuffers, \
        const float* __restrict__ wrkmemtm, \
        float* __restrict__ wrkmemtmtarget, \
        float* __restrict__ wrkmemaux, \
        const uint ndbCchns);

INSTANTIATE_SortBestDPscoresAndTMsAmongDPswifts(false);
INSTANTIATE_SortBestDPscoresAndTMsAmongDPswifts(true);
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// SortBestDPscoresAndTMsAmongDPswiftsComplex: sort best DP scores and then
// save them along with respective transformation matrices by considering all 
// partial DP swift scores calculated over all fragment factors;
// NOTE: complex version;
// nbranches, #final superposition-stage branches for further exploration
// (CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT);
// ndbCcpxs, total number of reference complexes in the chunk;
// ndbCstrs, total #reference chains in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// maxnstepsmem2, maxnsteps version for wrkmem2;
// NOTE: memory pointers should be aligned!
// wrkmemtm, input working memory of calculated transformation matrices;
// wrkmemtmtarget, working memory for iteration-best (target) transformation matrices;
// wrkmemaux, auxiliary working memory;
// wrkmem2, working memory for chain scores and assignments;
// 
__global__
void SortBestDPscoresAndTMsAmongDPswiftsComplex(
    const uint lTOPN,
    const uint nbranches,
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint maxnstepsmem2,
    const int tfmtarget,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2)
{
    enum {
        lYDIM = CUS1_TBSP_DPSCORE_MAX_YDIM,
        lXDIM = CUS1_TBSP_DPSCORE_MAX_XDIM,
        // lTOPN = CUS1_TBSP_DPSCORE_TOP_N,
        lREFN = CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT,
        lMAXS = CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS,
        lREFNxMAXS = lREFN * lMAXS
    };
    //complex index:
    uint dbcpxndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint secndx = blockIdx.z;//score section index
    __shared__ float scvCache[lYDIM][lXDIM+1];
    __shared__ uint ndxCache[lYDIM][lXDIM+1];

    scvCache[threadIdx.x][threadIdx.y] = 0.0f;
    ndxCache[threadIdx.x][threadIdx.y] = 0;

    __syncthreads();

    for(uint sfragfct = secndx * lTOPN + threadIdx.y; 
        sfragfct < (secndx + 1) * lTOPN && sfragfct < maxnsteps; 
        sfragfct += blockDim.y)
    {
        float dpscore = 0.0f;
        int convflag = CONVERGED_SCOREDP_bitval;
        const uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        const uint mloc2 = ((qryndx * maxnstepsmem2 + sfragfct) * nTWorkingMemory2VarsAssignment) * ndbCstrs;

        if(dbcpxndx < ndbCcpxs)
            convflag = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbcpxndx];

        if(convflag == 0/* dbcpxndx < ndbCstrs */)
            dpscore = wrkmem2[mloc2 + twm2aCpxScore * ndbCstrs + dbcpxndx];

        if(scvCache[threadIdx.x][threadIdx.y] < dpscore) {
            scvCache[threadIdx.x][threadIdx.y] = dpscore;
            ndxCache[threadIdx.x][threadIdx.y] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //sort scores and accompanying indices:
    BatcherSortYDIMparallel<lYDIM,false/*descending*/>(
        lYDIM, scvCache[threadIdx.x], ndxCache[threadIdx.x]);
    //no sync: sync'ed in BatcherSortYDIMparallel

    //scvCache[...][0..lTOPN-1] now contain s
    //uint sfragfct = ndxCache[threadIdx.x][threadIdx.y];

    //write scores first (NOTE: scores required only for testing):
    if(threadIdx.y < lTOPN && (secndx * lTOPN + threadIdx.y) < maxnsteps) {
        uint mloc = ((qryndx * maxnsteps + (secndx * lTOPN + threadIdx.y)) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbcpxndx < ndbCcpxs) {
            float dpscore = scvCache[threadIdx.x][threadIdx.y];
            //NOTE: convergence flags have to be set/reset again after new sorting.
            int convflag = 0;
            if(threadIdx.y == 0 && secndx == 0)
                convflag = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbcpxndx];
            if(0.0f < dpscore) convflag = convflag & (~CONVERGED_SCOREDP_bitval);//reset
            else convflag = convflag | CONVERGED_SCOREDP_bitval;//set
            //adjust global/local convergence
            wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbcpxndx] = convflag;
            //coalesced WRITE for multiple references (for test)
            // wrkmemaux[mloc + tawmvScore * ndbCstrs + dbcpxndx] = dpscore;
            // reset score/may not be reinitialized during refinement (bugfix: 231110/v0.13):
            wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbcpxndx] = 0.0f;
        }
    }

    //NOTE: no sync as long as caches are not overwritten from the last sync:
    //__syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbcpxndx = blockIdx.x * blockDim.x + threadIdx.y;

    constexpr int nmtxs = lXDIM / nTTranformMatrix;
    int ndx = 0;//relative reference index
    for(int i = 1; i < nmtxs; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //READ and WRITE iteration-best transformation matrices;
    //rearrange lREFN best performing mtxs at the first slots (sfragfct indices)
    for(int sx = 0; sx < (int)nbranches; sx += nmtxs) {
        if(threadIdx.x < nTTranformMatrix * nmtxs && 
           threadIdx.x < nTTranformMatrix * (ndx+1) && dbcpxndx < ndbCcpxs) {
            uint tid = threadIdx.x - nTTranformMatrix * ndx;
            uint sxx = sx + ndx;
            if(sxx < nbranches && sxx < maxnsteps) {
                //NOTE: indexing changed so that threadIdx.y refers to a different reference
                uint sfragfct = ndxCache[threadIdx.y][sxx];
                float dpscore = scvCache[threadIdx.y][sxx];
                ////NOTE: copy tfms irrespective of scores for consistency!
                if(0.0f < dpscore) {
                    uint mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbcpxndx) * nTTranformMatrix + tid;
                    //NOTE: lREFN for target tms!
                    uint mloct = ((qryndx * lREFNxMAXS + (secndx * nbranches + sxx)) * ndbCstrs + dbcpxndx) *
                            nTTranformMatrix + tid;
                    if(tfmtarget == WRKMEMTMIBEST)
                        mloct = ((qryndx * maxnsteps + sxx) * ndbCstrs + dbcpxndx) * nTTranformMatrix + tid;
                    else if(tfmtarget == TFMMEM)
                        mloct = (qryndx * ndbCstrs + dbcpxndx) * nTTranformMatrix + tid;
                    if((tfmtarget != TFMMEM) || (tfmtarget == TFMMEM && sxx == 0))
                        wrkmemtmtarget[mloct] = wrkmemtm[mlocs];//GMEM READ/WRITE
                }
            }
        }
    }

    //READ and WRITE iteration-best chain assignments;
    //rearrange best performing assignments at the first slots (sfragfct indices)
    int dbcpxN = 0, dbcpxdstN = 0;
    int qrycpxN = 0;//, qrycpxdstN = 0;
    if(threadIdx.x == 0) {
        qrycpxN = GetQueryStrField<INTYPE,pcx2DN>(qryndx);
        //query complex's 1st chain index:
        // qrycpxdstN = GetQueryStrField<LNTYPE,pcx2DDstN>(qryndx);
    }
    if(threadIdx.x == 0 && dbcpxndx < ndbCcpxs) {
        dbcpxN = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);
        dbcpxdstN = GetDbStrField<LNTYPE,pcx2DDstN>(dbcpxndx);
    }
    //NOTE: valid only when blockDim.x <= warpsize!
    //NOTE: warp syncs across blockDim.y dimension (complexes):
    qrycpxN = __shfl_sync(0xffffffff, qrycpxN, 0/*srcLane*/);
    dbcpxN = __shfl_sync(0xffffffff, dbcpxN, 0/*srcLane*/);
    // qrycpxdstN = __shfl_sync(0xffffffff, qrycpxdstN, 0/*srcLane*/);
    dbcpxdstN = __shfl_sync(0xffffffff, dbcpxdstN, 0/*srcLane*/);
    int Q = myhdmin(qrycpxN, dbcpxN);

    //NOTE: rearrange data in two phases;
    //NOTE: phase 1: reuse section twm2aChnScore for copying from the same address space:
    for(uint sx = 0; sx < nbranches && sx < maxnsteps; sx++) {
        const uint sfragfct = ndxCache[threadIdx.y][sx];
        //const float dpscore = scvCache[threadIdx.y][sx];
        //NOTE: copy assignments irrespective of dpscore:
        //NOTE: assignment checks only the CONVERGED_LOWTMSC_bitval flag at sfragfct 0!
        if(/* 0.0f < dpscore &&  */dbcpxndx < ndbCcpxs) {
            //write assignment (chain indices) to gmem;
            //NOTE: Q <=dbcpxN always!
            for(int q = threadIdx.x; q < Q; q += lXDIM) {
                const uint dbstrndx = (uint)(dbcpxdstN + q);//1st chain index + running index
                const uint mloc2s = ((qryndx * maxnstepsmem2 + sfragfct) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
                const uint mloc2t = ((qryndx * maxnstepsmem2 + (secndx * nbranches + sx)) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
                wrkmem2[mloc2t + twm2aChnScore * ndbCstrs + dbstrndx] = //GMEM READ/WRITE
                    wrkmem2[mloc2s + twm2aCCAssignment * ndbCstrs + dbstrndx];
            }
        }
    }

    __syncthreads();

    //NOTE: phase 2: copy back from section twm2aChnScore:
    for(uint sx = 0; sx < nbranches && sx < maxnsteps; sx++) {
        //const uint sfragfct = ndxCache[threadIdx.y][sx];
        //const float dpscore = scvCache[threadIdx.y][sx];
        //NOTE: copy assignments irrespective of dpscore:
        //NOTE: assignment checks only the CONVERGED_LOWTMSC_bitval flag at sfragfct 0!
        if(/* 0.0f < dpscore &&  */dbcpxndx < ndbCcpxs) {
            //write assignment (chain indices) to gmem;
            //NOTE: Q <=dbcpxN always!
            for(int q = threadIdx.x; q < Q; q += lXDIM) {
                const uint dbstrndx = (uint)(dbcpxdstN + q);//1st chain index + running index
                const uint mloc2t = ((qryndx * maxnstepsmem2 + (secndx * nbranches + sx)) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
                wrkmem2[mloc2t + twm2aCCAssignment * ndbCstrs + dbstrndx] = //GMEM READ/WRITE
                    wrkmem2[mloc2t + twm2aChnScore * ndbCstrs + dbstrndx];
            }
        }
    }
}



// -------------------------------------------------------------------------
// SortBestScoresAndTMsFromCPAssignmentsComplex: sort best complex scores
// originating from individual chain processing assignments;
// save these best scores along with respective transformation
// matrices to the first fragment factor locations;
// ndbCcpxs, total number of reference complexes in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// maxnstepsmem2, maxnsteps version for wrkmem2;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, memory section of DP scores;
// wrkmemtm, input working memory of calculated transformation matrices;
// wrkmemtmtarget, working memory for iteration-best (target) transformation matrices;
// wrkmemaux, auxiliary working memory;
// wrkmem2, working memory for chain scores and assignments;
// 
__global__
void SortBestScoresAndTMsFromCPAssignmentsComplex(
    const uint ndbCcpxs,
    const uint ndbCstrs,
    const uint maxnqrychains,
    const uint /* maxnrfnchains */,
    const uint maxnsteps,
    const uint effnsteps,
    const uint maxnstepsmem2,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ /* wrkmemaux */,
    float* __restrict__ wrkmem2)
{
    enum {
        lYDIM = CUS1_TBSP_CPXSCORE_MAX_YDIM,
        lXDIM = CUS1_TBSP_CPXSCORE_MAX_XDIM,
        lTOPN = CUS1_TBSP_CPXSCORE_TOP_N,
        MAX_NCHAINS = CUS1_TBSP_CPXSCORE_MAX_NCHAINS,
        lREFNxMAXS = CUS1_TBSP_DPSCORE_TOP_N_REFINEMENTxMAX_CONFIGS
    };
    //complex index (blockIdx.x, refn. serial number):
    uint dbcpxndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qrycpxndx = blockIdx.y;//query serial number
    __shared__ float scvCache[lYDIM][lXDIM+1];
    __shared__ uint ndxCache[lYDIM][lXDIM+1];
    //distances in #chains to the beginnings of reference complexes:
    __shared__ uint dbcpxdstN[lXDIM+1];
    __shared__ uint dbcpxN[lXDIM+1];

    scvCache[threadIdx.x][threadIdx.y] = 0.0f;
    ndxCache[threadIdx.x][threadIdx.y] = 0;
    if(threadIdx.y == 0) dbcpxdstN[threadIdx.x] = 0;

    //coalesced reads:
    if(threadIdx.y == 0 && dbcpxndx < ndbCcpxs) dbcpxdstN[threadIdx.x] = GetDbStrField<LNTYPE,pcx2DDstN>(dbcpxndx);
    if(threadIdx.y == 1 && dbcpxndx < ndbCcpxs) dbcpxN[threadIdx.x] = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);
    if(threadIdx.y == 2 && threadIdx.x == 0) dbcpxdstN[lXDIM] = GetQueryStrField<LNTYPE,pcx2DDstN>(qrycpxndx);
    if(threadIdx.y == 3 && threadIdx.x == 0) dbcpxN[lXDIM] = GetQueryStrField<INTYPE,pcx2DN>(qrycpxndx);

    __syncthreads();

    for(uint sfragfct = threadIdx.y;
        sfragfct < maxnsteps && sfragfct < effnsteps;
        sfragfct += blockDim.y)
    {
        if(ndbCcpxs <= dbcpxndx) continue;
        const uint linkrfnchain = sfragfct / maxnqrychains;
        const uint linkqrychain = sfragfct - linkrfnchain * maxnqrychains;
        if(MAX_NCHAINS < linkqrychain && MAX_NCHAINS < linkrfnchain) continue;
        if(dbcpxN[lXDIM] <= linkqrychain || dbcpxN[threadIdx.x] <= linkrfnchain) continue;
        //
        uint qryndx = sfragfct / maxnstepsmem2;
        uint sfragfct2 = sfragfct - qryndx * maxnstepsmem2;
        qryndx += dbcpxdstN[lXDIM];
        //
        float bscore = 0.0f;
        const uint mloc2 = ((qryndx * maxnstepsmem2 + sfragfct2) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
        //coalesced read for multiple references
        bscore = atomicExch(&wrkmem2[mloc2 + twm2aCpxScore * ndbCstrs + dbcpxndx], 0.0f);
        if(scvCache[threadIdx.x][threadIdx.y] < bscore) {
            scvCache[threadIdx.x][threadIdx.y] = bscore;
            ndxCache[threadIdx.x][threadIdx.y] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //sort scores and accompanying indices:
    BatcherSortYDIMparallel<lYDIM,false/*descending*/>(
        lYDIM, scvCache[threadIdx.x], ndxCache[threadIdx.x]);
    //no sync: sync'ed in BatcherSortYDIMparallel

    //scvCache[...][0..lYDIM-1] now contain sorted scores
    //uint sfragfct = ndxCache[threadIdx.x][threadIdx.y];

    //NOTE: no sync as long as caches are not overwritten from the last sync:
    //__syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbcpxndx = blockIdx.x * blockDim.x + threadIdx.y;

    constexpr int nmtxs = lXDIM / nTTranformMatrix;
    int ndx = 0;//relative reference index
    for(int i = 1; i < nmtxs; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //READ and WRITE iteration-best transformation matrices;
    //rearrange lTOPN best performing mtxs at the first slots (sfragfct indices)
    for(int sx = 0; sx < lTOPN; sx += nmtxs) {
        if(threadIdx.x < nTTranformMatrix * nmtxs && 
           threadIdx.x < nTTranformMatrix * (ndx+1) && dbcpxndx < ndbCcpxs) {
            uint tid = threadIdx.x - nTTranformMatrix * ndx;
            uint sxx = sx + ndx;
            if(sxx < lTOPN && sxx < maxnsteps) {
                //NOTE: indexing changed so that threadIdx.y refers to a different reference
                const uint sfragfct = ndxCache[threadIdx.y][sxx];
                const float bscore = scvCache[threadIdx.y][sxx];
                if(0.0f < bscore) {
                    //relative query and reference chain ids (tfm to be used for complex alignment refinement):
                    const uint linkrfnchain = sfragfct / maxnqrychains;
                    const uint linkqrychain = sfragfct - linkrfnchain * maxnqrychains;
                    const uint mlocs = ((dbcpxdstN[lXDIM] + linkqrychain) * ndbCstrs +
                        (dbcpxdstN[threadIdx.y] + linkrfnchain)) * nTTranformMatrix + tid;
                    //NOTE: lREFNxMAXS for target tms!
                    const uint mloct = ((qrycpxndx * lREFNxMAXS + sxx) * ndbCstrs + dbcpxndx) * nTTranformMatrix + tid;
                    wrkmemtmtarget[mloct] = wrkmemtm[mlocs];//GMEM READ/WRITE
                }
            }
        }
    }
}



// -------------------------------------------------------------------------
// ReformatDPswiftScores: reformat DP swift scores calculated over all 
// fragment factors previously and store them in the wrkmemaux memory 
// area for efficient parallel processing;
// NOTE: used exclusively for complexes!
// nqystrstartindex, #query chains to be added to the query index processed by the kernel;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure;
// maxnstepschains, maxnsteps version for temporary buffer tmpdpdiagbuffers;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, memory section of DP scores;
// wrkmemaux, auxiliary working memory;
// wrkmem2, working memory for chain scores and assignments;
// 
__global__
void ReformatDPswiftScores(
    const uint lTOPN,
    const uint nqystrstartindex,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint maxnstepschains,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2)
{
    enum {
        lYDIM = CUS1_TBSP_DPSCORE_MAX_YDIM,
        lXDIM = CUS1_TBSP_DPSCORE_MAX_XDIM
        // lTOPN = CUS1_TBSP_DPSCORE_TOP_N
    };
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndxpart = blockIdx.y;//query partial serial number
    const uint qryndx = nqystrstartindex + qryndxpart;//query serial number
    const uint secndx = blockIdx.z;//score section index
    //distances in positions to the beginnings of the reference structures:
    __shared__ uint dbstrdst[lXDIM+1];
    //lengths of references; the last index is for the query
    __shared__ int dbstrlen[lXDIM+1];
    __shared__ int dbstrtpf[lXDIM+1];
    //reference complex indices; the last index is for the query
    __shared__ int dbcpxndx[lXDIM+1];
    __shared__ int cnvflgs0[lXDIM+1];//global convergence flags

    if(threadIdx.y == 3 && threadIdx.x == 0)
        dbcpxndx[lXDIM] = GetQueryStrField<INTYPE,pps2DCpxI>(qryndx);
    if(threadIdx.y == 4 && threadIdx.x == 0) {
        uint dst = GetQueryDst(qryndx);
        dbstrtpf[lXDIM] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dst);
    }

    //coalesced reads of lengths an addresses of multiple reference structures:
    if(dbstrndx < ndbCstrs) {
        uint dst;
        if(threadIdx.y == 0) dbstrlen[threadIdx.x] = GetDbStrLength(dbstrndx);
        if(threadIdx.y == 1) dbstrdst[threadIdx.x] = dst = GetDbStrDst(dbstrndx);
        if(threadIdx.y == 1) dbstrtpf[threadIdx.x] = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dst);
        if(threadIdx.y == 2) dbcpxndx[threadIdx.x] = GetDbStrField<INTYPE,pps2DCpxI>(dbstrndx);
    }

    __syncthreads();

    if(threadIdx.y == 0) {
        cnvflgs0[threadIdx.x] = CONVERGED_LOWTMSC_bitval;
        uint mlocx0 = ((dbcpxndx[lXDIM] * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs) {
            cnvflgs0[threadIdx.x] = wrkmemaux[mlocx0 + tawmvConverged * ndbCstrs + dbcpxndx[threadIdx.x]];
            cnvflgs0[threadIdx.x] = (cnvflgs0[threadIdx.x] & CONVERGED_LOWTMSC_bitval);
        }
    }

    __syncthreads();

    for(uint sfragfct = secndx * lTOPN + threadIdx.y; 
        sfragfct < (secndx + 1) * lTOPN && sfragfct < maxnsteps; 
        sfragfct += blockDim.y)
    {
        float dpscore = 0.0f;
        int dblen = ndbCposs + dbxpad;
        int yofff = (qryndxpart * maxnstepschains + sfragfct) * dblen;
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;

        if(cnvflgs0[threadIdx.x]) continue;

        int convflag = CONVERGED_SCOREDP_bitval;
        uint mloc2 = ((qryndx * maxnstepschains + sfragfct) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
        uint mlocx = ((dbcpxndx[lXDIM] * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)
            convflag = wrkmemaux[mlocx + tawmvConverged * ndbCstrs + dbcpxndx[threadIdx.x]];

        if(convflag == 0 && dbstrndx < ndbCstrs &&
           GetMoleculeType(dbstrtpf[lXDIM]) == GetMoleculeType(dbstrtpf[threadIdx.x])) {
            //READ; uncoalesced for multiple references, but rare kernel call and 
            //compact thread block packaging counterbalance this inefficiency;
            //NOTE: last score is considered: assumed no banded alignment;
            dpscore = tmpdpdiagbuffers[doffs + dbstrdst[threadIdx.x] + dbstrlen[threadIdx.x]-1];
            dpscore = fabsf(dpscore);//scores negated in diagonal direction
        }

        if(dbstrndx < ndbCstrs)
            //write score:
            wrkmem2[mloc2 + twm2aChnScore * ndbCstrs + dbstrndx] = dpscore;
    }
}

// -------------------------------------------------------------------------
// ReformatDPScores: reformat DP scores (calculated for a single fragment
// factor) and store them in the wrkmem2 memory area for efficient
// parallel processing;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnstepsmem2, max number of steps for assignments in wrkmem2;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, memory section of DP scores;
// wrkmem2, working memory for chain scores and assignments;
// 
__global__
void ReformatDPScores(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnstepsmem2,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem2,
    const bool localseqaln)
{
    enum {lXDIM = CUS1_TBSP_DPSCORE1_MAX_XDIM};
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    //distances in positions to the beginnings of the reference structures:
    __shared__ uint dbstrdst[lXDIM+1];
    //lengths of references; the last index is for the query
    __shared__ int dbstrlen[lXDIM+1];
    __shared__ int dbstrtpf[lXDIM+1];

    if(threadIdx.x == 0) {
        uint dst = GetQueryDst(qryndx);
        dbstrtpf[lXDIM] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dst);
    }

    //coalesced reads of date of multiple reference structures:
    if(dbstrndx < ndbCstrs) {
        uint dst;
        dbstrlen[threadIdx.x] = GetDbStrLength(dbstrndx);
        dbstrdst[threadIdx.x] = dst = GetDbStrDst(dbstrndx);
        dbstrtpf[threadIdx.x] = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dst);
    }

    __syncthreads();

    float dpscore = 0.0f;
    int dblen = ndbCposs + dbxpad;
    int yofff = (qryndx * 1/*maxnsteps*/ + 0) * dblen;//NOTE: w/o maxnsteps (see dpw_btck)!
    int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;

    //NOTE: no convergence check because of additional cost of extra 
    //NOTE: readings (complex ids and flgas)!
    //NOTE: As a result, some dpscore can have non-meaningful values!
    const uint mloc2 = ((qryndx * maxnstepsmem2 + 0) * nTWorkingMemory2VarsAssignment) * ndbCstrs;

    if(dbstrndx < ndbCstrs) {
        //READ; uncoalesced for multiple references, but rare kernel call and 
        //compact thread block packaging counterbalance this inefficiency;
        //NOTE: last score is considered: assumed no banded alignment;
        if(GetMoleculeType(dbstrtpf[lXDIM]) == GetMoleculeType(dbstrtpf[threadIdx.x]))
            dpscore =
                tmpdpdiagbuffers[doffs + dbstrdst[threadIdx.x] + dbstrlen[threadIdx.x]-1];
        //diagonal direction correction:
        if(localseqaln == false) dpscore = fabsf(dpscore);//scores negated in diagonal direction
        else if(dpscore < DPSSDEFINITSCOREVAL_cmp) dpscore -= DPSSDEFINITSCOREVAL;//scores corrected
    }

    if(dbstrndx < ndbCstrs)
        //write score:
        wrkmem2[mloc2 + twm2aChnScore * ndbCstrs + dbstrndx] = dpscore;
}

// -------------------------------------------------------------------------
// ReformatCCRfnScores: reformat chain-to-chain refinement scores and store
// them in the wrkmem2 memory area for efficient parallel processing;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps0, max number of steps for assignments in wrkmemaux;
// maxnstepsmem2, max number of steps for assignments in wrkmem2;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// wrkmem2, working memory for chain scores and assignments;
// 
__global__
void ReformatCCRfnScores(
    const uint ndbCstrs,
    const uint maxnsteps0,
    const uint maxnstepsmem2,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2)
{
    enum {lXDIM = CUS1_TBSP_SCORE_SET_XDIM};
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;//structure index
    const uint qryndx = blockIdx.y;//query serial number
    // __shared__ uint dbstrdst[lXDIM+1];//distances
    // __shared__ int dbstrtpf[lXDIM+1];//types

    // if(threadIdx.x == 0) {
    //     uint dst = GetQueryDst(qryndx);
    //     dbstrtpf[lXDIM] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dst);
    // }

    // if(dbstrndx < ndbCstrs) {
    //     uint dst;
    //     dbstrdst[threadIdx.x] = dst = GetDbStrDst(dbstrndx);
    //     dbstrtpf[threadIdx.x] = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dst);
    // }

    // __syncthreads();

    float rfscore = 0.0f;

    const uint mloc0 = ((qryndx * maxnsteps0 + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
    const uint mloc2 = ((qryndx * maxnstepsmem2 + 0) * nTWorkingMemory2VarsAssignment) * ndbCstrs;

    if(dbstrndx < ndbCstrs) {
        //read/write score:
        // if(GetMoleculeType(dbstrtpf[lXDIM]) == GetMoleculeType(dbstrtpf[threadIdx.x]))
        rfscore = wrkmemaux[mloc0 + tawmvScore * ndbCstrs + dbstrndx];
        wrkmem2[mloc2 + twm2aChnScore * ndbCstrs + dbstrndx] = rfscore;
        wrkmem2[mloc2 + twm2aCpxScore * ndbCstrs + dbstrndx] = 0.0f;
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
