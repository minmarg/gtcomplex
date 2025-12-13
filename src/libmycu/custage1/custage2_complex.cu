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
#include "custage2_complex.cuh"


// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stage2_complex: stage-2 search for superposition between multiple
// complexes simultaneoulsy and alignment identification by DP using
// secondary structure information;
// GAP0, template parameter, flag of gap open cost 0;
// USESS, template parameter, flag of using secondary structure scoring;
// D02IND, template parameter, index of how the d0 distance threshold has to be computed;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool GAP0, bool USESS, int D02IND>
void stage2_complex::run_stage2_complex(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const bool check_for_low_scores,
    const float scorethld,
    const float prescore,
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
    //optimize alignment using ss information and best superposition:
    stage2_dpss_align_complex<GAP0,USESS,D02IND>(
        streamproc,
        maxnsteps,
        nqycpxs, ndbCcpxs,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        maxnqrychains, maxnrfnchains,
        tmpdpdiagbuffers,
        tmpdpbotbuffer,
        tmpdpalnpossbuffer,
        maxscoordsbuf,
        btckdata,
        wrkmemaux,
        wrkmem2,
        tfmmem);

    //refine alignment boundaries to improve scores
    stage1_complex::stage1_refinefrag_complex<false/* CONDITIONAL */>(
        stgraphs,
        stg1REFINE_INITIAL_DP/*fragments identified by DP*/,
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

    //refine alignment boundaries identified in the previous
    //substage by applying DP;
    //1. With a gap cost:
    stage1_dprefine_complex<false/*GAP0*/,false/*PRESCREEN*/>(
        stgraphs,
        streamproc,
        2/* maxndpiters */,
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

    //execution configuration for checking scores:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_cpxinit0(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_cpxinit0(
        (ndbCcpxs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, 1);

    if(check_for_low_scores && 0.0f < scorethld) {
        SetLowScoreConvergenceFlag<true/*COMPLEX*/>
            <<<nblcks_cpxinit0,nthrds_cpxinit0,0,streamproc>>>(
                scorethld, ndbCcpxs, maxnsteps, wrkmemaux, ndbCstrs);
        MYCUDACHECKLAST;
    }
}

// Instantiations
// 
#define INSTANTIATE_stage2_complex__run_stage2_complex(tpGAP0,tpUSESS,tpD02IND) \
    template void stage2_complex::run_stage2_complex<tpGAP0,tpUSESS,tpD02IND>( \
    std::map<CGKey,MyCuGraph>& stgraphs, \
    cudaStream_t streamproc, \
    const bool check_for_low_scores, \
    const float scorethld, \
    const float prescore, \
    const int maxndpiters, \
    const uint maxnsteps, \
    const uint minfraglen, \
    uint nqycpxs, uint ndbCcpxs, \
    uint nqystrs, uint ndbCstrs, \
    uint nqyposs, uint ndbCposs, \
    uint qycpx1len, uint dbcpx1len, \
    uint qystr1len, uint dbstr1len, \
    uint qystrnlen, uint dbstrnlen, \
    uint dbxpad, \
    const uint maxnqrychains, const uint maxnrfnchains, \
    float* __restrict__ /*scores*/,  \
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
    float* __restrict__ tfmmem, \
    uint* __restrict__ /*globvarsbuf*/);

INSTANTIATE_stage2_complex__run_stage2_complex(false,true,D02IND_DPSCAN);
INSTANTIATE_stage2_complex__run_stage2_complex(true,false,D02IND_SEARCH);



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage2_dpss_align_complex: obtain complex alignment based on the
// secondary structure information and best superposition so far;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool GAP0, bool USESS, int D02IND>
void stage2_complex::stage2_dpss_align_complex(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqycpxs, const uint ndbCcpxs,
    const uint nqystrs, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint ndbCposs,
    const uint qystr1len, const uint dbstr1len,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    const uint maxnqrychains, const uint maxnrfnchains,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    uint* __restrict__ /*maxscoordsbuf*/,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    MYMSG("stage2_complex::stage2_dpss_align_complex", 5);
    // static std::string preamb = "stage2_complex::stage2_dpss_align_complex: ";

    constexpr float gcost = {GAP0? 0.0f: -1.0f};
    static const float sswgt = 0.5f;
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //CUDP_2DCACHE_DIM_D x CUDP_2DCACHE_DIM_X;
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    //NOTE: now use block DIAGONALS, where blocks share a COMMON POINT 
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

    //execution configuration for reformating match for complexes:
    dim3 nthrds_rfmtmtch(CUDP_REFORMAT_MATCHED_DIM_X,CUDP_REFORMAT_MATCHED_DIM_Y,1);
    dim3 nblcks_rfmtmtch(ndbCcpxs,nqycpxs,1);

    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPTFMSSwBtck3264x
            <true/*GLOBTFM*/,GAP0,USESS,D02IND,true/*COOMPLEX*/>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber(unused)*/,
                sswgt, gcost,
                tfmmem, wrkmemaux, tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
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
    MakeChain2ChainAssignment<false/*WRITESCORE*/, true/*WRITEASSG*/,false/*PASS2*/>
        <<<nblcks_ch2ch,nthrds_ch2ch,szdsmem_ch2ch,streamproc>>>(
            nqystrs, ndbCstrs, ndbCcpxs,  maxnqrychains, maxnrfnchains,  maxnsteps, maxnstepsmem2,
            NULL/*tfmmemory*/, wrkmemaux, wrkmem2);
    MYCUDACHECKLAST;
    //}}

    //produce alignment for superposition
    BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/,true/*COMPLEX*/>
        <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
            btckdata, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
            stg2_cpx_complexstepnumber/*!=0*/,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/);
    MYCUDACHECKLAST;

    //reformat match;
    ReformatMatched4Complexes<<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/,
            stg2_cpx_complexstepnumber/*!=0*/,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer);
    MYCUDACHECKLAST;
}
