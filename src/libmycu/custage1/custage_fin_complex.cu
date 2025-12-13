/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <string>
#include <vector>

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/dputils.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/culayout/CuDeviceMemory.cuh"

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages2/covariance_fin_dp_refn_complete.cuh"
#include "libmycu/custages2/covariance_production_dp_refn_complete.cuh"
#include "libmycu/custages2/production_csscores.cuh"
#include "libmycu/custages2/production_2tmscore.cuh"

#include "libmycu/cudp/dpssw_tfm_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/cudp/constrained_btck2match.cuh"
#include "libmycu/cudp/production_match2aln.cuh"
#include "libmycu/cudp/reformatmatch.cuh"

#include "libmycu/cuassign/chainassign.cuh"

#include "libmycu/custage1/custage1.cuh"
#include "custage_fin_complex.cuh"


// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stagefin_complex: final stage for the refinement of the best 
// complex alignments obtained and output data production;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::run_stagefin_complex(
    cudaStream_t streamproc,
    const float d2equiv,
    const float /* scorethld */,
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
    float* __restrict__ /* wrkmem */,
    float* __restrict__ /* wrkmemccd */,
    float* __restrict__ /* wrkmemtm */,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem,
    char* __restrict__ alnsmem,
    uint* __restrict__ /*globvarsbuf*/)
{
    MYMSG("stagefin_complex::run_stagefin_complex", 4);
    static std::string preamb = "stagefin_complex::run_stagefin_complex: ";

    //produce alignment to refine superposition on:
    stagefin_align_unconstrained_complex(
        streamproc, maxnsteps,
        nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        maxnqrychains, maxnrfnchains,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer,
        maxscoordsbuf, btckdata, wrkmemaux, wrkmem2, tfmmem);

    //refine superposition at the finest scale
    stagefin_refine_complex(
        streamproc, maxnsteps,  minfraglen,
        nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qycpx1len, dbcpx1len, qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpdiagbuffers, tmpdpalnpossbuffer,
        wrkmemtmibest, wrkmemaux, tfmmem);

    //produce final alignment of matched positions:
    stagefin_align_constrained_complex(
        streamproc, maxnsteps,
        nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer,
        maxscoordsbuf, btckdata, wrkmemaux, wrkmem2, tfmmem);

    //produce full alignments for output: 
    stagefin_produce_output_alignment_complex(
        d2equiv,
        streamproc, maxnsteps,
        nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpalnpossbuffer, wrkmemaux, wrkmem2, alndatamem, alnsmem);

    //refine using production thresholds and calculate final scores for output
    stagefin_produce_output_scores_complex(
        streamproc, maxnsteps,  minfraglen,
        nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qycpx1len, dbcpx1len, qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpdiagbuffers, tmpdpalnpossbuffer, wrkmemtmibest, 
        wrkmemaux, alndatamem, tfmmem);


    stagefin_produce_CS_scores_complex(
        streamproc, maxnsteps, nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpalnpossbuffer, wrkmemaux, wrkmem2, tfmmem, alndatamem);


    stagefin_produce_2TMscores_complex(
        streamproc, maxnsteps, nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpalnpossbuffer, wrkmemaux, wrkmem2, tfmmem, alndatamem);

    //revert transformation matrices if needed
    stagefin_adjust_tfms_complex(
        streamproc, nqycpxs, ndbCcpxs, nqystrs, ndbCstrs, nqyposs, ndbCposs,
        wrkmemaux, tfmmem);
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagefin_align_complex: obtain alignment based on the best superposition;
// constrainedbtck, flag of using constrained backtracking;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::stagefin_align_complex(
    const bool constrainedbtck,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqycpxs, const uint ndbCcpxs,
    const uint nqystrs, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint ndbCposs,
    const uint qystr1len, const uint dbstr1len,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    uint* __restrict__ /*maxscoordsbuf*/,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    MYMSG("stagefin_complex::stagefin_align_complex", 5);
    static std::string preamb = "stagefin_complex::stagefin_align_complex: ";
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

    dim3 nthrds_const_mtch(CUDP_CONST_MATCH_DIM_X,CUDP_CONST_MATCH_DIM_Y,1);
    dim3 nblcks_const_mtch(ndbCstrs,nqystrs,1);

    //execution configuration for reformating match for complexes:
    dim3 nthrds_rfmtmtch(CUDP_REFORMAT_MATCHED_DIM_X,CUDP_REFORMAT_MATCHED_DIM_Y,1);
    dim3 nblcks_rfmtmtch(ndbCcpxs,nqycpxs,1);


    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPTFMSSwBtck3264x
            <true/*GLOBTFM*/,true/*GAP0*/,false/*USESS*/,D02IND_SEARCH,true/*COOMPLEX*/>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber(unused)*/,
                0.0f/* sswgt */, 0.0f/* gcost */,
                tfmmem, wrkmemaux, tmpdpdiagbuffers, tmpdpbotbuffer, btckdata,
                wrkmem2,
                maxnstepsmem2/*assgmaxnsteps*/,
                (0)/*assgstepnumber*/);
                //NOTE: alignment of assigned chains(!) using complex-level tfms
                //NOTE: when assgmaxnsteps is provided!
        MYCUDACHECKLAST;
    }

    //produce alignment for superposition
    if(constrainedbtck)
        ConstrainedBtckToMatched32x<true/*COMPLEX*/>
            <<<nblcks_const_mtch,nthrds_const_mtch,0,streamproc>>>(
                ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                btckdata, tfmmem, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
                sfin_cpx_complexstepnumber,
                sfin_cpx_complexstepnumber2,
                maxnstepsmem2/*assgmaxnsteps*/,
                (0)/*assgstepnumber*/);
    else
        BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/,true/*COMPLEX*/>
            <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                btckdata, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
                sfin_cpx_complexstepnumber,
                maxnstepsmem2/*assgmaxnsteps*/,
                (0)/*assgstepnumber*/);

    MYCUDACHECKLAST;

    //reformat match;
    ReformatMatched4Complexes<<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/,
            sfin_cpx_complexstepnumber,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefin_align_unconstrained_complex: obtain alignment based on the best
// superposition and unconstrained by distances and chain assignment;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::stagefin_align_unconstrained_complex(
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
    MYMSG("stagefin_complex::stagefin_align_unconstrained_complex", 5);
    // static std::string preamb = "stagefin_complex::stagefin_align_unconstrained_complex: ";
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
            <true/*GLOBTFM*/,true/*GAP0*/,false/*USESS*/,D02IND_SEARCH,true/*COOMPLEX*/>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber(unused)*/,
                0.0f/* sswgt */, 0.0f/* gcost */,
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
            sfin_cpx_complexstepnumber,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/);
    MYCUDACHECKLAST;

    //reformat match;
    ReformatMatched4Complexes<<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/,
            sfin_cpx_complexstepnumber,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefin_align_constrained_complex: obtain alignment based on the best
// superposition constrained by distance thresholds;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::stagefin_align_constrained_complex(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqycpxs, const uint ndbCcpxs,
    const uint nqystrs, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint ndbCposs,
    const uint qystr1len, const uint dbstr1len,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    uint* __restrict__ /*maxscoordsbuf*/,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    MYMSG("stagefin_complex::stagefin_align_constrained_complex", 5);
    // static std::string preamb = "stagefin_complex::stagefin_align_constrained_complex: ";
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
    dim3 nthrds_const_mtch(CUDP_CONST_MATCH_DIM_X,CUDP_CONST_MATCH_DIM_Y,1);
    dim3 nblcks_const_mtch(ndbCstrs,nqystrs,1);

    //execution configuration for reformating match for complexes:
    dim3 nthrds_rfmtmtch(CUDP_REFORMAT_MATCHED_DIM_X,CUDP_REFORMAT_MATCHED_DIM_Y,1);
    dim3 nblcks_rfmtmtch(ndbCcpxs,nqycpxs,1);


    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPTFMSSwBtck3264x
            <true/*GLOBTFM*/,true/*GAP0*/,false/*USESS*/,D02IND_SEARCH,true/*COOMPLEX*/>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber(unused)*/,
                0.0f/* sswgt */, 0.0f/* gcost */,
                tfmmem, wrkmemaux, tmpdpdiagbuffers, tmpdpbotbuffer, btckdata,
                wrkmem2,
                maxnstepsmem2/*assgmaxnsteps*/,
                (0)/*assgstepnumber*/);
                //NOTE: alignment of assigned chains(!) using complex-level tfms
                //NOTE: when assgmaxnsteps is provided!
        MYCUDACHECKLAST;
    }

    //produce alignment for superposition
    ConstrainedBtckToMatched32x<true/*COMPLEX*/>
        <<<nblcks_const_mtch,nthrds_const_mtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            btckdata, tfmmem, wrkmemaux, tmpdpalnpossbuffer,  wrkmem2,
            sfin_cpx_complexstepnumber,
            sfin_cpx_complexstepnumber2,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/);
    MYCUDACHECKLAST;

    //reformat match;
    ReformatMatched4Complexes<<<nblcks_rfmtmtch,nthrds_rfmtmtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/,
            sfin_cpx_complexstepnumber,  wrkmem2, wrkmemaux, tmpdpalnpossbuffer,
            1/*final*/);
    MYCUDACHECKLAST;
}



// -------------------------------------------------------------------------
// stagefin_refine_complex: refine for final superposition-best complex 
// alignments; 
// complete version;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::stagefin_refine_complex(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    const uint nqycpxs, const uint ndbCcpxs,
    const uint /* nqystrs */, const uint ndbCstrs,
    const uint /* nqyposs */, const uint ndbCposs,
    const uint qycpx1len, const uint dbcpx1len,
    const uint /*qystr1len*/, const uint /*dbstr1len*/,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    MYMSG("stagefin_complex::stagefin_refine_complex", 5);
    static std::string preamb = "stagefin_complex::stagefin_refine_complex: ";

    // const float scorethld = CLOptions::GetO_S();
    const int symmetric = CLOptions::GetC_SYMMETRIC();
    const int refinement = CLOptions::GetC_REFINEMENT();
    const int depth = CLOptions::GetC_DEPTH();

    //minimum length among largest
    int minlenmax = myhdmin(qycpx1len, dbcpx1len);
    int maxalnmax = minlenmax;//maximum alignment length

    //maximum number of fragment subdivisions
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    constexpr int sfragstep = FRAGREF_SFRAGSTEP;

    uint nlocsteps = 0;
    nlocsteps = (uint)myhdmax(1, GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen));
    nlocsteps *= (uint)nmaxsubfrags;//total number across all fragment lengths

    // if(nlocsteps < 1 || maxnsteps < nlocsteps)
    //     throw MYRUNTIME_ERROR(preamb +
    //     "Invalid number of superposition tests: "+std::to_string(nlocsteps));

    //NOTE: minimum of the largest structures to compare is assumed >=3;
    //step for the SECOND phase to final (finer-scale) refinement;
    constexpr int sfragstepmini = FRAGREF_SFRAGSTEP_mini;
    //max #fragment position factors around an identified position
    //**********************************************************************
    //NOTE: multiply maxalnmax by 2 since sub-optimal (first-phase) alignment
    //NOTE: position can be identified, e.g., at the end of alignment!
    //**********************************************************************
    uint maxnfragfcts = myhdmin(2 * maxalnmax, CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS);
    maxnfragfcts = (maxnfragfcts + sfragstepmini-1) / sfragstepmini;
    uint nlocsteps2 = maxnfragfcts;//total number for ONE fragment length
    if(refinement == CLOptions::csrFullASearch) nlocsteps2 *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps2 < 1 || maxnsteps < nlocsteps2)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps2));


    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, myhdmin(myhdmax(nlocsteps, nlocsteps2), maxnsteps));

    //initialize memory for best scores only;
    // if(0.0f < scorethld) {
        InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
        MYCUDACHECKLAST;
    // }


    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_arcmpl(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    dim3 nblcks_arcmpl(ndbCcpxs, myhdmin(nlocsteps, maxnsteps), nqycpxs);

    //refine alignment boundaries to improve scores
    if(symmetric)
        FinalFragmentBasedDPAlignmentRefinementPhase1
            <false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/,true/*COMPLEX*/>
            <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    else
        FinalFragmentBasedDPAlignmentRefinementPhase1
            <false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/,true/*COMPLEX*/>
            <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    MYCUDACHECKLAST;

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCcpxs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqycpxs, 1);

    SaveBestScoreAndTMAmongBests<true/*WRITEFRAGINFO*/,tawmvGrandBest,true/*FORCEWRITEFRAGINFO*/>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCcpxs,  maxnsteps, myhdmin(nlocsteps, maxnsteps),
            wrkmemtmibest, tfmmem, wrkmemaux, NULL/*wrkmemtmibest2nd*/,
            ndbCstrs);
    MYCUDACHECKLAST;

    if(depth == CLOptions::csdShallow || refinement == CLOptions::csrCoarseSearch)
        return;


    //second phase to production (finer-scale) refinement;
    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    nthrds_arcmpl = dim3(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    nblcks_arcmpl = dim3(ndbCcpxs, nlocsteps2, nqycpxs);

    if(refinement == CLOptions::csrFullASearch) {
        if(symmetric)
            FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearch
                <false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearch
                <false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    }
    else {
        if(symmetric)
            FinalFragmentBasedDPAlignmentRefinementPhase2
                <false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FinalFragmentBasedDPAlignmentRefinementPhase2
                <false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    }
    MYCUDACHECKLAST;

    //repeat finding the maximum among scores calculated for each fragment factor:
    SaveBestScoreAndTMAmongBests<false/*WRITEFRAGINFO*/>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCcpxs,  maxnsteps, nlocsteps2,
            wrkmemtmibest, tfmmem, wrkmemaux, NULL/*wrkmemtmibest2nd*/,
            ndbCstrs);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefin_produce_output_alignment_complex: produce full complex 
// alignments for output; 
// complete version;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::stagefin_produce_output_alignment_complex(
    const float d2equiv,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint /*nqycpxs*/, const uint /*ndbCcpxs*/,
    const uint nqystrs, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint ndbCposs,
    const uint /*qystr1len*/, const uint /*dbstr1len*/,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ alndatamem,
    char* __restrict__ alnsmem)
{
    MYMSG("stagefin_complex::stagefin_produce_output_alignment_complex", 5);
    static std::string preamb = "stagefin_complex::stagefin_produce_output_alignment_complex: ";
    static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //execution configuration for producing alignment from matched positions:
    dim3 nthrds_mtch2aln(CUDP_PRODUCTION_ALN_DIM_X,CUDP_PRODUCTION_ALN_DIM_Y,1);
    dim3 nblcks_mtch2aln(ndbCstrs,nqystrs,1);

    ProductionMatchToAlignment32x<true/*COMPLEX*/>
        <<<nblcks_mtch2aln,nthrds_mtch2aln,0,streamproc>>>(
            nodeletions, d2equiv, ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            tmpdpalnpossbuffer, wrkmemaux, alndatamem, alnsmem, wrkmem2,
            sfin_cpx_complexstepnumber,
            sfin_cpx_complexstepnumber2,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefin_produce_output_scores_complex: refine alignment 
// superpositions for complexes using production thresholds and 
// calculate final scores for output; complete version;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::stagefin_produce_output_scores_complex(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    const uint nqycpxs, const uint ndbCcpxs,
    const uint /* nqystrs */, const uint ndbCstrs,
    const uint /* nqyposs */, const uint ndbCposs,
    const uint qycpx1len, const uint dbcpx1len,
    const uint /*qystr1len*/, const uint /*dbstr1len*/,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem)
{
    MYMSG("stagefin_complex::stagefin_produce_output_scores_complex", 5);
    static std::string preamb = "stagefin_complex::stagefin_produce_output_scores_complex: ";

    const int symmetric = CLOptions::GetC_SYMMETRIC();
    const int refinement = CLOptions::GetC_REFINEMENT();
    //minimum length among largest
    int minlenmax = myhdmin(qycpx1len, dbcpx1len);
    int maxalnmax = minlenmax;//maximum alignment length

    //maximum number of fragments subdivisions
    const uint nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    int sfragstep = FRAGREF_SFRAGSTEP;

    uint nlocsteps = 0;
    nlocsteps = (uint)myhdmax(1, GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen));
    nlocsteps *= nmaxsubfrags;//total number across all fragment lengths

    // if(nlocsteps < 1 || maxnsteps < nlocsteps)
    //     throw MYRUNTIME_ERROR(preamb +
    //     "Invalid number of superposition tests: "+std::to_string(nlocsteps));

    //NOTE: minimum of the largest structures to compare is assumed >=3;
    //step for the SECOND phase to production (finer-scale) refinement;
    constexpr int sfragstepmini = FRAGREF_SFRAGSTEP_mini;
    //max #fragment position factors around an identified position
    //**********************************************************************
    //NOTE: multiply maxalnmax by 2 since sub-optimal (first-phase) alignment
    //NOTE: position can be identified at the end of alignment!
    //**********************************************************************
    uint maxnfragfcts = myhdmin(2 * maxalnmax, CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS);
    maxnfragfcts = (maxnfragfcts + sfragstepmini-1) / sfragstepmini;
    uint nlocsteps2 = 1;
    if(refinement >= CLOptions::csrFullSearch) nlocsteps2 = maxnfragfcts * nmaxsubfrags;//total number across all fragment lengths
    if(refinement == CLOptions::csrOneSearch ||
       refinement == CLOptions::csrCoarseSearch) nlocsteps2 = maxnfragfcts;//total number for ONE fragment length
    if(refinement == CLOptions::csrLogSearch) nlocsteps2 = 1;//log search inline

    if(nlocsteps2 < 1 || maxnsteps < nlocsteps2)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps2));

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqycpxs, myhdmin(myhdmax(nlocsteps, nlocsteps2), maxnsteps));

    //initialize memory for best scores only;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;



    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_arcmpl(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    dim3 nblcks_arcmpl(ndbCcpxs, myhdmin(nlocsteps, maxnsteps), nqycpxs);

    if(symmetric)
        ProductionFragmentBasedDPAlignmentRefinementPhase1<true/*TFM_DINV*/,true/*COMPLEX*/>
            <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest,
                wrkmemaux, alndatamem);
    else
        ProductionFragmentBasedDPAlignmentRefinementPhase1<false/*TFM_DINV*/,true/*COMPLEX*/>
            <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                nmaxsubfrags, maxnsteps, nlocsteps, sfragstep,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest,
                wrkmemaux, alndatamem);
    MYCUDACHECKLAST;

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCcpxs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqycpxs, 1);

    ProductionSaveBestScoresAndTMAmongBests
        <true/*WRITEFRAGINFO*/, false/*CONDITIONAL*/, true/*COMPLEX*/>
            <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
                ndbCcpxs,  maxnsteps, myhdmin(nlocsteps, maxnsteps),
                wrkmemtmibest, wrkmemaux, alndatamem, tfmmem,
                ndbCstrs);
    MYCUDACHECKLAST;



    //second phase to production (finer-scale) refinement;
    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    nthrds_arcmpl = dim3(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    nblcks_arcmpl = dim3(ndbCcpxs, nlocsteps2, nqycpxs);

    if(refinement == CLOptions::csrLogSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch<true/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, alndatamem, tfmmem);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch<false/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, alndatamem, tfmmem);
        MYCUDACHECKLAST;
        return;
    }

    if(refinement == CLOptions::csrOneSearch || refinement == CLOptions::csrCoarseSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2<true/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2<false/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    }
    else if(refinement >= CLOptions::csrFullSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch<true/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch<false/*TFM_DINV*/,true/*COMPLEX*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    }
    MYCUDACHECKLAST;

    //repeat finding the maximum among scores calculated for each fragment factor:
    ProductionSaveBestScoresAndTMAmongBests
        <false/*WRITEFRAGINFO*/, true/*CONDITIONAL*/, true/*COMPLEX*/>
            <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
                ndbCcpxs,  maxnsteps, nlocsteps2,
                wrkmemtmibest, wrkmemaux, alndatamem, tfmmem,
                ndbCstrs);
    MYCUDACHECKLAST;
}



// -------------------------------------------------------------------------
// stagefin_produce_CS_TMscores_complex: calculate chain-specific 
// TMscores and RMSDs given same complex-scale alignments and superpositions;
// complete version;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::stagefin_produce_CS_scores_complex(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint ndbCposs,
    const uint /*qystr1len*/, const uint /*dbstr1len*/,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem2,
    const float* __restrict__ tfmmem,
    float* __restrict__ alndatamem)
{
    MYMSG("stagefin_complex::stagefin_produce_CS_scores_complex", 5);
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //execution configuration for calculating scores:
    dim3 nthrds_csscores(CUDP_PRODUCTION_CS_SCORES_DIM_X,1,1);
    dim3 nblcks_csscores(ndbCstrs,nqystrs,1);

    //calculate chain-specific 2TM-scores (same alignments)
    ProductionCSScores
        <<<nblcks_csscores,nthrds_csscores,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            tmpdpalnpossbuffer, wrkmemaux, tfmmem, alndatamem, wrkmem2,
            sfin_cpx_complexstepnumber,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/);
    MYCUDACHECKLAST;
}



// -------------------------------------------------------------------------
// stagefin_produce_2TMscores_complex: calculate secondary TMscores,
// 2TMscores; complete version;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin_complex::stagefin_produce_2TMscores_complex(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint ndbCposs,
    const uint /*qystr1len*/, const uint /*dbstr1len*/,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem2,
    const float* __restrict__ tfmmem,
    float* __restrict__ alndatamem)
{
    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    if(bsectmscore == 0) return;

    MYMSG("stagefin_complex::stagefin_produce_2TMscores_complex", 5);
    const uint maxnstepsmem2 = CuMemoryBase::GetMaxNFragStepsMem2();

    //execution configuration for calculating 2tmscores:
    dim3 nthrds_2tmscos(CUDP_PRODUCTION_2TMSCORE_DIM_X,1,1);
    dim3 nblcks_2tmscos(ndbCstrs,nqystrs,1);

    //calculate 2TM-scores for complexes
    Production2TMscores<ctpvCOMPLEX>
        <<<nblcks_2tmscos,nthrds_2tmscos,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            tmpdpalnpossbuffer, wrkmemaux, tfmmem, alndatamem, wrkmem2,
            sfin_cpx_complexstepnumber,
            sfin_cpx_complexstepnumber2,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/);
    MYCUDACHECKLAST;

    //calculate chain-specific 2TM-scores (same alignments)
    Production2TMscores<ctpvCOMPLEX_CS>
        <<<nblcks_2tmscos,nthrds_2tmscos,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps,
            tmpdpalnpossbuffer, wrkmemaux, tfmmem, alndatamem, wrkmem2,
            sfin_cpx_complexstepnumber,
            sfin_cpx_complexstepnumber2,
            maxnstepsmem2/*assgmaxnsteps*/,
            (0)/*assgstepnumber*/);
    MYCUDACHECKLAST;
}



// -------------------------------------------------------------------------
// stagefin_adjust_tfms_complex: revert transformation matrices if needed
//
void stagefin_complex::stagefin_adjust_tfms_complex(
    cudaStream_t streamproc,
    const uint nqycpxs, const uint ndbCcpxs,
    const uint /*nqystrs*/, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint /*ndbCposs*/,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    // static std::string preamb = "stagefin::stagefin_adjust_tfms: ";
    static const int referenced = CLOptions::GetO_REFERENCED();
    if(referenced == 0) return;

    MYMSG("stagefin_complex::stagefin_adjust_tfms_complex", 5);
 
    dim3 nthrds_tfminit(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    dim3 nblcks_tfminit(
        (ndbCcpxs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
        nqycpxs, 1);

    //revert transformation matrices:
    RevertTfmMatrices<<<nblcks_tfminit,nthrds_tfminit,0,streamproc>>>(
        ndbCcpxs, wrkmemaux, tfmmem, ndbCstrs);
    MYCUDACHECKLAST;
 }
