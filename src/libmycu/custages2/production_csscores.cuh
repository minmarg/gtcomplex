/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __production_csscores_cuh__
#define __production_csscores_cuh__

#include "libgenp/gproc/gproc.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmymp/mpstages/scoringbase.h"
#include "libmymp/mpstages/transformbase.h"
#include "libmycu/custages/fields.cuh"

// =========================================================================
// ProductionCSScores: calculate chain-specific TM-scores & RMSDs and write
// them to memory;
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
    const float* __restrict__ wrkmem2 = NULL,
    const uint complexstepnumber = 0,
    const uint assgmaxnsteps = 0,
    const uint assgstepnumber = 0
);

// -------------------------------------------------------------------------
// UpdateOneAlnPosScore_CSscores: update TMscore unconditionally for one 
// alignment position;
// d02, d0 squared used for calculating score;
// pos, position in alignment buffer tmpdpalnpossbuffer for coordinates;
// dblen, step (db length) by which coordinates of different dimension 
// written in tmpdpalnpossbuffer;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tfm, address of the transformation matrix;
// scv, address of the vector of scores;
//
__device__ __forceinline__
void UpdateOneAlnPosScore_CSscores(
    float d02, int pos, int dblen,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tfm,
    float* __restrict__ scv)
{
    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

    float dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);

    scv[threadIdx.x] += GetPairScore(d02, dst);//score
}

// -------------------------------------------------------------------------
// CalcCSscoresUnrl_Complete: calculate/reduce UNNORMALIZED TMscores for 
// obtained superpositions; complete version; 
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, max number of steps to perform for each reference structure;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d02, distance threshold;
// complexstepnumber, step number corresponding to memory section for 
// matched coordinates for CHAINS;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tfmCache, cached transformation matrix;
// scvCache, cache for scores;
//
__device__ __forceinline__
void CalcCSscoresUnrl_Complete(
    const uint qryndx,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint /*qrydst*/, const uint dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos, const float d02,
    const uint complexstepnumber,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tfmCache,
    float* __restrict__  scvCache)
{
    //initialize cache
    scvCache[threadIdx.x] = 0.0f;

    __syncthreads();

    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: coordinates:
    const int yofff = (qryndx * maxnsteps + complexstepnumber/*sfragfctxndx*/) * dblen * nTDPAlignedPoss;

    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x; qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen;
        pos0 += blockDim.x)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: aligned coordinates in tmpdpalnpossbuffer are in the reverse order!
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + dbstrlen-1 - (rfnpos + pos0);
        UpdateOneAlnPosScore_CSscores(//no sync;
            d02, dppos, dblen, tmpdpalnpossbuffer,//coordinates
            tfmCache, scvCache//score cache
        );
    }

    //sync now:
    __syncthreads();

    //unroll until reaching warpSize 
    for(int xdim = (blockDim.x>>1); xdim >= 32; xdim >>= 1) {
        if(threadIdx.x < xdim)
            scvCache[threadIdx.x] += scvCache[threadIdx.x + xdim];
        __syncthreads();
    }

    //unroll warp for the score
    if(threadIdx.x < 32/*warpSize*/) {
        float sum = scvCache[threadIdx.x];
        sum = mywarpreducesum(sum);
        //write to the first SMEM data slot
        if(threadIdx.x == 0) scvCache[0] = sum;
    }

    //make the block's all threads see the reduced score scvCache[0]:
    __syncthreads();
}

// -------------------------------------------------------------------------
// SaveBestQRCSscores_Complete: complete version of saving the best
// scores calculated for the query and reference structures
// directly to the production output memory region;
// best, best TMscore calculated for the smaller length;
// gbest, best TMscore calculated for the greater length;
// d0, g0, normalizing inter-residue distances for smaller and greater length;
// rmsd, chain-specific rmsd valid for thread 0;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// qrylenorg, dbstrlenorg, query and reference lengths;
// NOTE: memory pointers should be aligned!
// alndatamem, memory for full alignment information, including scores;
// 
__device__ __forceinline__
void SaveBestQRCSscores_Complete(
    float best, float gbest,
    float d0, float g0, float rmsd,
    const uint qryndx,
    const uint dbstrndx,
    const uint ndbCstrs,
    const int qrylenorg,
    const int dbstrlenorg,
    float* __restrict__ alndatamem)
{
    //save best scores
    if(threadIdx.x == 0) {
        uint mloc = (qryndx * ndbCstrs + dbstrndx) * nTDP2OutputAlnData;
        //make best represent the query score:
        if(dbstrlenorg < qrylenorg) {
            myhdswap(best, gbest);
            myhdswap(d0, g0);
        }
        alndatamem[mloc + dp2oadRMSD_C] = rmsd;
        alndatamem[mloc + dp2oadScoreQ_C] = __fdividef(best, qrylenorg);
        alndatamem[mloc + dp2oadScoreR_C] = __fdividef(gbest, dbstrlenorg);
        alndatamem[mloc + dp2oadD0Q_C] = d0;
        alndatamem[mloc + dp2oadD0R_C] = g0;
    }
}

// -------------------------------------------------------------------------

#endif//__production_csscores_cuh__
