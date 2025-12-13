/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpw_complex_com_cuh__
#define __dpw_complex_com_cuh__

#include "libutil/macros.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/custages/fields.cuh"

// -------------------------------------------------------------------------
// GetComplexNoFromChainNo: get the number of the complex containing a given
// chain;
// xdim, template parameter, thread block's x-dimension;
// nwrpsdim, template parameter, #warps per thread block's x-dimension;
// fquery, template parameter, flag of whether strndx and ncomplexes 
// represent queries;
// strndx, structure serial number;
// ncomplexes, total number of complexes in the chunk;
// tmpSMbuf, temporary buffer of size nwrpsdim+1;
// 
template<int xdim, int nwrpsdim, int fquery>
__device__ __forceinline__
int GetComplexNoFromChainNo(
    const int strndx,
    const int ncomplexes,
    float* __restrict__ tmpSMbuf)
{
    //find minimum complex number satisfying the chain inclusion condition:
    int cpxnum = 999999;//complex number

    //get min complex index for each thread
    for(int cpx = threadIdx.x; cpx < ncomplexes && 999990 < cpxnum; cpx += xdim) {
        int chns2cpx;//#chains up to complex cpx
        if(fquery) chns2cpx = (int)GetQueryStrField<LNTYPE,pcx2DDstN>(cpx);
        else chns2cpx = (int)GetDbStrField<LNTYPE,pcx2DDstN>(cpx);
        if(strndx == chns2cpx) cpxnum = cpx;
        else if(strndx < chns2cpx) cpxnum = cpx - 1;
    }

    //per-warp-reduce for min (for tid:tid%32==0):
    cpxnum = mywarpreducemin(cpxnum);
    if((threadIdx.x & 31) == 0) tmpSMbuf[threadIdx.x>>llog2warpsize] = cpxnum;
    __syncthreads();

    //NOTE: assuming nwrpsdim<=32!
    //NOTE: (1024 is currently max for the thread block; 
    //NOTE: works otherwise but inconsistently)
    if(threadIdx.x < nwrpsdim) cpxnum = tmpSMbuf[threadIdx.x];
    cpxnum = mywarpreducemin(cpxnum);

    cpxnum = myhdmin(cpxnum, ncomplexes - 1);

    if(threadIdx.x == 0) tmpSMbuf[nwrpsdim] = cpxnum;
    __syncthreads();
    cpxnum = tmpSMbuf[nwrpsdim];
    //NOTE: no sync as long as tmpSMbuf[nwrpsdim] is used only here!

    return cpxnum;
}

// -------------------------------------------------------------------------
// ChainsWithinAssignment: get the flag of whether given query and reference
// structures represent assigned chains of the complexes they belong to;
// xdim, ydim, template parameters, thread block's x,y-dimensions;
// qrycpxndx, dbcpxndx, query and reference complex ids;
// qryndx, dbstrndx, query and reference structure (chain) global ids;
// ndbCstrs, #reference chunks in the chunk;
// maxnstepsmem2, #slots reserved for each query chain;
// stepnumber, step number corresponding to the slot to read from;
// wrkmem2, working memory containing assignment;
// tmpSMbuf, temporary buffer of size >=4;
// 
template<int xdim, int ydim>
__device__ __forceinline__
int ChainsWithinAssignment(
    const uint qrycpxndx, const uint dbcpxndx,
    const uint qryndx, const uint dbstrndx,
    const uint ndbCstrs,
    const uint maxnstepsmem2,
    const uint stepnumber,
    const float* __restrict__ wrkmem2,
    //NOTE: do not use restrict below due to compiler differences!
    float* tmpSMbuf)
{
    int qrycpxN, dbcpxN;//#chains
    uint qrycpxdstN, dbcpxdstN;//#chains up to these complexes
    int matched = 0;

    if(threadIdx.x == 0 && threadIdx.y == 0)
        tmpSMbuf[0] = GetQueryStrField<INTYPE,pcx2DN>(qrycpxndx);

    if((32 < xdim && 1 < ydim && threadIdx.x == 32 && threadIdx.y == 0) ||
    (xdim < 33 && 1 < ydim && threadIdx.x == 0 && threadIdx.y == 0) ||
    (ydim < 2 && threadIdx.x == 0 && threadIdx.y == 0))
        tmpSMbuf[1] = GetQueryStrField<LNTYPE,pcx2DDstN>(qrycpxndx);

    if((1 < ydim && threadIdx.x == 0 && threadIdx.y == 1) ||
    (32 < xdim && ydim < 2 && threadIdx.x == 32 && threadIdx.y == 0) ||
    (xdim < 33 && ydim < 2 && threadIdx.x == 0 && threadIdx.y == 0))
        tmpSMbuf[2] = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);

    if((32 < xdim && 1 < ydim && threadIdx.x == 32 && threadIdx.y == 1) ||
    (32 < xdim && ydim < 2 && threadIdx.x == 32 && threadIdx.y == 0) ||
    (xdim < 33 && 1 < ydim && threadIdx.x == 0 && threadIdx.y == 1) ||
    (xdim < 33 && ydim < 2 && threadIdx.x == 0 && threadIdx.y == 0))
        tmpSMbuf[3] = GetDbStrField<LNTYPE,pcx2DDstN>(dbcpxndx);

    __syncthreads();

    if(threadIdx.y == 0) {//this saves #registers
        qrycpxN = tmpSMbuf[0]; qrycpxdstN = tmpSMbuf[1];//float->(u)int
        dbcpxN = tmpSMbuf[2]; dbcpxdstN = tmpSMbuf[3];//float->(u)int
    }

    __syncthreads();

    if(threadIdx.y == 0) {
        //numbers of query and reference chains:
        int Q = qrycpxN, R = dbcpxN, swapped = 0;
        const int qq = (int)(qryndx - qrycpxdstN);//query chain sn within the complex
        const int rr = (int)(dbstrndx - dbcpxdstN);//reference chain sn within the complex

        if(R < Q) { myhdswap(Q, R); swapped = 1; }

        //NOTE: 1st warp calculates only:
        for(int nc = 0; nc < Q && !matched; nc += lWarpsize) {
            //reference complex's 1st chain index + running index:
            const int q = nc + (int)threadIdx.x;
            const uint locdbstrndx = dbcpxdstN + (uint)q;
            const uint mloct = ((qrycpxndx * maxnstepsmem2 + stepnumber) * nTWorkingMemory2VarsAssignment) * ndbCstrs;

            int r = 9999999;

            if(threadIdx.x < lWarpsize)
                r = fabsf(wrkmem2[mloct + twm2aCCAssignment * ndbCstrs + locdbstrndx]);//int<-float

            matched = (swapped? (qq == r && rr == q): (qq == q && rr == r));

            //NOTE: sync reduction only at y==0;
            matched = mywarpreducesum(matched);
            matched = __shfl_sync(0xffffffff, matched, 0/*srcLane*/);
        }
    }

    if(32 < xdim || 1 < ydim) {
        if(threadIdx.y == 0 && threadIdx.x == 0) tmpSMbuf[0] = matched;
        __syncthreads();
        matched = tmpSMbuf[0];
        //NOTE: no sync as long as tmpSMbuf[0] is not overwritten!
    }

    return matched;
}

// -------------------------------------------------------------------------

#endif//__dpw_complex_com_cuh__
