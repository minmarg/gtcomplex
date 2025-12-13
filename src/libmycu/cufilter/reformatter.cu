/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "reformatter.cuh"

// -------------------------------------------------------------------------
// MakeCandidateComplexList: make list of candidate reference (database)
// complexes proceeding to stages of more detailed superposition search and
// refinement;
// nqycpxs, total number of query complexes in the chunk;
// ndbCmpxs, total number of reference complexes in the chunk;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps for reference structures;
// maxnstepsmem2, max number of steps for assignments in wrkmem2;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// wrkmem2, working memory for assignments initially containing convergence flags;
// globvarsbuf, memory of new indices and addresses of passing references;
// NOTE: thread block is 2D (y-dim=2 for indices and addresses) and
// NOTE: processes the references over all queries for flags;
// 
__global__ void MakeCandidateComplexList(
    const uint nqycpxs, const uint ndbCmpxs,
    const uint nqystrs, const uint ndbCstrs,
    const uint maxnsteps, const uint maxnstepsmem2,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    uint* __restrict__ globvarsbuf)
{
    enum {
        //index for the previous prefix sum and padding
        PREPXS = 0,
        pad = 1,
        //-------
        lCNST = 999999,//initialization constant
        lXCXF = 0,//ndx for accumulated flags for processed complexes
        lXCXN = 1,//ndx for cached complex lengths (#chains)
        //-------
        lYDIM = CUFL_MAKECANDIDATELIST_YDIM,
        lXDIM = CUFL_MAKECANDIDATELIST_XDIM
    };

    constexpr uint sfragfct = 0;//fragment factor
    __shared__ int cpxdata[lYDIM + 2][lXDIM + pad];//complex data: flags and lengths
    // __shared__ int strdata[lXDIM + pad];//accumulated convergence flags for chains
    // __shared__ int cpxcxd[lXDIM + pad];//distances to complexes
    int* strdata = cpxdata[lYDIM];//accumulated convergence flags for chains
    int* cpxcxd = cpxdata[lYDIM+1];//distances to complexes

    if(threadIdx.x == 0)
        cpxdata[threadIdx.y][PREPXS] = 0;

    //iterate over complexes
    for(uint dbcpxndx0 = 0; dbcpxndx0 < ndbCmpxs; dbcpxndx0 += blockDim.x)
    {
        //update the prefix sums originated from processing the last data block:
        if(threadIdx.x == 0 && dbcpxndx0)
            cpxdata[threadIdx.y][PREPXS] = cpxdata[threadIdx.y][pad + lXDIM - 1];

        //NOTE: cpxdata is overwritten below, sync:
        __syncthreads();

        uint dbcpxndx = dbcpxndx0 + threadIdx.x;//complex index
        if(threadIdx.y == 0 && dbcpxndx < ndbCmpxs)
            cpxdata[lXCXN][pad + threadIdx.x] = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);
        if(threadIdx.y == 1 && dbcpxndx < ndbCmpxs)
            cpxcxd[pad + threadIdx.x] = GetDbStrField<LNTYPE,pcx2DDstN>(dbcpxndx);

        //set the last index
        if(threadIdx.y == 0 &&
        (((dbcpxndx + 1 == ndbCmpxs)) || (dbcpxndx + 1 < ndbCmpxs && threadIdx.x + 1 == blockDim.x)))
            cpxcxd[0] = threadIdx.x;

        __syncthreads();

        const int tidclast = cpxcxd[0];//tid for the last complex read

        //no sync as long as cpxcxd[0] not overwritten;
        //index of the first and last chains of complexes under process:
        const int ndxs0 = cpxcxd[pad];
        const int ndxsn = cpxcxd[pad + tidclast] + cpxdata[lXCXN][pad + tidclast];

        //initialize flags
        if(threadIdx.y == 0 && threadIdx.x == 0) strdata[0] = 0;
        if(threadIdx.y == 1) cpxdata[lXCXF][pad + threadIdx.x] = lCNST;


        //{{iterate over all chains within the interval of complexes being processed
        for(uint dbstrndx0 = ndxs0; dbstrndx0 < ndxsn && dbstrndx0 < ndbCstrs; dbstrndx0 += blockDim.x)
        {
            //update the prefix sum obtained from processing last data:
            if(threadIdx.y == 0 && threadIdx.x == 0 && ndxs0 < dbstrndx0)
                strdata[0] = strdata[pad + lXDIM - 1];
            __syncthreads();

            uint dbstrndx = dbstrndx0 + threadIdx.x;//reference index
            int value = 0;//convergence flag
            //get convergence flags (over all queries)
            if(threadIdx.y == 0 && dbstrndx < ndxsn && dbstrndx < ndbCstrs) {
                for(uint qryndx = 0; qryndx < nqystrs; qryndx++) {
                    const uint mloc2 = ((qryndx * maxnstepsmem2 + 0) * nTWorkingMemory2VarsAssignment) * ndbCstrs;
                    int lconv = wrkmem2[mloc2 + twm2aTmpConvergence * ndbCstrs + dbstrndx];//float->int
                    value += ((lconv & CONVERGED_LOWTMSC_bitval) != 0);
                }
            }

            if(threadIdx.y == 0)
                //convergence flags set for all queries imply consideration for skipping the chain
                strdata[pad + threadIdx.x] = (value = (nqystrs <= value));

            __syncthreads();

            //calculate inclusive prefix sums of flags to obtain counts:
            for(uint xdim = 1; xdim < blockDim.x; xdim <<= 1) {
                if(threadIdx.y == 0 && xdim <= threadIdx.x)
                    value += strdata[pad + threadIdx.x - xdim];
                __syncthreads();
                if(threadIdx.y == 0 && xdim <= threadIdx.x)
                    strdata[pad + threadIdx.x] = value;
                __syncthreads();
            }

            //correct the prefix sum by adding the previous last value:
            if(threadIdx.y == 0) strdata[pad + threadIdx.x] += strdata[0];
            __syncthreads();

            //determine converged (all chains of) complexes within the current portion of complexes:
            if(threadIdx.y == lXCXF && dbcpxndx < ndbCmpxs) {
                int ncxd = cpxcxd[pad + threadIdx.x];//#chains up to complex dbcpxndx
                int ncxn = cpxdata[lXCXN][pad + threadIdx.x];//#complex chains
                //last chain up to and including complex dbcpxndx:
                int lastchain = ncxd + ncxn - 1;
                //#chains converged up to the 1st chain (exclusive) of complex dbcpxndx:
                int nstrconv =
                    (ncxd < dbstrndx0 + 1)
                    ? 0
                    : ((dbstrndx0 + 1 <= ncxd && ncxd < dbstrndx0 + 1 + blockDim.x)
                       ? strdata[pad + ncxd - 1 - dbstrndx0]: -1);
                //#chains converged up to and including complex dbcpxndx:
                int nstrconvi =
                    (dbstrndx0 <= lastchain && lastchain < dbstrndx0 + blockDim.x)
                    ? strdata[pad + lastchain - dbstrndx0]: -1;
                int cppval = cpxdata[lXCXF][pad + threadIdx.x];//complex flag
                //save nstrconv if unset and chain index is within the range
                if(lCNST <= cppval && 0 <= nstrconv)
                    cpxdata[lXCXF][pad + threadIdx.x] = cppval = -nstrconv;
                if(cppval <= 0 && 0 <= nstrconvi) {
                    //if #converged chains for complex dbcpxndx == #complex chains (all):
                    if(nstrconvi + cppval == ncxn) cpxdata[lXCXF][pad + threadIdx.x] = 0;
                    else cpxdata[lXCXF][pad + threadIdx.x] = 1;
                }
            }
        }//}}for1(;dbstrndx0 < ndxsn;)


        //reset flag initializers
        if(threadIdx.y == lXCXF && cpxdata[lXCXF][pad + threadIdx.x] != 1)
            cpxdata[lXCXF][pad + threadIdx.x] = 0;

        __syncthreads();


        //{{iterate over chains and unset convergence for those with corresponding complexes not converged
        for(uint dbstrndx0 = ndxs0; dbstrndx0 < ndxsn && dbstrndx0 < ndbCstrs; dbstrndx0 += blockDim.x)
        {
            uint dbstrndx = dbstrndx0 + threadIdx.x;//reference index
            int ndxcmpx = -1;//complex index not found for dbstrndx
            //binary search to find the complex index and its convergence flag:
            //(NOTE: loop executed individually for each thread; however,)
            //(NOTE: consecutive threads in a warp don't branch normally /depends on #complexes/)
            for(int l = 0, r = tidclast; threadIdx.y == 0 && ndxcmpx < 0 && l <= r;) {
                int m = (l + r) >> 1;
                int ncxd = cpxcxd[pad + m];//#chains up to complex dbcpxndx0 + m
                int ncxn = cpxdata[lXCXN][pad + m];//#complex chains
                if(ncxd + ncxn <= dbstrndx) l = m + 1;
                else if(dbstrndx < ncxd) r = m - 1;
                else if(ndxcmpx < 0) ndxcmpx = m;
            }
            //no sync: every thread works with its data;
            //reset convergence for all queries
            if(threadIdx.y == 0 && 0 <= ndxcmpx && dbstrndx < ndxsn && dbstrndx < ndbCstrs)
            {
                for(uint qrycpx = 0; qrycpx < nqycpxs; qrycpx++) {
                    uint mloc0 = ((qrycpx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
                    wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx] = //SET or RESET...
                        (cpxdata[lXCXF][pad + ndxcmpx] == 0)? (float)CONVERGED_LOWTMSC_bitval: 0.0f;
                }
            }
        }//}}for2(;dbstrndx0 < ndxsn;)


        //sync now: cpxdata[lXCXN] overwritten below
        __syncthreads();

        //reuse cpxcxd to read complex types for later reuse in this iteration
        if(threadIdx.y == 1 && dbcpxndx < ndbCmpxs)
            cpxcxd[pad + threadIdx.x] = GetDbStrField<INTYPE,pcx2DType>(dbcpxndx);

        int cpdval = cpxdata[threadIdx.y][pad + threadIdx.x];

        //set complex data to 0 where convflag is set
        if(threadIdx.y == lXCXN && cpxdata[lXCXF][pad + threadIdx.x] == 0)
            cpxdata[lXCXN][pad + threadIdx.x] = cpdval = 0;

        //save #complex chains before prefix sum for later reuse in this iteration
        if(threadIdx.y == lXCXN)
            strdata[pad + threadIdx.x] = cpxdata[lXCXN][pad + threadIdx.x];

        __syncthreads();

        //calculate inclusive prefix sums of flags to obtain new indices and
        //addresses for complexes:
        for(uint xdim = 1; xdim < blockDim.x; xdim <<= 1) {
            if(xdim <= threadIdx.x)
                cpdval += cpxdata[threadIdx.y][pad + threadIdx.x - xdim];
            __syncthreads();
            if(xdim <= threadIdx.x)
                cpxdata[threadIdx.y][pad + threadIdx.x] = cpdval;
            __syncthreads();
        }

        //correct the prefix sums by adding the previously obtained values:
        cpxdata[threadIdx.y][pad + threadIdx.x] += cpxdata[threadIdx.y][PREPXS];
        __syncthreads();

        //write to GMEM indices and addresses and update complex fields:
        if(dbcpxndx < ndbCmpxs) {
            //write starting at slot fdNewComplexIndex
            uint mloc = (threadIdx.y + fdNewComplexIndex) * ndbCstrs + dbcpxndx;
            int newndx = cpxdata[lXCXF][pad + threadIdx.x];
            int cpdvalprev = cpxdata[threadIdx.y][pad + threadIdx.x - 1];
            cpdval = cpxdata[threadIdx.y][pad + threadIdx.x];
            //set to 0 for filtered-out complexes:
            if(cpdval == cpdvalprev) cpdval = cpdvalprev = newndx = 0;
            //write adjusted addresses to gmem:
            if(threadIdx.y == lXCXN) cpdval = cpdvalprev;
            globvarsbuf[mloc] = cpdval;

            //NOTE: update complex fields on device in the same run:
            //NOTE that cpdval==0 implies non-incremented values for both 
            //cpxdata[lXCXF] and [lXCXN], and both cpdval==0 across threadIdx.y
            if(threadIdx.y == lXCXN && newndx) {
                //new index (prefix sum) is stored in smem and its cpdval != 0:
                //complex type now is stored in cpxcxd:
                //(strdata now contains #complex chains):
                int nchns = strdata[pad + threadIdx.x];
                int type = cpxcxd[pad + threadIdx.x];
                newndx--;//adjust index appropriately (NOTE: new address is already valid)
                //write complex-specific fields to new gmem location
                SetDbStrField<INTYPE,pcx2DN>(newndx, nchns);
                //cpdvalprev is new address calculated and stored in cpxdata[lXCXN][pad + threadIdx.x - 1]
                SetDbStrField<LNTYPE,pcx2DDstN>(newndx, cpdvalprev);
                SetDbStrField<INTYPE,pcx2DType>(newndx, type);
                //NOTE: write backreferences in the structure-specific fields here!
                //NOTE: the below implementation using a loop is inefficient with
                //NOTE: uncoalesced memory access; however, the duration of the entire
                //NOTE: action of a single thread block in this kernel is negligible
                //NOTE: and a more efficient implementation offers insignificant gain.
                for(int cndx = cpdvalprev; cndx < cpdvalprev + nchns; cndx++)
                    SetDbStrField<INTYPE,pps2DCpxI>(cndx, newndx);
            }
        }
        __syncthreads();

    }//for(;dbcpxndx0 < ndbCmpxs;)
}



// -------------------------------------------------------------------------
// MakeCandidateComplexList2: make list of candidate reference (database)
// complexes proceeding to stages of more detailed superposition search and
// refinement;
// This corresponds to structure-based prefiltering and is based on single
// flags set for each complex pair once (even if complexes have multiple chains);
// nqycpxs, total number of query complexes in the chunk;
// ndbCcpxs, total number of reference complexes in the chunk;
// nqystrs, total number of query structures/chains in the chunk;
// ndbCstrs, total number of reference structures/chains in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// globvarsbuf, memory of new indices and addresses of passing references;
// NOTE: thread block is 2D (y-dim=2 for indices and addresses) and
// NOTE: processes the references over all queries for flags;
// 
__global__ void MakeCandidateComplexList2(
    const uint nqycpxs, const uint ndbCcpxs,
    const uint /* nqystrs */, const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux,
    uint* __restrict__ globvarsbuf)
{
    enum {
        //index for the previous prefix sum and padding
        PREPXS = 0,
        pad = 1,
        //-------
        lXCXF = 0,//ndx for accumulated flags for processed complexes
        lXCXN = 1,//ndx for cached complex lengths (#chains)
        //-------
        lYDIM = CUFL_MAKECANDIDATELIST_YDIM,
        //NOTE: CUFL_MAKECANDIDATELIST_XDIM>MAX_QUERY_STRUCTURES_PER_CHUNK! (query complexes actually):
        lXDIM = CUFL_MAKECANDIDATELIST_XDIM
    };

    constexpr uint sfragfct = 0;//fragment factor
    __shared__ int cpxdata[lYDIM + 3][lXDIM + pad];//complex data: flags and lengths
    // __shared__ int strdata[lXDIM + pad];//accumulated convergence flags for chains
    // __shared__ int cpxcxd[lXDIM + pad];//distances to complexes
    int* strdata = cpxdata[lYDIM];//accumulated convergence flags for chains
    int* cpxcxd = cpxdata[lYDIM+1];//distances to complexes
    int* qrycxd = cpxdata[lYDIM+2];//distances to query complexes


    //get distances (in chains) to query complexes:
    for(uint qrycpx = threadIdx.x; threadIdx.y == 0 && qrycpx < nqycpxs; qrycpx += blockDim.x)
        qrycxd[pad + qrycpx] = GetQueryStrField<LNTYPE,pcx2DDstN>(qrycpx);

    __syncthreads();

    //iterate over complexes and populate convergence flags so that they do not 
    //intertwine with chain-specific flags (avoiding overwriting complex-specific flags below):
    for(int qrycpx = nqycpxs - 1; 0 <= qrycpx; qrycpx--)
    {
        for(int dbcpxndx0 = ndbCcpxs - 1; 0 <= dbcpxndx0; dbcpxndx0 -= blockDim.x)
        {
            int dbcpxndx = dbcpxndx0 - (int)threadIdx.x;//complex index

            uint mloc0 = ((qrycpx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
            //locationn of the 1st chain of query complex qrycpx:
            /// uint mloct = ((qrycxd[pad + qrycpx] * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

            //FIRST, read at most blockDim.x complex flags (reuse cache strdata):
            if(0 <= dbcpxndx && threadIdx.y == 0)
                strdata[pad + threadIdx.x] = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbcpxndx];

            __syncthreads();

            //NEXT, write complex flags:
            //NOTE: dbcpxndx <= dbstrndx always by definition;
            //NOTE: hence, this writing won't overwrite complex data when iterating backwards!
            if(0 <= dbcpxndx && threadIdx.y == 0) {
                int dbstrndx = GetDbStrField<LNTYPE,pcx2DDstN>(dbcpxndx);
                //locationn of the 1st chain of reference complex dbcpxndx:
                /// wrkmemaux[mloct + tawmvConverged * ndbCstrs + dbstrndx] = strdata[pad + threadIdx.x];
                wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx] = strdata[pad + threadIdx.x];
            }
        }
    }


    if(threadIdx.x == 0) cpxdata[threadIdx.y][PREPXS] = 0;

    //iterate over complexes
    for(uint dbcpxndx0 = 0; dbcpxndx0 < ndbCcpxs; dbcpxndx0 += blockDim.x)
    {
        //update the prefix sums originated from processing the last data block:
        if(threadIdx.x == 0 && dbcpxndx0)
            cpxdata[threadIdx.y][PREPXS] = cpxdata[threadIdx.y][pad + lXDIM - 1];

        //NOTE: cpxdata is overwritten below, sync:
        __syncthreads();

        uint dbcpxndx = dbcpxndx0 + threadIdx.x;//complex index
        if(threadIdx.y == 0 && dbcpxndx < ndbCcpxs)
            cpxdata[lXCXN][pad + threadIdx.x] = GetDbStrField<INTYPE,pcx2DN>(dbcpxndx);
        if(threadIdx.y == 1 && dbcpxndx < ndbCcpxs)
            cpxcxd[pad + threadIdx.x] = GetDbStrField<LNTYPE,pcx2DDstN>(dbcpxndx);

        //set the last index
        if(threadIdx.y == 0 &&
        (((dbcpxndx + 1 == ndbCcpxs)) || (dbcpxndx + 1 < ndbCcpxs && threadIdx.x + 1 == blockDim.x)))
            cpxcxd[0] = threadIdx.x;

        __syncthreads();

        const int tidclast = cpxcxd[0];//tid for the last complex read

        //no sync as long as cpxcxd[0] not overwritten;
        //index of the first and last chains of complexes under process:
        const int ndxs0 = cpxcxd[pad];
        const int ndxsn = cpxcxd[pad + tidclast] + cpxdata[lXCXN][pad + tidclast];


        if(threadIdx.y == 0) cpxdata[lXCXF][pad + threadIdx.x] = 0;

        //get convergence flags (over all queries)
        if(threadIdx.y == 0 && dbcpxndx < ndbCcpxs) {
            int value = 0;//convergence flag
            for(uint qrycpx = 0; qrycpx < nqycpxs; qrycpx++) {
                /// uint mloc0 = ((qrycxd[pad + qrycpx] * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
                uint mloc0 = ((qrycpx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
                int lconv = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + (cpxcxd[pad + threadIdx.x]/*dbstrndx:1st*/)];
                value += ((lconv & CONVERGED_LOWTMSC_bitval) != 0);
            }
            //converged (all chains of) complexes within the current portion of complexes:
            cpxdata[lXCXF][pad + threadIdx.x] = !(nqycpxs <= value);//NOTE: 0=>converged! (to be filtered out)
        }

        __syncthreads();


        //{{iterate over chains and unset convergence for those with corresponding complexes not converged
        for(uint dbstrndx0 = ndxs0; dbstrndx0 < ndxsn && dbstrndx0 < ndbCstrs; dbstrndx0 += blockDim.x)
        {
            uint dbstrndx = dbstrndx0 + threadIdx.x;//reference index
            int ndxcmpx = -1;//complex index not found for dbstrndx
            //binary search to find the complex index and its convergence flag:
            //(NOTE: loop executed individually for each thread; however,)
            //(NOTE: consecutive threads in a warp don't branch normally /depends on #complexes/)
            for(int l = 0, r = tidclast; threadIdx.y == 0 && ndxcmpx < 0 && l <= r;) {
                int m = (l + r) >> 1;
                int ncxd = cpxcxd[pad + m];//#chains up to complex dbcpxndx0 + m
                int ncxn = cpxdata[lXCXN][pad + m];//#complex chains
                if(ncxd + ncxn <= dbstrndx) l = m + 1;
                else if(dbstrndx < ncxd) r = m - 1;
                else if(ndxcmpx < 0) ndxcmpx = m;
            }
            //no sync: every thread works with its data;
            //reset convergence for all queries
            if(threadIdx.y == 0 && 0 <= ndxcmpx && dbstrndx < ndxsn && dbstrndx < ndbCstrs)
            {
                /// for(uint qryndx = 0; qryndx < nqystrs; qryndx++) {
                for(uint qrycpx = 0; qrycpx < nqycpxs; qrycpx++) {
                    /// uint mloc0 = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
                    uint mloc0 = ((qrycpx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
                    wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx] = //SET or RESET...
                        (cpxdata[lXCXF][pad + ndxcmpx] == 0)? (float)CONVERGED_LOWTMSC_bitval: 0.0f;
                }
            }
        }//}}for2(;dbstrndx0 < ndxsn;)


        //sync now: cpxdata[lXCXN] overwritten below
        __syncthreads();

        //reuse cpxcxd to read complex types for later reuse in this iteration
        if(threadIdx.y == 1 && dbcpxndx < ndbCcpxs)
            cpxcxd[pad + threadIdx.x] = GetDbStrField<INTYPE,pcx2DType>(dbcpxndx);

        int cpdval = cpxdata[threadIdx.y][pad + threadIdx.x];

        //set complex data to 0 where convflag is set
        if(threadIdx.y == lXCXN && cpxdata[lXCXF][pad + threadIdx.x] == 0)
            cpxdata[lXCXN][pad + threadIdx.x] = cpdval = 0;

        //save #complex chains before prefix sum for later reuse in this iteration
        if(threadIdx.y == lXCXN)
            strdata[pad + threadIdx.x] = cpxdata[lXCXN][pad + threadIdx.x];

        __syncthreads();

        //calculate inclusive prefix sums of flags to obtain new indices and
        //addresses for complexes:
        for(uint xdim = 1; xdim < blockDim.x; xdim <<= 1) {
            if(xdim <= threadIdx.x)
                cpdval += cpxdata[threadIdx.y][pad + threadIdx.x - xdim];
            __syncthreads();
            if(xdim <= threadIdx.x)
                cpxdata[threadIdx.y][pad + threadIdx.x] = cpdval;
            __syncthreads();
        }

        //correct the prefix sums by adding the previously obtained values:
        cpxdata[threadIdx.y][pad + threadIdx.x] += cpxdata[threadIdx.y][PREPXS];
        __syncthreads();

        //write to GMEM indices and addresses and update complex fields:
        if(dbcpxndx < ndbCcpxs) {
            //write starting at slot fdNewComplexIndex
            uint mloc = (threadIdx.y + fdNewComplexIndex) * ndbCstrs + dbcpxndx;
            int newndx = cpxdata[lXCXF][pad + threadIdx.x];
            int cpdvalprev = cpxdata[threadIdx.y][pad + threadIdx.x - 1];
            cpdval = cpxdata[threadIdx.y][pad + threadIdx.x];
            //set to 0 for filtered-out complexes:
            if(cpdval == cpdvalprev) cpdval = cpdvalprev = newndx = 0;
            //write adjusted addresses to gmem:
            if(threadIdx.y == lXCXN) cpdval = cpdvalprev;
            globvarsbuf[mloc] = cpdval;

            //NOTE: update complex fields on device in the same run:
            //NOTE that cpdval==0 implies non-incremented values for both 
            //cpxdata[lXCXF] and [lXCXN], and both cpdval==0 across threadIdx.y
            if(threadIdx.y == lXCXN && newndx) {
                //new index (prefix sum) is stored in smem and its cpdval != 0:
                //complex type now is stored in cpxcxd:
                //(strdata now contains #complex chains):
                int nchns = strdata[pad + threadIdx.x];
                int type = cpxcxd[pad + threadIdx.x];
                newndx--;//adjust index appropriately (NOTE: new address is already valid)
                //write complex-specific fields to new gmem location
                SetDbStrField<INTYPE,pcx2DN>(newndx, nchns);
                //cpdvalprev is new address calculated and stored in cpxdata[lXCXN][pad + threadIdx.x - 1]
                SetDbStrField<LNTYPE,pcx2DDstN>(newndx, cpdvalprev);
                SetDbStrField<INTYPE,pcx2DType>(newndx, type);
                //NOTE: write backreferences in the structure-specific fields here!
                //NOTE: the below implementation using a loop is inefficient with
                //NOTE: uncoalesced memory access; however, the duration of the entire
                //NOTE: action of a single thread block in this kernel is negligible
                //NOTE: and a more efficient implementation offers insignificant gain.
                for(int cndx = cpdvalprev; cndx < cpdvalprev + nchns; cndx++)
                    SetDbStrField<INTYPE,pps2DCpxI>(cndx, newndx);
            }
        }
        __syncthreads();

    }//for(;dbcpxndx0 < ndbCmpxs;)
}



// -------------------------------------------------------------------------
// MakeDbCandidateList: make list of reference structure (database)
// candidates proceeding to stages of more detailed superposition search and
// refinement;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// globvarsbuf, memory of new indices and addresses of passing references;
// NOTE: thread block is 2D (y-dim=2 for indices and addresses) and
// NOTE: processes the reference structures over all queries for flags;
// 
__global__ void MakeDbCandidateList(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    uint* __restrict__ globvarsbuf)
{
    enum {
        //index for the previous prefix sum and padding
        PREPXS = 0,
        pad = 1,
        lXFLG = fdNewReferenceIndex,//index for reference structure convergence flags/new indices
        lXLEN = fdNewReferenceAddress,//index for reference structure convergence lengths/new addresses
        lYDIM = CUFL_MAKECANDIDATELIST_YDIM,
        lXDIM = CUFL_MAKECANDIDATELIST_XDIM
    };
    constexpr uint sfragfct = 0;//fragment factor
    __shared__ int strdata[lYDIM][lXDIM + pad];

    if(threadIdx.x == 0)
        strdata[threadIdx.y][PREPXS] = 0;

    for(uint dbstrndx0 = 0; dbstrndx0 < ndbCstrs; dbstrndx0 += blockDim.x)
    {
        //update the prefix sums originated from processing the last data block:
        if(threadIdx.x == 0 && dbstrndx0)
            strdata[threadIdx.y][PREPXS] = strdata[threadIdx.y][pad + lXDIM - 1];

        //NOTE: strdata is overwritten below, sync:
        __syncthreads();

        uint dbstrndx = dbstrndx0 + threadIdx.x;//reference index
        int value = 0;//convflag for lXFLG and dbstrlen for lXLEN

        //get convergence flags (over all queries)
        if(threadIdx.y == lXFLG && dbstrndx < ndbCstrs) {
            for(uint qryndx = 0; qryndx < nqystrs; qryndx++) {
                uint mloc0 = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
                int lconv = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx];//float->int
                value += ((lconv & CONVERGED_LOWTMSC_bitval) != 0);
            }
        }
        if(threadIdx.y == lXFLG)
            //progressing structures have no convergence flags set for all queries
            strdata[threadIdx.y][pad + threadIdx.x] = value = (value < nqystrs);

        //get reference lengths
        if(threadIdx.y == lXLEN && dbstrndx < ndbCstrs)
            strdata[threadIdx.y][pad + threadIdx.x] = value = GetDbStrLength(dbstrndx);

        __syncthreads();

        //set reference lengths to 0 where convflag is set
        if(threadIdx.y == lXLEN && strdata[lXFLG][pad + threadIdx.x] == 0)
            strdata[threadIdx.y][pad + threadIdx.x] = value = 0;

        __syncthreads();

        //calculate inclusive (!) prefix sums for both flags, which then give indices,
        //and lengths for addresses:
        for(uint xdim = 1; xdim < blockDim.x; xdim <<= 1) {
            if(xdim <= threadIdx.x)
                value += strdata[threadIdx.y][pad + threadIdx.x - xdim];
            __syncthreads();
            if(xdim <= threadIdx.x)
                strdata[threadIdx.y][pad + threadIdx.x] = value;
            __syncthreads();
        }

        //correct the prefix sums by adding the previously obtained values:
        strdata[threadIdx.y][pad + threadIdx.x] += strdata[threadIdx.y][PREPXS];
        __syncthreads();

        //write to GMEM:
        if(dbstrndx < ndbCstrs) {
            uint mloc = threadIdx.y * ndbCstrs + dbstrndx;
            int valueprev = strdata[threadIdx.y][pad + threadIdx.x - 1];
            value = strdata[threadIdx.y][pad + threadIdx.x];
            //set to 0 for filtered-out structures:
            if(value == valueprev) value = valueprev = 0;
            //write adjusted addresses to gmem:
            if(threadIdx.y == lXLEN) value = valueprev;
            globvarsbuf[mloc] = value;
        }
        __syncthreads();
    }
}



// -------------------------------------------------------------------------
// ReformatStructureDataPartStore: reformat a reference database chunk to
// include candidates proceeding to stages of more detailed superposition
// search and refinement; this part corresponds to storing data to
// secondary (temporary) location first;
// nqystrs, total number of queries in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxndbCposs, max number of db structure positions in the chunk;
// maxnsteps, max number of steps allocated for each reference structure;
// ndbCstrs2, total number of selected reference structures;
// ndbCposs2, total number of positions of selected reference structures;
// dbstr1len2, length of the largest reference structure selected;
// it is used also for address adjustment;
// NOTE: memory pointers should be aligned!
// globvarsbuf, memory of new indices and addresses of selected references;
// wrkmemaux, auxiliary working memory;
// tfmmem, memory of transformation matrices;
// tmpdpdiagbuffers, temporary memory large enough (!) to contain all data to
// be copied from structure data, wrkmemaux, and tfmmem;
// nqycpxs, total number of query complexes in the chunk; 0 for single-chain version;
// ndbCcpxs, total number of reference complexes in the chunk; 0 for single-chain version;
// NOTE: thread block is 1D and processes a fragment of each reference structure;
// NOTE: thread block's x-dimension assumed to be 32!
// 
__global__ void ReformatStructureDataPartStore(
    uint nqystrs,
    const uint ndbCstrs,
    const uint maxndbCposs,
    const uint maxnsteps,
    // const uint ndbCstrs2,
    // const uint ndbCposs2,
    // const uint dbstr1len2,
    const uint* __restrict__ globvarsbuf,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tfmmem,
    float* __restrict__ tmpdpdiagbuffers,
    const uint nqycpxs,
    const uint ndbCcpxs)
{
    enum {
        lXNDX = fdNewReferenceIndex,//index for reference structure convergence flags/new indices
        lXADD = fdNewReferenceAddress,//index for reference structure convergence lengths/new addresses
        lXDIM = CUFL_STORECANDIDATEDATA_XDIM
    };
    // blockIdx.x is the block index of positions for one reference;
    // blockIdx.y is the reference serial number;
    // const uint dbstrfrg = blockIdx.x;
    const uint dbstrndx = blockIdx.y;
    //relative position index:
    const uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint newndx, newaddr;
    INTYPE length;
    LNTYPE dbstrdst;//address;

    if(threadIdx.x == 0) {
        newndx = globvarsbuf[lXNDX * ndbCstrs + dbstrndx];
        newaddr = globvarsbuf[lXADD * ndbCstrs + dbstrndx];
    }

    newndx = __shfl_sync(0xffffffff, newndx, 0/*srcLane*/);
    newaddr = __shfl_sync(0xffffffff, newaddr, 0/*srcLane*/);

    if(newndx)
    {
        //adjust index appropriately (NOTE: newaddr is already valid):
        newndx--;

        if(threadIdx.x == 0) {
            length = GetDbStrField<INTYPE,pps2DLen>(dbstrndx);
            dbstrdst = GetDbStrField<LNTYPE,pps2DDist>(dbstrndx);
        }

        if(blockIdx.x == 0 && threadIdx.x == 0) {
            //read originals:
            INTYPE type = GetDbStrField<INTYPE,pps2DType>(dbstrndx);
            //write structure-specific fields to new gmem location:
            ((INTYPE*)(tmpdpdiagbuffers + pps2DLen * maxndbCposs))[newndx] = length;
            ((INTYPE*)(tmpdpdiagbuffers + pps2DType * maxndbCposs))[newndx] = type;
            ((LNTYPE*)(tmpdpdiagbuffers + pps2DDist * maxndbCposs))[newndx] = newaddr;
        }

        length = __shfl_sync(0xffffffff, length, 0/*srcLane*/);
        dbstrdst = __shfl_sync(0xffffffff, dbstrdst, 0/*srcLane*/);

        //read and write position-specific fields:
        if(pos < length) {
            FPTYPE coord0 = GetDbStrField<FPTYPE,pmv2DCoords+0>(dbstrdst + pos);
            FPTYPE coord1 = GetDbStrField<FPTYPE,pmv2DCoords+1>(dbstrdst + pos);
            FPTYPE coord2 = GetDbStrField<FPTYPE,pmv2DCoords+2>(dbstrdst + pos);
            INTYPE icho = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst + pos);
            INTYPE rnum = GetDbStrField<INTYPE,pmv2DResNumber>(dbstrdst + pos);
            CHTYPE rsd = GetDbStrField<CHTYPE,pmv2Drsd>(dbstrdst + pos);
            CHTYPE ssa = GetDbStrField<CHTYPE,pmv2Dss>(dbstrdst + pos);
            ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+0) * maxndbCposs))[newaddr + pos] = coord0;
            ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+1) * maxndbCposs))[newaddr + pos] = coord1;
            ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+2) * maxndbCposs))[newaddr + pos] = coord2;
            ((INTYPE*)(tmpdpdiagbuffers + pmv2D_Ins_Ch_Ord * maxndbCposs))[newaddr + pos] = icho;
            ((INTYPE*)(tmpdpdiagbuffers + pmv2DResNumber * maxndbCposs))[newaddr + pos] = rnum;
            ((INTYPE*)(tmpdpdiagbuffers + pmv2Drsd * maxndbCposs))[newaddr + pos] = rsd;//NOTE INTYPE
            ((INTYPE*)(tmpdpdiagbuffers + pmv2Dss * maxndbCposs))[newaddr + pos] = ssa;//NOTE INTYPE
        }

        //read and write position-specific fields of 3D index:
        if(pos < length) {
            uint offset = pmv2DTotFlds * maxndbCposs;
            FPTYPE ndxcrd0 = GetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+0>(dbstrdst + pos);
            FPTYPE ndxcrd1 = GetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+1>(dbstrdst + pos);
            FPTYPE ndxcrd2 = GetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+2>(dbstrdst + pos);
            INTYPE left = GetIndxdDbStrField<INTYPE,pmv2DNdxLeft>(dbstrdst + pos);
            INTYPE right = GetIndxdDbStrField<INTYPE,pmv2DNdxRight>(dbstrdst + pos);
            INTYPE ndxorg = GetIndxdDbStrField<INTYPE,pmv2DNdxOrgndx>(dbstrdst + pos);
            ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+0) * maxndbCposs))[newaddr + pos] = ndxcrd0;
            ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+1) * maxndbCposs))[newaddr + pos] = ndxcrd1;
            ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+2) * maxndbCposs))[newaddr + pos] = ndxcrd2;
            ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxLeft * maxndbCposs))[newaddr + pos] = left;
            ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxRight * maxndbCposs))[newaddr + pos] = right;
            ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxOrgndx * maxndbCposs))[newaddr + pos] = ndxorg;
        }
    }//if(newndx)

    if(nqycpxs && ndbCcpxs) {
        if(ndbCcpxs <= dbstrndx) return;//NOTE: dbstrndx defines a block; sync irrelevant!
        if(threadIdx.x == 0)
            newndx = globvarsbuf[fdNewComplexIndex * ndbCstrs + dbstrndx];
        newndx = __shfl_sync(0xffffffff, newndx, 0/*srcLane*/);
        if(newndx == 0) return;//all exit
        //adjust complex index (NOTE!) appropriately:
        newndx--;
        //NOTE: dbstrndx is now reference complex index!
        //NOTE: nqystrs is now #query complexes:
        nqystrs = nqycpxs;
    }

    //read and write query-reference statistics obtained so far:
    for(uint qryndx = blockIdx.x; qryndx < nqystrs; qryndx++)
    {
        //NOTE: expected length across db references >=20 by definition: GetDEV_EXPCT_DBPROLEN()!
        const uint offset = (pmv2DTotFlds + pmv2DTotIndexFlds + qryndx) * maxndbCposs;
        const uint f = threadIdx.x;
        if(f < nTAuxWorkingMemoryVars) {
            //NOTE: uncoalesced read, coalesced write:
            uint mloc = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + f) * ndbCstrs;
            //NOTE: nTAuxWorkingMemoryVars * max(ndbCstrs) < max(ndbCposs) by definition (memory layout)
            uint tloc = offset + nTAuxWorkingMemoryVars * newndx;
            tmpdpdiagbuffers[tloc + f] = wrkmemaux[mloc + dbstrndx];
        }
        //only the last block loops until all query sections are processed:
        if(blockIdx.x + 1 < gridDim.x) break;
    }

    //read and write query-reference transformation matrices obtained so far:
    for(uint qryndx = blockIdx.x; qryndx < nqystrs; qryndx++)
    {
        const uint offset = (pmv2DTotFlds + pmv2DTotIndexFlds + nqystrs + qryndx) * maxndbCposs;
        const uint f = threadIdx.x;
        if(f < nTTranformMatrix) {
            //NOTE: coalesced read, coalesced write:
            uint mloc = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
            //NOTE: nTTranformMatrixv * max(ndbCstrs) < max(ndbCposs) by definition (memory layout)
            uint tloc = offset + nTTranformMatrix * newndx;
            tmpdpdiagbuffers[tloc + f] = tfmmem[mloc + f];
        }
        //only the last block loops until all query sections are processed:
        if(blockIdx.x + 1 < gridDim.x) break;
    }
}



// -------------------------------------------------------------------------
// ReformatStructureDataPartLoad: reformat a reference database chunk to
// include candidates proceeding to stages of more detailed superposition
// search and refinement; this part corresponds to data load from secondary
// (temporary) location;
// nqystrs, total number of queries in the chunk;
// maxndbCposs, max number of db structure positions the chunk can accommodate;
// maxnsteps, max number of steps allocated for each reference structure;
// ndbCstrs2, total number of selected reference structures;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary memory containing all data to be copied back to
// structure data, wrkmemaux, and tfmmem;
// wrkmemaux, auxiliary working memory;
// tfmmem, memory of transformation matrices;
// nqycpxs, total number of query complexes in the chunk; 0 for single-chain version;
// ndbCcpxs, total number of reference complexes in the chunk; 0 for single-chain version;
// NOTE: thread block is 1D and processes a fragment of each reference structure;
// NOTE: thread block's x-dimension assumed to be 32!
// 
__global__ void ReformatStructureDataPartLoad(
    uint nqystrs,
    const uint maxndbCposs,
    const uint maxnsteps,
    const uint ndbCstrs2,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem,
    const uint nqycpxs,
    const uint ndbCcpxs)
{
    enum {lXDIM = CUFL_LOADCANDIDATEDATA_XDIM};
    // blockIdx.x is the block index of positions for one reference;
    // blockIdx.y is the reference serial number;
    // const uint dbstrfrg = blockIdx.x;
    const uint dbstrndx = blockIdx.y;//corresponds to newndx of part store
    //relative position index:
    const uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    LNTYPE dbstrdst;//address; corresponds to newaddr of part store
    INTYPE length;

    if(threadIdx.x == 0) {
        //read structure-specific fields from a stored location:
        length = ((INTYPE*)(tmpdpdiagbuffers + pps2DLen * maxndbCposs))[dbstrndx];
        dbstrdst = ((LNTYPE*)(tmpdpdiagbuffers + pps2DDist * maxndbCposs))[dbstrndx];
        if(blockIdx.x == 0) {
            //only the first block writes structure-specific fields:
            INTYPE type = ((INTYPE*)(tmpdpdiagbuffers + pps2DType * maxndbCposs))[dbstrndx];
            SetDbStrField<INTYPE,pps2DLen>(dbstrndx, length);
            SetDbStrField<LNTYPE,pps2DDist>(dbstrndx, dbstrdst);
            SetDbStrField<INTYPE,pps2DType>(dbstrndx, type);
        }
    }

    length = __shfl_sync(0xffffffff, length, 0/*srcLane*/);
    dbstrdst = __shfl_sync(0xffffffff, dbstrdst, 0/*srcLane*/);

    //read and write position-specific fields:
    if(pos < length) {
        FPTYPE coord0 = ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+0) * maxndbCposs))[dbstrdst + pos];
        FPTYPE coord1 = ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+1) * maxndbCposs))[dbstrdst + pos];
        FPTYPE coord2 = ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+2) * maxndbCposs))[dbstrdst + pos];
        INTYPE icho = ((INTYPE*)(tmpdpdiagbuffers + pmv2D_Ins_Ch_Ord * maxndbCposs))[dbstrdst + pos];
        INTYPE rnum = ((INTYPE*)(tmpdpdiagbuffers + pmv2DResNumber * maxndbCposs))[dbstrdst + pos];
        CHTYPE rsd = ((INTYPE*)(tmpdpdiagbuffers + pmv2Drsd * maxndbCposs))[dbstrdst + pos];//NOTE INTYPE
        CHTYPE ssa = ((INTYPE*)(tmpdpdiagbuffers + pmv2Dss * maxndbCposs))[dbstrdst + pos];//NOTE INTYPE
        SetDbStrField<FPTYPE,pmv2DCoords+0>(dbstrdst + pos, coord0);
        SetDbStrField<FPTYPE,pmv2DCoords+1>(dbstrdst + pos, coord1);
        SetDbStrField<FPTYPE,pmv2DCoords+2>(dbstrdst + pos, coord2);
        SetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst + pos, icho);
        SetDbStrField<INTYPE,pmv2DResNumber>(dbstrdst + pos, rnum);
        SetDbStrField<CHTYPE,pmv2Drsd>(dbstrdst + pos, rsd);
        SetDbStrField<CHTYPE,pmv2Dss>(dbstrdst + pos, ssa);
    }

    //read and write position-specific fields of 3D index:
    if(pos < length) {
        uint offset = pmv2DTotFlds * maxndbCposs;
        FPTYPE ndxcrd0 = ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+0) * maxndbCposs))[dbstrdst + pos];
        FPTYPE ndxcrd1 = ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+1) * maxndbCposs))[dbstrdst + pos];
        FPTYPE ndxcrd2 = ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+2) * maxndbCposs))[dbstrdst + pos];
        INTYPE left = ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxLeft * maxndbCposs))[dbstrdst + pos];
        INTYPE right = ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxRight * maxndbCposs))[dbstrdst + pos];
        INTYPE ndxorg = ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxOrgndx * maxndbCposs))[dbstrdst + pos];
        SetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+0>(dbstrdst + pos, ndxcrd0);
        SetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+1>(dbstrdst + pos, ndxcrd1);
        SetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+2>(dbstrdst + pos, ndxcrd2);
        SetIndxdDbStrField<INTYPE,pmv2DNdxLeft>(dbstrdst + pos, left);
        SetIndxdDbStrField<INTYPE,pmv2DNdxRight>(dbstrdst + pos, right);
        SetIndxdDbStrField<INTYPE,pmv2DNdxOrgndx>(dbstrdst + pos, ndxorg);
    }

    if(nqycpxs && ndbCcpxs) {
        if(ndbCcpxs <= dbstrndx) return;//NOTE: dbstrndx defines a block; sync irrelevant!
        //NOTE: dbstrndx is now reference complex index!
        //NOTE: nqystrs is now #query complexes:
        nqystrs = nqycpxs;
    }

    //read and write query-reference statistics obtained so far:
    for(uint qryndx = blockIdx.x; qryndx < nqystrs; qryndx++)
    {
        //NOTE: expected length across db references >=20 by definition: GetDEV_EXPCT_DBPROLEN()!
        const uint offset = (pmv2DTotFlds + pmv2DTotIndexFlds + qryndx) * maxndbCposs;
        const uint f = threadIdx.x;
        if(f < nTAuxWorkingMemoryVars) {
            //NOTE: coalesced read, uncoalesced write:
            uint mloc = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + f) * ndbCstrs2;
            uint tloc = offset + nTAuxWorkingMemoryVars * dbstrndx;
            wrkmemaux[mloc + dbstrndx] = tmpdpdiagbuffers[tloc + f];
            //reset convergence flags:
            if(f == tawmvConverged) wrkmemaux[mloc + dbstrndx] = 0.0f;
        }
        //only the last block loops until all query sections are processed:
        if(blockIdx.x + 1 < gridDim.x) break;
    }

    //read and write query-reference transformation matrices obtained so far:
    for(uint qryndx = blockIdx.x; qryndx < nqystrs; qryndx++)
    {
        const uint offset = (pmv2DTotFlds + pmv2DTotIndexFlds + nqystrs + qryndx) * maxndbCposs;
        const uint f = threadIdx.x;
        if(f < nTTranformMatrix) {
            //NOTE: coalesced read, coalesced write:
            uint mloc = (qryndx * ndbCstrs2 + dbstrndx) * nTTranformMatrix;
            uint tloc = offset + nTTranformMatrix * dbstrndx;
            tfmmem[mloc + f] = tmpdpdiagbuffers[tloc + f];
        }
        //only the last block loops until all query sections are processed:
        if(blockIdx.x + 1 < gridDim.x) break;
    }
}
