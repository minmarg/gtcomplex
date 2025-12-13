/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "coverage.cuh"

// -------------------------------------------------------------------------
// CheckMaxCoverage: calculate maximum coverage between the queries and 
// reference structures and set the skip flag (convergence) if it is 
// below the threshold; the function actually resets convergence;
// covthreshold, coverage threshold;
// ntotqcpxs, total #queries processed and being processed so far;
// // nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnstepsmem2, max number of steps for each reference structure in wrkmem2;
// NOTE: memory pointers should be aligned!
// wrkmem2, working memory (for chain scores and assignments), used for convergence flags here;
// NOTE: thread block is 1D and processes query-reference structure pairs;
// 
__global__ void CheckMaxCoverage(
    const float covthreshold,
    const int ntotqcpxs,
    const uint ndbCstrs,
    const uint maxnstepsmem2,
    float* __restrict__ wrkmem2)
{
    //index of the reference structure:
    const uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint mloc2 = ((qryndx * maxnstepsmem2 + 0) * nTWorkingMemory2VarsAssignment + twm2aTmpConvergence) * ndbCstrs;

    if(ndbCstrs <= dbstrndx) return;//no sync below: exit

    const int dbcpxglobndx = GetDbStrField<INTYPE,pps2DType>(dbstrndx);

    if(dbcpxglobndx < 0 || ntotqcpxs <= dbcpxglobndx) return;//no sync below; exit

    const int qrydst = GetQueryDst(qryndx);
    const int qrytypeind = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(qrydst);
    const int dbstrdst = GetDbStrDst(dbstrndx);
    const int dbstrtypeind = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst);

    if(GetMoleculeType(qrytypeind) != GetMoleculeType(dbstrtypeind)) return;//no sync below; exit

    int addr, qrylen, dbstrlen;
    int qrycpxndx = GetQueryStrField<INTYPE,pps2DCpxI>(qryndx);
    int dbcpxndx = GetDbStrField<INTYPE,pps2DCpxI>(dbstrndx);
    GetQueryComplexAddressLength(qrycpxndx, addr, qrylen);
    GetDbComplexAddressLength(dbcpxndx, addr, dbstrlen);

    if((float)myhdmax(qrylen, dbstrlen) * covthreshold <= (float)myhdmin(qrylen, dbstrlen)) {
        //reset convergence flag:
        wrkmem2[mloc2 + dbstrndx] = 0;
    }
}
