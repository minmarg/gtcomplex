/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __reformatmatch_cuh__
#define __reformatmatch_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"

// =========================================================================
// ReformatMatched4Complexes: reformat matched (aligned) positions of 
// full complex (all chains) to obtain continuous stretch of aligned 
// positions
template<
    int MAX_NCHAINS = 0,
    int CPXTYPENUMBERSCT = sfin_cpx_complextypenumber2>
__global__
void ReformatMatched4Complexes(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint assgmaxnsteps,
    const uint assgstepnumber,
    const uint complexstepnumber,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpalnpossbuffer,
    const int final = 0
);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

#endif//__reformatmatch_cuh__
