/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __constrained_btck2match_cuh__
#define __constrained_btck2match_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"

// =========================================================================
// ConstrainedBtckToMatched32x: copy the coordinates of matched (aligned) 
// positions within a given distance threshold to destination location for 
// final refinement; use 32x unrolling;
template<bool COMPLEX = false>
__global__
void ConstrainedBtckToMatched32x(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const char* __restrict__ btckdata,
    const float* __restrict__ tfmmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem2 = NULL,
    const uint complexstepnumber = 0,
    const uint complexstepnumber2 = 1,
    const uint assgmaxnsteps = 0,
    const uint assgstepnumber = 0
);

// =========================================================================
// -------------------------------------------------------------------------

#endif//__constrained_btck2match_cuh__
