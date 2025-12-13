/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <tuple>
#include <vector>
#include <utility>
#include <algorithm>

#include "libutil/CLOptions.h"
#include "libutil/format.h"

// #include "libgenp/gdats/PM2DVectorFields.h"
// #include "libgenp/gdats/PMBatchStrData.h"
// #include "libgenp/goutp/TdAlnWriter.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gproc/gproc.h"

// #include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "TdFinalizer.h"

// -------------------------------------------------------------------------
// CompressResultsPlainComplex: process results for complexes; compress 
// these results for passing them to the writing thread; use plain format;
// qrycpxnr, query complex number relative to the current chunk;
// NOTE: all operations performed under lock
//
void TdFinalizer::CompressResultsPlainComplex(
    int qrycpxnr, int qrycpxN, int qrycpxDstN)
{
    MYMSG("TdFinalizer::CompressResultsPlainComplex", 4);
    static const unsigned int indent = OUTPUTINDENT;
    // static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    const unsigned int dscwidth = MAX_DESCRIPTION_LENGTH;//no wrap (pathname)
    static const int complexes = (CLOptions::GetB_CHAINS() == 0);
    static const unsigned int alnwidth = CLOptions::GetO_WRAP();
    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    static const int sortby = CLOptions::GetO_SORT();
    static const unsigned int sznl = (int)strlen(NL);
    static const unsigned int headlines = 3 + (bsectmscore==1);
    static const unsigned int footlines = 4;//tfm lines
    static const unsigned int maxsernlen = 13;//max length for serial number
    const unsigned int width = indent < dscwidth? dscwidth: 1;
    // const int qrycpxNend = qrycpxN + qrycpxDstN;
    size_t szannot = 0UL;
    size_t szcpxhead = 0UL, szalns = 0UL;
    size_t szalnswodesc = 0UL;
    const char* desc;//structure description
    int written, sernr = 0;//initialize serial number here
    char errbuf[BUF_MAX];

    if(!cubp_set_querypmbeg_[0] || !cubp_set_querypmend_[0] || !cubp_set_querydesc_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlainComplex: Null query data.");

    if(cubp_set_bdbCpmbeg_[0] && cubp_set_bdbCpmend_[0] && !cubp_set_bdbCdesc_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlainComplex: Null structure descriptions.");

    int rc = 0, ri = 0;

    for(rc = 0, ri = 0; ri < qrynstrs_; rc++) 
    {
        int rfncpxN = GetDbStructureField<INTYPE>(rc, pcx2DN);
        int rfncpxDstN = (int)GetDbStructureField<LNTYPE>(rc, pcx2DDstN);
        // int rfncpxNend = rfncpxN + rfncpxDstN;

        if(rfncpxDstN != ri) {
            sprintf(errbuf, "TdFinalizer::CompressResultsPlainComplex: "
                "Inconsistent reference chain no. %d: %d", ri, rfncpxDstN);
            throw MYRUNTIME_ERROR(errbuf);
        }

        if(rfncpxN < 1) {
            sprintf(errbuf, "TdFinalizer::CompressResultsPlainComplex: "
                "Invalid #chains for relative reference complex no. %d (%d): %d", 
                rc, ri, rfncpxN);
            throw MYRUNTIME_ERROR(errbuf);
        }

        if(qrynstrs_ < ri + rfncpxN) {
            sprintf(errbuf, "TdFinalizer::CompressResultsPlainComplex: "
                "#chains for relative reference complex no. %d (%d) > "
                "#total chains (%d): %d", 
                rc, ri, qrynstrs_, rfncpxN);
            throw MYRUNTIME_ERROR(errbuf);
        }

        GetDbStructureDesc(desc, rfncpxDstN/*strndx*/);

        size_t cpxdesclen = mymin((size_t)dsclen, (strlen(desc) + 2));
        size_t cpxdescfrags = (cpxdesclen + width - 1)/width;
        size_t cpxdesclenperline = mymin((size_t)dscwidth, cpxdesclen);

        szcpxhead += maxsernlen + 4 * sznl;//alignment separator, including serial number
        szcpxhead += cpxdescfrags * (cpxdesclenperline + sznl) + sznl;
        szcpxhead += headlines * (fpc_maxlinelen + sznl) + 2 * sznl;
        szcpxhead += footlines * (fpc_maxlinelen + sznl) + sznl;

        size_t szalnswodesc1 = 0UL;

        GetSizeOfCompressedResultsPlainComplex(qrycpxN, rfncpxN, rfncpxDstN, &szalnswodesc1);

        szannot += fpc_maxlinelen + sznl;
        szalnswodesc += szalnswodesc1;

        ri += rfncpxN;
    }

    szalns = szcpxhead + szalnswodesc;

    if(cubp_set_qrysernrbeg_ < 0 || qrysernr_ < cubp_set_qrysernrbeg_ || 
       cubp_set_qrysernrbeg_ + (int)cubp_set_nqystrs_ <= qrysernr_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlainComplex: Invalid query indices.");

    annotations_.reset();
    alignments_.reset();

    ReserveVectors(rc);

    if(szalns < szannot || 
       szalnswodesc > 4 * 
            (cubp_set_sz_alndata_ + cubp_set_sz_tfmmatrices_ + cubp_set_sz_alns_))
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlainComplex: "
        "Size of formatted results is unusually large.");

    if(szannot < 1 || szalns < 1) return;

    annotations_.reset((char*)std::malloc(szannot));
    alignments_.reset((char*)std::malloc(szalns));

    if(!annotations_ || !alignments_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlainComplex: Not enough memory.");

    if(!srtindxs_ || !scores_ || !alnptrs_ || !annotptrs_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlainComplex: Not enough memory.");

    char* annptr = annotations_.get();
    char* outptr = alignments_.get();

    for(rc = 0, ri = 0; ri < qrynstrs_; rc++) 
    {
        int rfncpxN = GetDbStructureField<INTYPE>(rc, pcx2DN);
        int rfncpxDstN = (int)GetDbStructureField<LNTYPE>(rc, pcx2DDstN);
        // int rfncpxNend = rfncpxN + rfncpxDstN;
        ri += rfncpxN;

        float tmscoreq = GetOutputAlnDataFieldComplex<float>(qrycpxnr, rc, dp2oadScoreQ);
        float tmscorer = GetOutputAlnDataFieldComplex<float>(qrycpxnr, rc, dp2oadScoreR);
        float rmsd = GetOutputAlnDataFieldComplex<float>(qrycpxnr, rc, dp2oadRMSD);
        float tmscoregrt = mymax(tmscoreq, tmscorer);
        float tmscorehmn = (0.0f < tmscoregrt)? (2.f * tmscoreq * tmscorer) / (tmscoreq + tmscorer): 0.0f;
        float tmscore = tmscoregrt;
        if(sortby == CLOptions::osTMscoreReference) tmscore = tmscorer;
        if(sortby == CLOptions::osTMscoreQuery) tmscore = tmscoreq;
        if(sortby == CLOptions::osTMscoreHarmonic) tmscore = tmscorehmn;
        if(sortby == CLOptions::osRMSD) tmscore = -rmsd;
        if(bsectmscore && sortby > CLOptions::osRMSD) {
            float sectmscoreq = GetOutputAlnDataFieldComplex<float>(qrycpxnr, rc, dp2oad2ScoreQ);
            float sectmscorer = GetOutputAlnDataFieldComplex<float>(qrycpxnr, rc, dp2oad2ScoreR);
            tmscore = mymax(sectmscoreq, sectmscorer);
            float sectmscorehmn = (0.0f < tmscore)? (2.f * sectmscoreq * sectmscorer) / (sectmscoreq + sectmscorer): 0.0f;
            if(sortby == CLOptions::os2TMscoreReference) tmscore = sectmscorer;
            if(sortby == CLOptions::os2TMscoreQuery) tmscore = sectmscoreq;
            if(sortby == CLOptions::os2TMscoreHarmonic) tmscore = sectmscorehmn;
        }

        if(tmscoregrt < cubp_set_scorethld_) continue;

        unsigned int alnlen = 0;
        int psts, idts, gaps, nchns;

        GetStatisticsForComplexPair(
            qrycpxN, rfncpxN, rfncpxDstN,  &alnlen, &psts, &idts, &gaps, &nchns);

        if(alnlen < 1) continue;

        //save the addresses of the annotations and alignment records
        srtindxs_->push_back(sernr++);
        scores_->push_back(tmscore);
        annotptrs_->push_back(annptr);
        alnptrs_->push_back(outptr);

        int dbstrlen = GetDbCpxLength(rc);

        GetDbStructureDesc(desc, rfncpxDstN/*strndx*/);
        std::string::size_type posc, posm;
        std::string descopy = desc;
        if(complexes && 
          (posc = descopy.rfind(" Chn:")) != std::string::npos) {
            if((posm = descopy.rfind(" (M:")) != std::string::npos)
                descopy = descopy.substr(0,posc) + descopy.substr(posm);
            else descopy = descopy.substr(0,posc);
        }

        //make annotation
        MakeAnnotationPlainComplex( 
            annptr, qrycpxnr, rc, qrycpxN, rfncpxN,
            descopy.c_str(), alnlen, psts, idts, gaps, nchns, dbstrlen);
        *annptr++ = 0;//end of record

        //reserve space for serial number:
        written = sprintf(outptr,"%13c%s",' ',NL);
        outptr += written;
        //put the description...
        int outpos = 0;
        int linepos = 0;
        char addsep = '>';
        FormatDescription(
            outptr, descopy.c_str(),
            dsclen, indent, width, outpos, linepos, addsep);

        PutNL(outptr);

        FormatScoresPlainComplex(
            outptr, qrycpxnr, rc, qrycpxN, rfncpxN,
            alnlen, psts, idts, gaps, qrystrlen_, dbstrlen);

        FormatFooterPlainComplex(outptr, qrycpxnr, rc);

        FormatAlignmentPlainComplex(
            outptr, qrycpxnr, rc, qrycpxN, rfncpxN, qrycpxDstN, rfncpxDstN, alnwidth);

        // PutNL(outptr);

        written = sprintf( outptr,"%s%s",NL,NL);
        outptr += written;
        *outptr++ = 0;//end of record
    }
}



// -------------------------------------------------------------------------
// GetStatisticsForComplexPair, get alignment statistics for a complex pair;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// rfncpxDstN, distance in chains to te current reference complex;
// palnlen, ppsts, pidts, pgaps, nchns, numbers of alignment symbols,
// positives, identities, gaps, and aligned chains;
// NOTE: addresses assumed to be valid!
// inline
void TdFinalizer::GetStatisticsForComplexPair(
    int qrycpxN, int rfncpxN, int rfncpxDstN,
    unsigned int* palnlen, int* ppsts, int* pidts, int* pgaps, int* nchns) const
{
    *palnlen = 0;
    *ppsts = *pidts = *pgaps = *nchns = 0;

    if(!cubp_set_h_results_) return;

    for(int qrychnndx = 0; qrychnndx < qrycpxN; qrychnndx++)
        for(int dbchnndx = 0; dbchnndx < rfncpxN; dbchnndx++)
        {
            const int dbstrndx = rfncpxDstN + dbchnndx;

            //chainn assignment:
            float chnassg = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadOrgStrNo);
            if(chnassg == 0.0f) continue;

            // float tmscoreq = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadScoreQ_C);
            // float tmscorer = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadScoreR_C);
            // float tmscore = mymax(tmscoreq, tmscorer);
            // if(tmscore < cubp_set_scorethld_) continue;

            unsigned int alnlen = (unsigned int)
                GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadAlnLength);
            //NOTE: alignment length can be 0 due to pre-screening
            if(alnlen < 1) continue;

            unsigned int dbstr2dst = GetDbStructureField<unsigned int>(dbstrndx, pps2DDist);
            unsigned int dbstrlen = (unsigned int)GetDbStructureField<INTYPE>(dbstrndx, pps2DLen);
            //if true, the structure has not been processed due to memory restrictions
            if(qrynposits_ < dbstr2dst + dbstrlen) continue;

            int psts = (int)GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadPstvs);
            int idts = (int)GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadIdnts);
            int gaps = (int)GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadNGaps);

            *palnlen += alnlen;
            *ppsts += psts; *pidts += idts; *pgaps += gaps;
            (*nchns)++;
        }
}



// -------------------------------------------------------------------------
// MakeAnnotationPlainComplex: format complex description;
// NOTE: space is assumed to be pre-allocated;
// outptr, pointer to the output buffer;
// qrycpxndx, rfncpxndx, query and reference complex indices;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// desc, reference complex description;
// alnlen, psts, idts, gaps, nchns, numbers of alignment symbols, positives, 
// identities, gaps,and aligned chains;
// dbcpxlen, reference complex length;
inline
void TdFinalizer::MakeAnnotationPlainComplex( 
    char*& outptr,
    int qrycpxndx, int rfncpxndx, int /*qrycpxN*/, int rfncpxN,
    const char* desc, const unsigned int alnlen,
    const int psts, const int idts, const int gaps, const int /*nchns*/,
    const int dbcpxlen) const
{
    int written;
    float rmsd = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadRMSD);
    float tmscoreq = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadScoreQ);
    float tmscorer = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadScoreR);

    //reserve for serial number
    written = sprintf(outptr,"%6c",' ');
    outptr += written;

    int desclen = strlen(desc);
    if(desclen <= ANNOTATION_DESCLEN)
        written = sprintf(outptr," %-" TOSTR(ANNOTATION_DESCLEN) "s",desc);
    else {
        written = sprintf(outptr," ...");
        outptr += written;
        written = sprintf(outptr,"%-" TOSTR(ANNOTATION_DESCLEN_3l) "s",
            desc+desclen-ANNOTATION_DESCLEN_3l);
    }
    outptr += written;

    written = sprintf(outptr," %6.4f %6.4f %5.2f",tmscorer,tmscoreq,rmsd);
    outptr += written;

    written = sprintf(outptr," %5d %5d %4d%% %5d %4d  %5d%s",
        (int)alnlen-gaps, idts,idts*100/alnlen,psts,rfncpxN, dbcpxlen,NL);
    outptr += written;
}



// -------------------------------------------------------------------------
// FormatScoresPlainComplex: format complex scores;
// outptr, pointer to the output buffer;
// qrycpxndx, rfncpxndx, query and reference complex indices;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// palnlen, ppsts, pidts, pgaps, numbers of alignment symbols, positives, 
// identities, and gaps;
// querylen, query length;
// dbstrlen, reference complex length;
inline
void TdFinalizer::FormatScoresPlainComplex(
    char*& outptr,
    int qrycpxndx, int rfncpxndx,  int qrycpxN, int rfncpxN,
    unsigned int alnlen, int psts, int idts, int gaps,
    int querylen, int dbstrlen)
{
    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    int written;

    float rmsd = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadRMSD);
    float tmscoreq = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadScoreQ);
    float tmscorer = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadScoreR);
    float sectmscoreq = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oad2ScoreQ);
    float sectmscorer = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oad2ScoreR);
    float d0q = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadD0Q);
    float d0r = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadD0R);

    written = sprintf(outptr,"  Length (#chains): Refn. = %d (%d), Query = %d (%d)%s%s",
        dbstrlen, rfncpxN, querylen, qrycpxN, NL, NL);
    outptr += written;
    written = 
    sprintf(outptr," TM-score (Refn./Query) = %.5f / %.5f, "
        "d0 (Refn./Query) = %.2f / %.2f,  RMSD = %.2f A",
        tmscorer, tmscoreq, d0r, d0q, rmsd);
    outptr += written;
    written = sprintf(outptr,"%s",NL);
    outptr += written;
    if(bsectmscore) {
        written = 
        sprintf(outptr," 2TM-score (Refn./Query) = %.5f / %.5f", sectmscorer, sectmscoreq);
        outptr += written;
        written = sprintf(outptr,"%s",NL);
        outptr += written;
    }
    if(idts > 0) {
        written = 
            sprintf(outptr," Identities = %d/%d (%d%%)",idts,alnlen,idts*100/alnlen);
        outptr += written;
    }
    if(psts > 0) {
        if(idts) {
            written = sprintf( outptr,",");
            outptr += written;
        }
        written = 
            sprintf(outptr," Matched = %d/%d (%d%%)",psts,alnlen,psts*100/alnlen);
        outptr += written;
    }
    if(gaps > 0) {
        if(idts || psts) {
            written = sprintf( outptr,",");
            outptr += written;
        }
        written = sprintf( outptr," Gaps = %d/%d (%d%%)",gaps,alnlen,gaps*100/alnlen);
        outptr += written;
    }
    written = sprintf( outptr,"%s%s",NL,NL);
    outptr += written;
}

// -------------------------------------------------------------------------
// FormatFooterPlainComplex, format tfm;
// outptr, pointer to the output buffer;
// qrycpxndx, rfncpxndx, query and reference complex indices;
inline
void TdFinalizer::FormatFooterPlainComplex(
    char*& outptr, int qrycpxndx, int rfncpxndx)
{
    static const int referenced = CLOptions::GetO_REFERENCED();
    //address of the relevant transformation matrix data:
    float* ptfmmtx = GetOutputTfmMtxAddressComplex(qrycpxndx, rfncpxndx);
    int written;

    written =
        sprintf(outptr," Rotation [3,3] and translation [3,1] for %s:%s",
            (referenced? "Reference":"Query"), NL);
    outptr += written;

    written = sprintf(outptr,"  %10.6f %10.6f %10.6f    %14.6f%s",
        ptfmmtx[tfmmRot_0_0],ptfmmtx[tfmmRot_0_1],ptfmmtx[tfmmRot_0_2],ptfmmtx[tfmmTrl_0],NL);
    outptr += written;
    written = sprintf(outptr,"  %10.6f %10.6f %10.6f    %14.6f%s",
        ptfmmtx[tfmmRot_1_0],ptfmmtx[tfmmRot_1_1],ptfmmtx[tfmmRot_1_2],ptfmmtx[tfmmTrl_1],NL);
    outptr += written;
    written = sprintf(outptr,"  %10.6f %10.6f %10.6f    %14.6f%s%s",
        ptfmmtx[tfmmRot_2_0],ptfmmtx[tfmmRot_2_1],ptfmmtx[tfmmRot_2_2],ptfmmtx[tfmmTrl_2],NL,NL);
    outptr += written;
}



// -------------------------------------------------------------------------
// FormatAlignmentPlainComplex, format chain-specific alignments;
// outptr, pointer to the output buffer;
// qrycpxndx, rfncpxndx, query and reference complex indices;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// qrycpxDstN, rfncpxDstN, distance in chains to the current complexes;
// alnlen, alignment length;
// width, alignment output width;
inline
void TdFinalizer::FormatAlignmentPlainComplex(
    char*& outptr,
    const int qrycpxndx, const int /*rfncpxndx*/,
    const int qrycpxN, const int rfncpxN,
    const int qrycpxDstN, const int rfncpxDstN,
    const int width)
{
    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();
    static const int sortby = CLOptions::GetO_SORT();
    static const char* pattrn = " Chn:";
    static const size_t lenpattrn = strlen(pattrn);
    char locbuf[BUF_MAX];
    int written, nbytes, fgaps;

    // int nminchns = mymin(qrycpxN, rfncpxN);

    tmp_c2csorted_.clear();
    tmp_c2cscores_.clear();

    for(int qrychnndx = 0; qrychnndx < qrycpxN; qrychnndx++)
        for(int dbchnndx = 0; dbchnndx < rfncpxN; dbchnndx++)
        {
            const int dbstrndx = rfncpxDstN + dbchnndx;

            //chainn assignment:
            float chnassg = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadOrgStrNo);
            if(chnassg == 0.0f) continue;

            unsigned int alnlen = (unsigned int)
                GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadAlnLength);
            //NOTE: alignment length can be 0 due to pre-screening
            if(alnlen < 1) continue;

            unsigned int dbstr2dst = GetDbStructureField<unsigned int>(dbstrndx, pps2DDist);
            unsigned int dbstrlen = (unsigned int)GetDbStructureField<INTYPE>(dbstrndx, pps2DLen);
            //if true, the structure has not been processed due to memory restrictions
            if(qrynposits_ < dbstr2dst + dbstrlen) continue;

            float tmscoreq = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadScoreQ_C);
            float tmscorer = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadScoreR_C);
            float rmsd = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadRMSD_C);
            float tmscoregrt = mymax(tmscoreq, tmscorer);
            float tmscorehmn = (0.0f < tmscoregrt)? (2.f * tmscoreq * tmscorer) / (tmscoreq + tmscorer): 0.0f;
            float tmscore = tmscoregrt;
            if(sortby == CLOptions::osTMscoreReference) tmscore = tmscorer;
            if(sortby == CLOptions::osTMscoreQuery) tmscore = tmscoreq;
            if(sortby == CLOptions::osTMscoreHarmonic) tmscore = tmscorehmn;
            if(sortby == CLOptions::osRMSD) tmscore = -rmsd;
            if(bsectmscore && sortby > CLOptions::osRMSD) {
                float sectmscoreq = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oad2ScoreQ_C);
                float sectmscorer = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oad2ScoreR_C);
                tmscore = mymax(sectmscoreq, sectmscorer);
                float sectmscorehmn = (0.0f < tmscore)? (2.f * sectmscoreq * sectmscorer) / (sectmscoreq + sectmscorer): 0.0f;
                if(sortby == CLOptions::os2TMscoreReference) tmscore = sectmscorer;
                if(sortby == CLOptions::os2TMscoreQuery) tmscore = sectmscoreq;
                if(sortby == CLOptions::os2TMscoreHarmonic) tmscore = sectmscorehmn;
            }

            // if(tmscoregrt < cubp_set_scorethld_) continue;

            tmp_c2csorted_.push_back(std::make_tuple(tmscore, qrychnndx, dbchnndx));
        }

    std::sort(tmp_c2csorted_.begin(), tmp_c2csorted_.end(),
        [](auto const& t1, auto const& t2) {
            return std::get<0>(t1) > std::get<0>(t2);
        }
    );

    int i = 0;

    written = sprintf(outptr,
        " %5s %5s %5s  %10s %10s %6s  %5s %5s %11s %11s %5s%s",
        "#", "R_Chn", "Q_Chn",  "R_TM-score", "Q_TM-score", "RMSD",
        "#algn", "Q_Len", "Q_Segment", "R_Segment", "R_Len", NL);
    outptr += written;
    written = sprintf(outptr,
        " %5s %5s %5s  %10s %10s %6s  %5s %5s %11s %11s %5s%s",
        "---", "-----", "-----",  "----------", "----------", "------",
        "-----", "-----", "-----------", "-----------", "-----", NL);
    outptr += written;

    //print short annotation first:
    for(auto const& c2ct: tmp_c2csorted_)
    {
        const int qrychnndx = std::get<1>(c2ct);
        const int dbchnndx = std::get<2>(c2ct);
        const int qrystrndx = qrycpxDstN + qrychnndx;
        const int dbstrndx = rfncpxDstN + dbchnndx;

        unsigned int querylen = (unsigned int)GetQueryField<INTYPE>(qrystrndx, pps2DLen);
        unsigned int dbstrlen = (unsigned int)GetDbStructureField<INTYPE>(dbstrndx, pps2DLen);
        unsigned int dbstrdst = (unsigned int)GetDbStructureField<INTYPE>(dbstrndx, pps2DDist);
        int dbsttype = (int)GetDbStructureField<INTYPE>(dbstrdst, pmv2D_Ins_Ch_Ord);
        dbsttype = GetMoleculeType(dbsttype);
        const char* dbsttypestr = GetMoleculeTypeStr(dbsttype);
        float rmsd = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadRMSD_C);
        float tmscoreq = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadScoreQ_C);
        float tmscorer = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadScoreR_C);
        float sectmscoreq = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oad2ScoreQ_C);
        float sectmscorer = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oad2ScoreR_C);
        float d0q = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadD0Q_C);
        float d0r = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadD0R_C);
        int psts = (int)GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadPstvs);
        int idts = (int)GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadIdnts);
        int gaps = (int)GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadNGaps);
        unsigned int alnlen = (unsigned int)
            GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadAlnLength);
        //alignment beginning coordinates:
        unsigned int alnbegcoords = GetOutputAlnDataFieldChain<unsigned int>(qrychnndx, dbstrndx, dp2oadBegCoords);
        int qrybeg = GetCoordY(alnbegcoords);
        int trgbeg = GetCoordX(alnbegcoords);
        //alignment end coordinates:
        unsigned int alnendcoords = GetOutputAlnDataFieldChain<unsigned int>(qrychnndx, dbstrndx, dp2oadEndCoords);
        int qryend = GetCoordY(alnendcoords);
        int trgend = GetCoordX(alnendcoords);

        const char* rdesc, *qdesc;
        std::string rchnstr;// = "R_Chn:";
        std::string qchnstr;// = "Q_Chn:";
        GetDbStructureDesc(rdesc, dbstrndx);
        std::string::size_type posc, poss;
        tmp_summary_ = rdesc;
        if((posc = tmp_summary_.rfind(pattrn)) != std::string::npos) {
            size_t nmaxch = 5;
            if((poss = tmp_summary_.find_first_of(' ', posc + 1)) != std::string::npos)
                nmaxch = mymin(nmaxch, poss - posc - lenpattrn);
            rchnstr += tmp_summary_.substr(posc + lenpattrn, nmaxch);
        }
        GetQueryDesc(qdesc, qrystrndx);
        tmp_summary_ = qdesc;
        if((posc = tmp_summary_.rfind(pattrn)) != std::string::npos) {
            size_t nmaxch = 5;
            if((poss = tmp_summary_.find_first_of(' ', posc + 1)) != std::string::npos)
                nmaxch = mymin(nmaxch, poss - posc - lenpattrn);
            qchnstr += tmp_summary_.substr(posc + lenpattrn, nmaxch);
        }

        written = sprintf(outptr,
            " %5d %5s %5s  %10.4f %10.4f %6.2f  %5d %5d %5d-%-5d %5d-%-5d %5d%s",
            ++i, rchnstr.c_str(), qchnstr.c_str(),  tmscorer,tmscoreq,rmsd,
            (int)alnlen - gaps, querylen, qrybeg,qryend,trgbeg,trgend, dbstrlen, NL);
        outptr += written;

        tmp_summary_ = rdesc;
        if((posc = tmp_summary_.rfind(DIRSEPSTR)) != std::string::npos)
            tmp_summary_ = tmp_summary_.substr(posc + 1);
        if(ANNOTATION_DESCLEN < tmp_summary_.size()) {
            tmp_summary_ = tmp_summary_.substr(tmp_summary_.size() - ANNOTATION_DESCLEN);
            tmp_summary_[0] = tmp_summary_[1] = tmp_summary_[2] = '.';
        }

        sprintf(locbuf,"%s  [%d] Chain/Length: Refn. = %s / %d, Query = %s / %d   Type:%s   (%s)%s%s", NL,
            i, rchnstr.c_str(), dbstrlen, qchnstr.c_str(), querylen, dbsttypestr, tmp_summary_.c_str(), NL, NL);

        tmp_summary_.clear();
        tmp_summary_ += locbuf;
        sprintf(locbuf," TM-score (Refn./Query) = %.5f / %.5f, "
            "d0 (Refn./Query) = %.2f / %.2f,  RMSD = %.2f A%s",
            tmscorer, tmscoreq, d0r, d0q, rmsd, NL);
        tmp_summary_ += locbuf;
        if(bsectmscore) {
            sprintf(locbuf," 2TM-score (Refn./Query) = %.5f / %.5f%s", sectmscorer, sectmscoreq, NL);
            tmp_summary_ += locbuf;
        }
        if(idts > 0) {
            sprintf(locbuf," Identities = %d/%d (%d%%)", idts, alnlen, idts*100/alnlen);
            tmp_summary_ += locbuf;
        }
        if(psts > 0) {
            if(idts) tmp_summary_ += ",";
            sprintf(locbuf," Matched = %d/%d (%d%%)", psts, alnlen, psts*100/alnlen);
            tmp_summary_ += locbuf;
        }
        if(gaps > 0) {
            if(idts || psts) tmp_summary_ += ",";
            sprintf(locbuf," Gaps = %d/%d (%d%%)", gaps, alnlen, gaps*100/alnlen);
            tmp_summary_ += locbuf;
        }
        tmp_summary_ += NL; tmp_summary_ += NL;

        tmp_c2cscores_.push_back(
            std::make_tuple(qrychnndx, dbchnndx, tmp_summary_, alnlen, alnbegcoords, alnendcoords));
    }

    PutNL(outptr);

    //alignments themselves go next:
    for(auto const& c2cs: tmp_c2cscores_)
    {
        int qrychnndx = std::get<0>(c2cs);
        int dbchnndx = std::get<1>(c2cs);
        const std::string& summary = std::get<2>(c2cs);
        unsigned int alnlen = std::get<3>(c2cs);
        unsigned int alnbegcoords = std::get<4>(c2cs);
        unsigned int alnendcoords = std::get<5>(c2cs);

        int qrybeg = GetCoordY(alnbegcoords);
        int trgbeg = GetCoordX(alnbegcoords);
        int trgend = GetCoordX(alnendcoords);

        written = sprintf(outptr, "%s", summary.c_str());
        outptr += written;

        const int qrystrndx = qrycpxDstN + qrychnndx;
        const int dbstrndx = rfncpxDstN + dbchnndx;

        int querylen = (int)GetQueryField<INTYPE>(qrystrndx, pps2DLen);
        int dbstrdst = (int)GetDbStructureField<INTYPE>(dbstrndx, pps2DDist);
        int qrystrprtlen = GetQueryCpxPartialLength(qrycpxndx, qrychnndx);
        int dbalnlen = qrynposits_ + qrynstrs_ * (querylen + 1);

        //alignment beginning position:
        int alnbeg = 
            offsetalns_ + //offset to the alignments produced for the current query complex
            nTDP2OutputAlignmentSSS *
            (qrychnndx * qrynposits_ + qrynstrs_ * (qrystrprtlen + qrychnndx)) + //offset regarding #query chains
            dbstrdst + dbstrndx * (querylen + 1);//alignment position for dbstrndx within the query chain section

        //beginning of the alignment:
        const char* palnbeg = GetBegOfAlns() + alnbeg;
        const char* p;

        for(int f = 0; f < (int)alnlen; f += width, palnbeg += width)
        {
            nbytes = alnlen - f;
            if(width < nbytes) nbytes = width;
            if(1) {
                p = GetAlnSectionAt(palnbeg, dp2oaQuerySSS, dbalnlen);
                written = sprintf(outptr,"%-13s","struct");
                outptr += written;
                strncpy(outptr, p, nbytes);
                outptr += nbytes;
                PutNL(outptr);
            }
            {   p = GetAlnSectionAt(palnbeg, dp2oaQuery, dbalnlen);
                written = sprintf(outptr,"Query: %5u ", qrybeg);
                outptr += written;
                strncpy(outptr, p, nbytes);
                outptr += nbytes;
                fgaps = (int)(std::count(p, p+nbytes, '-'));
                qrybeg += nbytes - fgaps;
                written = sprintf(outptr," %-5d%s", nbytes<=fgaps? qrybeg: qrybeg-1,NL);
                outptr += written;
            }
            {   p = GetAlnSectionAt(palnbeg, dp2oaMiddle, dbalnlen);
                written = sprintf(outptr,"%13c",' ');
                outptr += written;
                strncpy(outptr, p, nbytes);
                outptr += nbytes;
                PutNL(outptr);
            }
            {   p = GetAlnSectionAt(palnbeg, dp2oaTarget, dbalnlen);
                if(nodeletions && f)
                    written = sprintf(outptr,"Refn.: %5s ", "...");
                else
                    written = sprintf(outptr,"Refn.: %5u ", trgbeg);
                outptr += written;
                strncpy(outptr, p, nbytes);
                outptr += nbytes;
                fgaps = (int)(std::count(p, p+nbytes, '-'));
                trgbeg += nbytes - fgaps;
                if(nodeletions) {
                    if((int)alnlen <= f + width)
                        written = sprintf(outptr," %-5d%s", trgend, NL);
                    else written = sprintf(outptr," %-5s%s", "...", NL);
                } else
                    written = sprintf(outptr," %-5d%s", nbytes<=fgaps? trgbeg: trgbeg-1,NL);
                outptr += written;
            }
            if(1) {
                p = GetAlnSectionAt(palnbeg, dp2oaTargetSSS, dbalnlen);
                written = sprintf(outptr,"%-13s","struct");
                outptr += written;
                strncpy(outptr, p, nbytes);
                outptr += nbytes;
                PutNL(outptr);
            }
            PutNL(outptr);
        }
    }
}



// -------------------------------------------------------------------------
// GetSizeOfCompressedResultsPlainComplex: get total size required for
// annotations and complete alignments; using plain format;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// rfncpxDstN, distance in chains to the current reference complex;
// szalns, size of complete alignments (with descriptions);
//
inline
void TdFinalizer::GetSizeOfCompressedResultsPlainComplex(
    int qrycpxN, int rfncpxN, int rfncpxDstN,
    size_t* szalns) const
{
    MYMSG("TdFinalizer::GetSizeOfCompressedResultsPlainComplex", 5);
    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    static const unsigned int sznl = (int)strlen(NL);
    static const unsigned int indent = OUTPUTINDENT;
    const unsigned int alnwidth = CLOptions::GetO_WRAP();
    static const unsigned int headlines = 3 + (bsectmscore==1);//#lines for scores, etc.
    int alnsize;//alignment size

    *szalns = 0UL;

    //on the first receive of results, they can be empty if 
    // there are no hits found
    if(!cubp_set_h_results_) return;

    for(int qrychnndx = 0; qrychnndx < qrycpxN; qrychnndx++)
        for(int dbchnndx = 0; dbchnndx < rfncpxN; dbchnndx++)
        {
            const int dbstrndx = rfncpxDstN + dbchnndx;

            //chain assignment:
            float chnassg = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadOrgStrNo);
            if(chnassg == 0.0f) continue;

            // float tmscoreq = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadScoreQ_C);
            // float tmscorer = GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadScoreR_C);
            // float tmscore = mymax(tmscoreq, tmscorer);
            // if(tmscore < cubp_set_scorethld_) continue;

            unsigned int alnlen = (unsigned int)
                GetOutputAlnDataFieldChain<float>(qrychnndx, dbstrndx, dp2oadAlnLength);
            //NOTE: alignment length can be 0 due to pre-screening
            if(alnlen < 1) continue;

            unsigned int dbstr2dst = GetDbStructureField<unsigned int>(dbstrndx, pps2DDist);
            unsigned int dbstrlen = (unsigned int)GetDbStructureField<INTYPE>(dbstrndx, pps2DLen);
            //if true, the structure has not been processed due to memory restrictions
            if(qrynposits_ < dbstr2dst + dbstrlen) continue;

            unsigned int varwidth = alnlen < alnwidth? alnlen: alnwidth;
            //size of the alignment section:
            int alnfrags = (alnlen + alnwidth - 1)/alnwidth;
            int alnlines = nTDP2OutputAlignmentSSS;

            alnsize = headlines * (fpc_maxlinelen+sznl) + 2 * sznl;
            alnsize += alnfrags * alnlines * (varwidth + 2 * indent + sznl) + (alnfrags + 1) * sznl;
            alnsize += 2 * sznl;

            *szalns += fpc_maxlinelen + sznl;
            *szalns += alnsize;
        }

    //heading for short annotation:
    if(*szalns)
        *szalns += 2 * fpc_maxlinelen + 2 * sznl;

    MYMSGBEGl(5)
        char msgbuf[KBYTE];
        sprintf(msgbuf,
            "TdFinalizer::GetSizeOfCompressedResultsPlainComplex: "
            "szalns %zu", *szalns);
        MYMSG(msgbuf, 5);
    MYMSGENDl
}
