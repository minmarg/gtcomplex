/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
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
// CompressResultsJSONComplex: process results for complexes; compress 
// these results for passing them to the writing thread; use JSON format;
// qrycpxnr, query complex number relative to the current chunk;
// NOTE: all operations performed under lock
//
void TdFinalizer::CompressResultsJSONComplex(
    int qrycpxnr, int qrycpxN, int qrycpxDstN)
{
    MYMSG( "TdFinalizer::CompressResultsJSONComplex", 4 );
    // static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    static const int complexes = (CLOptions::GetB_CHAINS() == 0);
    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    static const int sortby = CLOptions::GetO_SORT();
    static const unsigned int sznl = (int)strlen(NL);
    static const unsigned int opencloselines = 2;//number of lines for opening/closing record sections
    static const unsigned int anntopencloselines = 2;//#lines for annotation opening/closing section(s)
    static const unsigned int anntheadlines = 10;//!lines for statistics in annotation, including trunc. description
    static const unsigned int headlines = 15;//number of lines for complex alignment statistics
    static const unsigned int footlines = 3;//number of lines for tfm and related data
    size_t szannot = 0UL;
    size_t szcpxhead = 0UL, szalns = 0UL;
    size_t szalnswodesc = 0UL;
    const char* desc;//structure description
    int written, sernr = 0;//initialize serial number here
    char errbuf[BUF_MAX];

    if(!cubp_set_querypmbeg_[0] || !cubp_set_querypmend_[0] || !cubp_set_querydesc_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSONComplex: Null query data.");

    if(cubp_set_bdbCpmbeg_[0] && cubp_set_bdbCpmend_[0] && !cubp_set_bdbCdesc_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSONComplex: Null structure descriptions.");

    int rc = 0, ri = 0;

    for(rc = 0, ri = 0; ri < qrynstrs_; rc++) 
    {
        int rfncpxN = GetDbStructureField<INTYPE>(rc, pcx2DN);
        int rfncpxDstN = (int)GetDbStructureField<LNTYPE>(rc, pcx2DDstN);
        // int rfncpxNend = rfncpxN + rfncpxDstN;

        if(rfncpxDstN != ri) {
            sprintf(errbuf, "TdFinalizer::CompressResultsJSONComplex: "
                "Inconsistent reference chain no. %d: %d", ri, rfncpxDstN);
            throw MYRUNTIME_ERROR(errbuf);
        }

        if(rfncpxN < 1) {
            sprintf(errbuf, "TdFinalizer::CompressResultsJSONComplex: "
                "Invalid #chains for relative reference complex no. %d (%d): %d", 
                rc, ri, rfncpxN);
            throw MYRUNTIME_ERROR(errbuf);
        }

        if(qrynstrs_ < ri + rfncpxN) {
            sprintf(errbuf, "TdFinalizer::CompressResultsJSONComplex: "
                "#chains for relative reference complex no. %d (%d) > "
                "#total chains (%d): %d", 
                rc, ri, qrynstrs_, rfncpxN);
            throw MYRUNTIME_ERROR(errbuf);
        }

        GetDbStructureDesc(desc, rfncpxDstN/*strndx*/);

        size_t cpxdesclen = mymin((size_t)dsclen, (strlen(desc) + 2));

        szcpxhead += (cpxdesclen + fpc_json_maxfieldlen + sznl + 2);//2 quotes
        szcpxhead += opencloselines * (fpc_json_maxopencloselinelen + sznl);
        szcpxhead += headlines * (fpc_json_maxlinelen + sznl);
        szcpxhead += footlines * (3 * fpc_json_maxlinelen + sznl);

        size_t szalnswodesc1 = 0UL;

        GetSizeOfCompressedResultsJSONComplex(qrycpxN, rfncpxN, rfncpxDstN, &szalnswodesc1);

        szannot += anntopencloselines * (fpc_json_maxopencloselinelen + sznl);
        szannot += anntheadlines * (fpc_json_maxlinelen + sznl);
        szalnswodesc += szalnswodesc1;

        ri += rfncpxN;
    }

    szalns = szcpxhead + szalnswodesc;

    if(cubp_set_qrysernrbeg_ < 0 || qrysernr_ < cubp_set_qrysernrbeg_ || 
       cubp_set_qrysernrbeg_ + (int)cubp_set_nqystrs_ <= qrysernr_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSONComplex: Invalid query indices.");

    annotations_.reset();
    alignments_.reset();

    ReserveVectors(rc);

    if(szalns < szannot || 
       szalnswodesc > 10 * 
            (cubp_set_sz_alndata_ + cubp_set_sz_tfmmatrices_ + cubp_set_sz_alns_))
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSONComplex: "
        "Size of formatted results is unusually large.");

    FormatQueryInfoJSONComplex(qrycpxnr, qrycpxN, qrycpxDstN);

    if(szannot < 1 || szalns < 1) return;

    annotations_.reset((char*)std::malloc(szannot));
    alignments_.reset((char*)std::malloc(szalns));

    if(!annotations_ || !alignments_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSONComplex: Not enough memory.");

    if(!srtindxs_ || !scores_ || !alnptrs_ || !annotptrs_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSONComplex: Not enough memory.");

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
        MakeAnnotationJSONComplex( 
            annptr, qrycpxnr, rc, qrycpxN, rfncpxN,
            descopy.c_str(), alnlen, psts, idts, gaps, nchns, dbstrlen);
        *annptr++ = 0;//end of record

        //compress the alignment and relative information...
        written = sprintf(outptr,
                "    {\"hit_record\": {%s"
                "      \"reference_description\": \"",NL);
        outptr += written;

        //put the description...
        int outpos = 0;
        FormatDescriptionJSON(outptr, descopy.c_str(), descopy.size(), dsclen, outpos);
        written = sprintf(outptr,"\",%s",NL);
        outptr += written;

        FormatScoresJSONComplex(
            outptr, qrycpxnr, rc, qrycpxN, rfncpxN,
            alnlen, psts, idts, gaps, qrystrlen_, dbstrlen);

        FormatFooterJSONComplex(outptr, qrycpxnr, rc);

        FormatAlignmentJSONComplex(
            outptr, qrycpxnr, rc, qrycpxN, rfncpxN, qrycpxDstN, rfncpxDstN);

        written = sprintf(outptr,"    }}");//,%s",NL);
        outptr += written;
        *outptr++ = 0;//end of record
    }
}

// -------------------------------------------------------------------------
// FormatQueryInfoJSONComplex, format query general information;
// qrycpxndx, query complex index;
// qrycpxN, #chains of the query complex;
// qrycpxDstN, distance in chains to the complex;
inline
void TdFinalizer::FormatQueryInfoJSONComplex(
    const int /*qrycpxndx*/, const int qrycpxN, const int qrycpxDstN)
{
    static const char* pattrn = " Chn:";
    static const size_t lenpattrn = strlen(pattrn);
    char locbuf[BUF_MAX];

    qryinfo_.clear();

    sprintf(locbuf,"    \"chain_list\": [%s",NL);
    qryinfo_ += locbuf;

    for(int qrychnndx = 0; qrychnndx < qrycpxN; qrychnndx++)
    {
        const int qrystrndx = qrycpxDstN + qrychnndx;
        unsigned int querylen = (unsigned int)GetQueryField<INTYPE>(qrystrndx, pps2DLen);
        unsigned int qrystrdst = (unsigned int)GetQueryField<INTYPE>(qrystrndx, pps2DDist);
        int qrysttype = (int)GetQueryField<INTYPE>(qrystrdst, pmv2D_Ins_Ch_Ord);
        qrysttype = GetMoleculeType(qrysttype);
        const char* dbsttypestr = GetMoleculeTypeStr(qrysttype);
        const char* qdesc;
        std::string qchnstr;// = "Q_Chn:";
        std::string::size_type posc, poss;
        GetQueryDesc(qdesc, qrystrndx);
        tmp_summary_ = qdesc;
        if((posc = tmp_summary_.rfind(pattrn)) != std::string::npos) {
            size_t nmaxch = 5;
            if((poss = tmp_summary_.find_first_of(' ', posc + 1)) != std::string::npos)
                nmaxch = mymin(nmaxch, poss - posc - lenpattrn);
            qchnstr += tmp_summary_.substr(posc + lenpattrn, nmaxch);
        }

        sprintf(locbuf,
                "      {\"chain_details\": {%s"
                "        \"id\": \"%s\",%s"
                "        \"length\": %d,%s"
                "        \"type\": \"%s\"%s"
                "      }}",
            NL,qchnstr.c_str(),NL,querylen,NL,dbsttypestr,NL);
        qryinfo_ += locbuf;

        sprintf(locbuf, "%s%s", (qrychnndx + 1 < qrycpxN)? ",": "", NL);
        qryinfo_ += locbuf;
    }

    sprintf(locbuf,"    ]%s",NL);
    qryinfo_ += locbuf;
}

// -------------------------------------------------------------------------
// MakeAnnotationJSONComplex: format complex description in JSON format;
// NOTE: space is assumed to be pre-allocated;
// outptr, pointer to the output buffer;
// qrycpxndx, rfncpxndx, query and reference complex indices;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// desc, reference complex description;
// alnlen, psts, idts, gaps, nchns, numbers of alignment symbols, positives, 
// identities, gaps,and aligned chains;
// dbcpxlen, reference complex length;
inline
void TdFinalizer::MakeAnnotationJSONComplex( 
    char*& outptr,
    int qrycpxndx, int rfncpxndx, int /*qrycpxN*/, int rfncpxN,
    const char* desc, const unsigned int alnlen,
    const int psts, const int idts, const int gaps, const int /*nchns*/,
    const int dbcpxlen) const
{
    const unsigned int anndesclen = ANNOTATION_DESCLEN;
    float rmsd = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadRMSD);
    float tmscoreq = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadScoreQ);
    float tmscorer = GetOutputAlnDataFieldComplex<float>(qrycpxndx, rfncpxndx, dp2oadScoreR);
    int desclen = strlen(desc);
    int written, outpos = 0;

    written = sprintf(outptr,
                "    {\"summary_entry\": {%s"
                "      \"description\": \"",NL);
    outptr += written;

    FormatDescriptionJSON(outptr, desc, desclen, anndesclen, outpos);

    written = sprintf(outptr,"\",%s",NL);
    outptr += written;

    written = sprintf(outptr,
                "      \"tmscore_refn\": %.4f,%s"
                "      \"tmscore_query\": %.4f,%s"
                "      \"rmsd\": %.2f,%s"
                "      \"n_aligned_residues\": %d,%s"
                "      \"n_identities\": %d,%s"
                "      \"%%_identities\": %d,%s"
                "      \"n_matched\": %d,%s"
                "      \"n_refrn_chains\": %d,%s"
                "      \"refrn_length\": %d%s"
                "    }}",
        tmscorer,NL, tmscoreq,NL, rmsd,NL,
        (int)alnlen - gaps,NL, idts,NL, idts * 100 / alnlen,NL, psts,NL, rfncpxN,NL,
        dbcpxlen,NL);
    outptr += written;
}

// -------------------------------------------------------------------------
// FormatScoresJSONComplex: format complex scores in JSON format;
// outptr, pointer to the output buffer;
// qrycpxndx, rfncpxndx, query and reference complex indices;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// palnlen, ppsts, pidts, pgaps, numbers of alignment symbols, positives, 
// identities, and gaps;
// querylen, query length;
// dbstrlen, reference complex length;
inline
void TdFinalizer::FormatScoresJSONComplex(
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

    static const char* na = "NA";
    char tm2qbuf[BUF_MAX/2], tm2rbuf[BUF_MAX/2];
    const char* tm2q = na, *tm2r = na;

    if(bsectmscore) {
        sprintf(tm2qbuf,"%.5g",sectmscoreq); tm2q = tm2qbuf;
        sprintf(tm2rbuf,"%.5g",sectmscorer); tm2r = tm2rbuf;
    }

    written = sprintf(outptr,
            "      \"query_length\": %d,%s"
            "      \"reference_length\": %d,%s"
            "      \"n_query_chains\": %d,%s"
            "      \"n_refrn_chains\": %d,%s"
            "      \"tmscore_refrn\": %.5f,%s"
            "      \"tmscore_query\": %.5f,%s"
            "      \"d0_refrn\": %.2f,%s"
            "      \"d0_query\": %.2f,%s"
            "      \"rmsd\": %.2f,%s"
            "      \"2tmscore_refrn\": \"%s\",%s"
            "      \"2tmscore_query\": \"%s\",%s"
            "      \"n_identities\": %d,%s"
            "      \"n_matched\": %d,%s"
            "      \"n_gaps\": %d,%s"
            "      \"n_aligned\": %d,%s",
            querylen,NL,dbstrlen,NL,qrycpxN,NL,rfncpxN,NL,
            tmscorer,NL,tmscoreq,NL,d0r,NL,d0q,NL,rmsd,NL,
            tm2r,NL,tm2q,NL,
            idts,NL,psts,NL,gaps,NL,alnlen,NL);
    outptr += written;
}

// -------------------------------------------------------------------------
// FormatFooterJSONComplex, format tfm in JSON format;
// outptr, pointer to the output buffer;
// qrycpxndx, rfncpxndx, query and reference complex indices;
inline
void TdFinalizer::FormatFooterJSONComplex(
    char*& outptr, int qrycpxndx, int rfncpxndx)
{
    static const int referenced = CLOptions::GetO_REFERENCED();
    //address of the relevant transformation matrix data:
    float* ptfmmtx = GetOutputTfmMtxAddressComplex(qrycpxndx, rfncpxndx);
    int written;

    written = sprintf(outptr,
            "      \"tfm_referenced\": %d,%s",referenced,NL);
    outptr += written;
    written = sprintf(outptr,
            "      \"rotation_matrix_rowmajor\": ["
            "%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f],%s",
        ptfmmtx[tfmmRot_0_0],ptfmmtx[tfmmRot_0_1],ptfmmtx[tfmmRot_0_2],
        ptfmmtx[tfmmRot_1_0],ptfmmtx[tfmmRot_1_1],ptfmmtx[tfmmRot_1_2],
        ptfmmtx[tfmmRot_2_0],ptfmmtx[tfmmRot_2_1],ptfmmtx[tfmmRot_2_2],
        NL);
    outptr += written;
    written = sprintf(outptr,
            "      \"translation_vector\": [%.6f, %.6f, %.6f],%s",
        ptfmmtx[tfmmTrl_0],ptfmmtx[tfmmTrl_1],ptfmmtx[tfmmTrl_2],NL);
    outptr += written;
}

// -------------------------------------------------------------------------
// FormatAlignmentJSONComplex, format chain alignments;
// outptr, pointer to the output buffer;
// qrycpxndx, rfncpxndx, query and reference complex indices;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// qrycpxDstN, rfncpxDstN, distance in chains to the current complexes;
inline
void TdFinalizer::FormatAlignmentJSONComplex(
    char*& outptr,
    const int qrycpxndx, const int /*rfncpxndx*/,
    const int qrycpxN, const int rfncpxN,
    const int qrycpxDstN, const int rfncpxDstN)
{
    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();
    static const int sortby = CLOptions::GetO_SORT();
    static const char* pattrn = " Chn:";
    static const char* na = "NA";
    static const size_t lenpattrn = strlen(pattrn);
    char locbuf[TIMES2(BUF_MAX)];
    char tm2qbuf[BUF_MAX/2], tm2rbuf[BUF_MAX/2];
    const char* tm2q = na, *tm2r = na;
    int written;

    // int nminchns = mymin(qrycpxN, rfncpxN);

    tmp_c2csorted_.clear();
    tmp_c2cscores_.clear();

    for(int qrychnndx = 0; qrychnndx < qrycpxN; qrychnndx++)
        for(int dbchnndx = 0; dbchnndx < rfncpxN; dbchnndx++)
        {
            const int dbstrndx = rfncpxDstN + dbchnndx;

            //chain assignment:
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

    written = sprintf(outptr,"      \"assignment_table\": [%s",NL);
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
                    "        {\"assignment_entry\": {%s"
                    "          \"number\": %d,%s"
                    "          \"refrn_chain\": \"%s\",%s"
                    "          \"query_chain\": \"%s\",%s"
                    "          \"tmscore_refrn\": %.4f,%s"
                    "          \"tmscore_query\": %.4f,%s"
                    "          \"rmsd\": %.2f,%s"
                    "          \"n_aligned_residues\": %d,%s"
                    "          \"query_chain_length\": %d,%s"
                    "          \"query_chain_from\": %d,%s"
                    "          \"query_chain_to\": %d,%s"
                    "          \"refrn_chain_from\": %d,%s"
                    "          \"refrn_chain_to\": %d,%s"
                    "          \"refrn_chain_length\": %d%s"
                    "        }}",
            NL,++i,NL,rchnstr.c_str(),NL,qchnstr.c_str(),NL,  tmscorer,NL,tmscoreq,NL,rmsd,NL,
            (int)alnlen - gaps,NL, querylen,NL,qrybeg,NL,qryend,NL, trgbeg,NL,trgend,NL,dbstrlen,NL);
        outptr += written;

        written = sprintf(outptr, "%s%s", ((size_t)i < tmp_c2csorted_.size())? ",": "", NL);
        outptr += written;

        tmp_summary_ = rdesc;
        if((posc = tmp_summary_.rfind(DIRSEPSTR)) != std::string::npos)
            tmp_summary_ = tmp_summary_.substr(posc + 1);
        if(ANNOTATION_DESCLEN < tmp_summary_.size()) {
            tmp_summary_ = tmp_summary_.substr(tmp_summary_.size() - ANNOTATION_DESCLEN);
            tmp_summary_[0] = tmp_summary_[1] = tmp_summary_[2] = '.';
        }

        tm2q = na; tm2r = na;

        if(bsectmscore) {
            sprintf(tm2qbuf,"%.5g",sectmscoreq); tm2q = tm2qbuf;
            sprintf(tm2rbuf,"%.5g",sectmscorer); tm2r = tm2rbuf;
        }

        sprintf(locbuf,
            "        {\"chain_alignment_entry\": {%s"
            "          \"number\": %d,%s"
            "          \"refrn_name\": \"%s\",%s"
            "          \"refrn_chain\": \"%s\",%s"
            "          \"query_chain\": \"%s\",%s",
            NL,i,NL,tmp_summary_.c_str(),NL,rchnstr.c_str(),NL,qchnstr.c_str(),NL);
        tmp_summary_.clear();
        tmp_summary_ += locbuf;
        sprintf(locbuf,
            "          \"query_chain_length\": %d,%s"
            "          \"refrn_chain_length\": %d,%s"
            "          \"alignment_type\": \"%s\",%s"
            "          \"alignment\": {%s",
            querylen,NL,dbstrlen,NL,dbsttypestr,NL,NL);
        tmp_summary_ += locbuf;
        sprintf(locbuf,
            "            \"tmscore_refrn\": %.5f,%s"
            "            \"tmscore_query\": %.5f,%s"
            "            \"d0_refrn\": %.2f,%s"
            "            \"d0_query\": %.2f,%s"
            "            \"rmsd\": %.2f,%s",
            tmscorer,NL,tmscoreq,NL,d0r,NL,d0q,NL,rmsd,NL);
        tmp_summary_ += locbuf;
        sprintf(locbuf,
            "            \"2tmscore_refrn\": \"%s\",%s"
            "            \"2tmscore_query\": \"%s\",%s",
            tm2r,NL,tm2q,NL);
        tmp_summary_ += locbuf;
        sprintf(locbuf,
            "            \"n_identities\": %d,%s"
            "            \"n_matched\": %d,%s"
            "            \"n_gaps\": %d,%s"
            "            \"n_aligned\": %d,%s",
            idts,NL,psts,NL,gaps,NL,alnlen,NL);
        tmp_summary_ += locbuf;

        tmp_c2cscores_.push_back(
            std::make_tuple(qrychnndx, dbchnndx, tmp_summary_, alnlen, alnbegcoords, alnendcoords));
    }

    written = sprintf(outptr,
        "      ],%s"
        "      \"alignment_section\": [%s",NL,NL);
    outptr += written;
    i = 0;

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
        const char* pquerysss = GetAlnSectionAt(palnbeg, dp2oaQuerySSS, dbalnlen);
        const char* pqueryaln = GetAlnSectionAt(palnbeg, dp2oaQuery, dbalnlen);
        const char* prefnaln = GetAlnSectionAt(palnbeg, dp2oaTarget, dbalnlen);
        const char* prefnsss = GetAlnSectionAt(palnbeg, dp2oaTargetSSS, dbalnlen);
        const char* pmiddle = GetAlnSectionAt(palnbeg, dp2oaMiddle, dbalnlen);

        int nbytes = alnlen;
        int ngapsqry = (int)(std::count(pqueryaln, pqueryaln + nbytes, '-'));//#gaps in query
        int ngapsrfn = (int)(std::count(prefnaln, prefnaln + nbytes, '-'));//#gaps in reference

        unsigned int qryend = (unsigned int)(qrybeg + nbytes - ngapsqry);
        unsigned int rfnend = (unsigned int)(nodeletions? trgend: (trgbeg + nbytes - ngapsrfn));

        written = sprintf(outptr,
                "            \"query_chain_from\": %u,%s"
                "            \"query_chain_to\": %u,%s"
                "            \"refrn_chain_from\": %u,%s"
                "            \"refrn_chain_to\": %u,%s",
                qrybeg,NL,(nbytes <= ngapsqry)? qryend: qryend - 1,NL,
                trgbeg,NL,(nbytes <= ngapsrfn)? rfnend: rfnend - 1,NL);
        outptr += written;

        written = sprintf(outptr,"            \"query_chain_secstr\": \"");
        outptr += written;
        strncpy(outptr, pquerysss, nbytes);
        outptr += nbytes;
        written = sprintf(outptr,"\",%s            \"refrn_chain_secstr\": \"",NL);
        outptr += written;
        strncpy(outptr, prefnsss, nbytes);
        outptr += nbytes;
        written = sprintf(outptr,"\",%s            \"query_chain_aln\": \"",NL);
        outptr += written;
        strncpy(outptr, pqueryaln, nbytes);
        outptr += nbytes;
        written = sprintf(outptr,"\",%s            \"refrn_chain_aln\": \"",NL);
        outptr += written;
        strncpy(outptr, prefnaln, nbytes);
        outptr += nbytes;
        written = sprintf(outptr,"\",%s            \"middle\": \"",NL);
        outptr += written;
        strncpy(outptr, pmiddle, nbytes);
        outptr += nbytes;
        written = sprintf(outptr,"\"%s          }%s        }}%s%s",
            NL,NL,((size_t)(++i) < tmp_c2cscores_.size())? ",": "",NL);
        outptr += written;
    }

    written = sprintf(outptr,"      ]%s",NL);
    outptr += written;
}



// -------------------------------------------------------------------------
// GetSizeOfCompressedResultsJSONComplex: get total size required for
// annotations and complete alignments; using JSON format;
// qrycpxN, rfncpxN, #chains of respective query and reference complexes;
// rfncpxDstN, distance in chains to the current reference complex;
// szalns, size of complete alignments (with descriptions);
//
inline
void TdFinalizer::GetSizeOfCompressedResultsJSONComplex(
    int qrycpxN, int rfncpxN, int rfncpxDstN,
    size_t* szalns) const
{
    MYMSG("TdFinalizer::GetSizeOfCompressedResultsJSONComplex", 5);
    static const unsigned int sznl = (int)strlen(NL);
    static const unsigned int opencloselines = 4;//number of lines for opening/closing record sections
    static const unsigned int assgnheadlines = 15;//number of lines for assignment, including open/close
    static const unsigned int alnheadlines = 25;//number of lines for scores & other statistics, including closing
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

            unsigned int varwidth = alnlen;
            //size of the alignment section:
            int alnlines = nTDP2OutputAlignmentSSS;

            alnsize = opencloselines * (fpc_json_maxopencloselinelen + sznl);
            alnsize += assgnheadlines * (fpc_json_maxlinelen + sznl);
            alnsize += alnheadlines * (fpc_json_maxlinelen + sznl);
            alnsize += alnlines * (varwidth + fpc_json_maxfieldlen + sznl + 2) + 2 * (fpc_json_maxfieldlen + sznl + 2);//2 qts.

            // *szalns += fpc_json_maxlinelen + fpc_json_maxfieldlen + sznl + 2;//+2 (quotes)
            *szalns += alnsize;
        }

    MYMSGBEGl(5)
        char msgbuf[KBYTE];
        sprintf(msgbuf,
            "TdFinalizer::GetSizeOfCompressedResultsJSONComplex: "
            "szalns %zu", *szalns);
        MYMSG(msgbuf, 5);
    MYMSGENDl
}
