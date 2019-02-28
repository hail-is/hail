"""A work in progress pipeline to combine (g)VCFs into an alternate format"""

import hail as hl
from hail.matrixtable import MatrixTable
from hail.expr import StructExpression
from hail.expr.expressions import expr_call, expr_array, expr_int32
from hail.ir import TableKeyBy
from hail.typecheck import typecheck


def transform_one(mt: MatrixTable) -> MatrixTable:
    """transforms a gvcf into a form suitable for combining

    The input to this should be some result of either :func:`.import_vcf` or
    :func:`.import_vcfs`.  Be aware of requiredness issues surrounding `LAD` and `LPL` in the
    output of this function and the output of :func:`.combine_gvcfs`. The output of
    :func:`.combine_gvcfs` will have `LAD` and `LPL` elements as not-required. Without the
    `array_elements_required` flag being set to `True` in :func:`.import_vcf` and
    :func:`.import_vcfs`, the LAD and LPL fields' elements in the output of this function will
    be required which will lead to a type error if the output of this function and the output
    of :func:`.combine_gvcfs` are passed to :func:`.combine_gvcfs`.
    """
    mt = mt.annotate_entries(
        # local (alt) allele index into global (alt) alleles
        LA=hl.range(0, hl.len(mt.alleles)),
        END=mt.info.END,
        BaseQRankSum=mt.info['BaseQRankSum'],
        ClippingRankSum=mt.info['ClippingRankSum'],
        MQ=mt.info['MQ'],
        MQRankSum=mt.info['MQRankSum'],
        ReadPosRankSum=mt.info['ReadPosRankSum'],
        LGT=mt.GT,
        LAD=mt.AD,
        LPL=mt.PL,
        LPGT=mt.PGT
    )
    mt = mt.annotate_rows(
        info=mt.info.annotate(
            SB_TABLE=hl.array([
                hl.agg.sum(mt.entry.SB[0]),
                hl.agg.sum(mt.entry.SB[1]),
                hl.agg.sum(mt.entry.SB[2]),
                hl.agg.sum(mt.entry.SB[3]),
            ])
        ).select(
            "MQ_DP",
            "QUALapprox",
            "RAW_MQ",
            "VarDP",
            "SB_TABLE",
        ))
    mt = mt.drop('SB', 'qual', 'filters', 'GT', 'AD', 'PL', 'PGT')

    return mt

def combine(ts):
    def merge_alleles(alleles):
        from hail.expr.functions import _num_allele_type, _allele_ints
        return hl.rbind(
            alleles.map(lambda a: hl.or_else(a[0], ''))
                   .fold(lambda s, t: hl.cond(hl.len(s) > hl.len(t), s, t), ''),
            lambda ref:
            hl.rbind(
                alleles.map(
                    lambda al: hl.rbind(
                        al[0],
                        lambda r:
                        hl.array([ref]).extend(
                            al[1:].map(
                                lambda a:
                                hl.rbind(
                                    _num_allele_type(r, a),
                                    lambda at:
                                    hl.cond(
                                        (_allele_ints['SNP'] == at) |
                                        (_allele_ints['Insertion'] == at) |
                                        (_allele_ints['Deletion'] == at) |
                                        (_allele_ints['MNP'] == at) |
                                        (_allele_ints['Complex'] == at),
                                        a + ref[hl.len(r):],
                                        a)))))),
                lambda lal:
                hl.struct(
                    globl=hl.array([ref]).extend(hl.rbind(
                        hl.array(hl.set(hl.flatten(lal)).remove(ref)),
                        lambda arr:
                        hl.filter(lambda a: a != '<NON_REF>', arr)
                        .extend(hl.filter(lambda a: a == '<NON_REF>', arr)))),
                    local=lal)))

    def renumber_entry(entry, old_to_new) -> StructExpression:
        # global index of alternate (non-ref) alleles
        return entry.annotate(LA=entry.LA.map(lambda lak: old_to_new[lak]))


    # pylint: disable=protected-access
    tmp = ts.annotate(
        alleles=merge_alleles(ts.data.map(lambda d: d.alleles)),
        rsid=hl.find(hl.is_defined, ts.data.map(lambda d: d.rsid)),
        info=hl.struct(
            MQ_DP=hl.sum(ts.data.map(lambda d: d.info.MQ_DP)),
            QUALapprox=hl.sum(ts.data.map(lambda d: d.info.QUALapprox)),
            RAW_MQ=hl.sum(ts.data.map(lambda d: d.info.RAW_MQ)),
            VarDP=hl.sum(ts.data.map(lambda d: d.info.VarDP)),
            SB_TABLE=hl.array([
                hl.sum(ts.data.map(lambda d: d.info.SB_TABLE[0])),
                hl.sum(ts.data.map(lambda d: d.info.SB_TABLE[1])),
                hl.sum(ts.data.map(lambda d: d.info.SB_TABLE[2])),
                hl.sum(ts.data.map(lambda d: d.info.SB_TABLE[3]))
            ])))
    tmp = tmp.annotate(
        __entries=hl.bind(
            lambda combined_allele_index:
            hl.range(0, hl.len(tmp.data)).flatmap(
                lambda i:
                hl.cond(hl.is_missing(tmp.data[i].__entries),
                        hl.range(0, hl.len(tmp.g[i].__cols))
                          .map(lambda _: hl.null(tmp.data[i].__entries.dtype.element_type)),
                        hl.bind(
                            lambda old_to_new: tmp.data[i].__entries.map(lambda e: renumber_entry(e, old_to_new)),
                            hl.range(0, hl.len(tmp.alleles.local[i])).map(
                                lambda j: combined_allele_index[tmp.alleles.local[i][j]])))),
            hl.dict(hl.range(0, hl.len(tmp.alleles.globl)).map(
                lambda j: hl.tuple([tmp.alleles.globl[j], j])))))
    tmp = tmp.annotate(alleles=tmp.alleles.globl)
    tmp = tmp.annotate_globals(__cols=hl.flatten(tmp.g.map(lambda g: g.__cols)))

    return tmp.drop('data', 'g')


def combine_gvcfs(mts):
    """merges vcfs using multi way join"""

    # pylint: disable=protected-access
    def localize(mt):
        return mt._localize_entries('__entries', '__cols')

    ts = hl.Table._multi_way_zip_join([localize(mt) for mt in mts], 'data', 'g')
    combined = combine(ts)
    return combined._unlocalize_entries('__entries', '__cols', ['s'])


@typecheck(lgt=expr_call, la=expr_array(expr_int32))
def lgt_to_gt(lgt, la):
    """A method for transforming Local GT and Local Alleles into the true GT"""
    return hl.call(la[lgt[0]], la[lgt[1]])


def summarize(mt):
    """Computes summary statistics

    Note
    ----
    You will not be able to run :func:`.combine_gvcfs` with the output of this
    function.
    """
    mt = hl.experimental.densify(mt)
    return mt.annotate_rows(info=hl.rbind(
        hl.agg.call_stats(lgt_to_gt(mt.LGT, mt.LA), mt.alleles),
        lambda gs: hl.struct(
            # here, we alphabetize the INFO fields by GATK convention
            AC=gs.AC,
            AF=gs.AF,
            AN=gs.AN,
            BaseQRankSum=hl.median(hl.agg.collect(mt.entry.BaseQRankSum)),
            ClippingRankSum=hl.median(hl.agg.collect(mt.entry.ClippingRankSum)),
            DP=hl.agg.sum(mt.entry.DP),
            MQ=hl.median(hl.agg.collect(mt.entry.MQ)),
            MQRankSum=hl.median(hl.agg.collect(mt.entry.MQRankSum)),
            MQ_DP=mt.info.MQ_DP,
            QUALapprox=mt.info.QUALapprox,
            RAW_MQ=mt.info.RAW_MQ,
            ReadPosRankSum=hl.median(hl.agg.collect(mt.entry.ReadPosRankSum)),
            SB_TABLE=mt.info.SB_TABLE,
            VarDP=mt.info.VarDP,
        )))

def finalize(mt):
    """Drops entry fields no longer needed for combining.

    Note
    ----
    You will not be able to run :func:`.combine_gvcfs` with the output of this
    function.
    """
    return mt.drop('BaseQRankSum', 'ClippingRankSum', 'MQ', 'MQRankSum', 'ReadPosRankSum')


def reannotate(mt, gatk_ht, summ_ht):
    """Re-annotate a sparse MT with annotations from certain GATK tools

    `gatk_ht` should be a table from the rows of a VCF, with `info` having at least
    the following fields.  Be aware that fields not present in this list will
    be dropped.
    ```
        struct {
            AC: array<int32>,
            AF: array<float64>,
            AN: int32,
            BaseQRankSum: float64,
            ClippingRankSum: float64,
            DP: int32,
            FS: float64,
            MQ: float64,
            MQRankSum: float64,
            MQ_DP: int32,
            NEGATIVE_TRAIN_SITE: bool,
            POSITIVE_TRAIN_SITE: bool,
            QD: float64,
            QUALapprox: int32,
            RAW_MQ: float64,
            ReadPosRankSum: float64,
            SB_TABLE: array<int32>,
            SOR: float64,
            VQSLOD: float64,
            VarDP: int32,
            culprit: str
        }
    ```
    `summarize_ht` should be the output of :func:`.summarize` as a rows table.

    Note
    ----
    You will not be able to run :func:`.combine_gvcfs` with the output of this
    function.
    """
    def check(ht):
        keys = list(ht.key)
        if keys[0] != 'locus':
            raise TypeError(f'table inputs must have first key "locus", found {keys}')
        if keys != ['locus']:
            return hl.Table(TableKeyBy(ht._tir, ['locus'], is_sorted=True))
        return ht

    gatk_ht, summ_ht = [check(ht) for ht in (gatk_ht, summ_ht)]
    return mt.annotate_rows(
        info=hl.rbind(
            gatk_ht[mt.locus].info, summ_ht[mt.locus].info,
            lambda ginfo, hinfo: hl.struct(
                AC=hl.or_else(hinfo.AC, ginfo.AC),
                AF=hl.or_else(hinfo.AF, ginfo.AF),
                AN=hl.or_else(hinfo.AN, ginfo.AN),
                BaseQRankSum=hl.or_else(hinfo.BaseQRankSum, ginfo.BaseQRankSum),
                ClippingRankSum=hl.or_else(hinfo.ClippingRankSum, ginfo.ClippingRankSum),
                DP=hl.or_else(hinfo.DP, ginfo.DP),
                FS=ginfo.FS,
                MQ=hl.or_else(hinfo.MQ, ginfo.MQ),
                MQRankSum=hl.or_else(hinfo.MQRankSum, ginfo.MQRankSum),
                MQ_DP=hl.or_else(hinfo.MQ_DP, ginfo.MQ_DP),
                NEGATIVE_TRAIN_SITE=ginfo.NEGATIVE_TRAIN_SITE,
                POSITIVE_TRAIN_SITE=ginfo.POSITIVE_TRAIN_SITE,
                QD=ginfo.QD,
                QUALapprox=hl.or_else(hinfo.QUALapprox, ginfo.QUALapprox),
                RAW_MQ=hl.or_else(hinfo.RAW_MQ, ginfo.RAW_MQ),
                ReadPosRankSum=hl.or_else(hinfo.ReadPosRankSum, ginfo.ReadPosRankSum),
                SB_TABLE=hl.or_else(hinfo.SB_TABLE, ginfo.SB_TABLE),
                SOR=ginfo.SOR,
                VQSLOD=ginfo.VQSLOD,
                VarDP=hl.or_else(hinfo.VarDP, ginfo.VarDP),
                culprit=ginfo.culprit,
            )),
        qual=gatk_ht[mt.locus].qual,
        filters=gatk_ht[mt.locus].filters,
    )


# NOTE: these are just @chrisvittal's notes on how gVCF fields are combined
#       some of it is copied from GenomicsDB's wiki.
# always missing items include MQ, HaplotypeScore, InbreedingCoeff
# items that are dropped by CombineGVCFs and so set to missing are MLEAC, MLEAF
# Notes on info aggregation, The GenomicsDB wiki says the following:
#   The following operations are supported:
#       "sum" sum over valid inputs
#       "mean"
#       "median"
#       "element_wise_sum"
#       "concatenate"
#       "move_to_FORMAT"
#       "combine_histogram"
#
#   Operations for the fields
#   QUAL: set to missing
#   INFO {
#       BaseQRankSum: median, # NOTE : move to format for combine
#       ClippingRankSum: median, # NOTE : move to format for combine
#       DP: sum
#       ExcessHet: median, # NOTE : this can also be dropped
#       MQ: median, # NOTE : move to format for combine
#       MQ_DP: sum,
#       MQ0: median,
#       MQRankSum: median, # NOTE : move to format for combine
#       QUALApprox: sum,
#       RAW_MQ: sum
#       ReadPosRankSum: median, # NOTE : move to format for combine
#       SB_TABLE: elementwise sum, # NOTE: after being moved from FORMAT as SB
#       VarDP: sum
#   }
#   FORMAT {
#       END: move from INFO
#   }
#
# The following are Truncated INFO fields for the specific VCFs this tool targets
# ##INFO=<ID=BaseQRankSum,Number=1,Type=Float>
# ##INFO=<ID=ClippingRankSum,Number=1,Type=Float>
# ##INFO=<ID=DP,Number=1,Type=Integer>
# ##INFO=<ID=END,Number=1,Type=Integer>
# ##INFO=<ID=ExcessHet,Number=1,Type=Float>
# ##INFO=<ID=MQ,Number=1,Type=Float>
# ##INFO=<ID=MQRankSum,Number=1,Type=Float>
# ##INFO=<ID=MQ_DP,Number=1,Type=Integer>
# ##INFO=<ID=QUALapprox,Number=1,Type=Integer>
# ##INFO=<ID=RAW_MQ,Number=1,Type=Float>
# ##INFO=<ID=ReadPosRankSum,Number=1,Type=Float>
# ##INFO=<ID=VarDP,Number=1,Type=Integer>
#
# As of 2/15/19, the schema returned by the combiner is as follows:
# ----------------------------------------
# Global fields:
#     None
# ----------------------------------------
# Column fields:
#     's': str
# ----------------------------------------
# Row fields:
#     'locus': locus<GRCh38>
#     'alleles': array<str>
#     'rsid': str
#     'info': struct {
#         MQ_DP: int32,
#         QUALapprox: int32,
#         RAW_MQ: float64,
#         VarDP: int32,
#         SB_TABLE: array<int64>
#     }
# ----------------------------------------
# Entry fields:
#     'LAD': array<int32>
#     'DP': int32
#     'GQ': int32
#     'LGT': call
#     'MIN_DP': int32
#     'LPGT': call
#     'PID': str
#     'LPL': array<int32>
#     'LA': array<int32>
#     'END': int32
#     'BaseQRankSum': float64
#     'ClippingRankSum': float64
#     'MQ': float64
#     'MQRankSum': float64
#     'ReadPosRankSum': float64
# ----------------------------------------
# Column key: ['s']
# Row key: ['locus', 'alleles']
# ----------------------------------------
