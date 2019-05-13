"""A work in progress pipeline to combine (g)VCFs into an alternate format"""

import hail as hl
from hail import MatrixTable, Table
from hail.expr import StructExpression
from hail.expr.expressions import expr_call, expr_array, expr_int32
from hail.ir import Apply, TableKeyBy, TableMapRows, TopLevelReference
from hail.typecheck import typecheck

_transform_rows_function_map = {}
_merge_function_map = {}

def localize(mt):
    if isinstance(mt, MatrixTable):
        return mt._localize_entries('__entries', '__cols')
    return mt

def unlocalize(mt):
    if isinstance(mt, Table):
        return mt._unlocalize_entries('__entries', '__cols', ['s'])
    return mt

# vardp_outlier is needed due to a mistake in generating gVCFs
def transform_one(mt, vardp_outlier=100_000) -> Table:
    """transforms a gvcf into a form suitable for combining

    The input to this should be some result of either :func:`.import_vcf` or
    :func:`.import_vcfs` with `array_elements_required=False`.

    There is a strong assumption that this function will be called on a matrix
    table with one column.
    """
    mt = localize(mt)
    if mt.row.dtype not in _transform_rows_function_map:
        f = hl.experimental.define_function(
            lambda row: hl.rbind(
                hl.len(row.alleles), '<NON_REF>' == row.alleles[-1],
                lambda alleles_len, has_non_ref: hl.struct(
                    locus=row.locus,
                    alleles=hl.cond(has_non_ref, row.alleles[:-1], row.alleles),
                    rsid=row.rsid,
                    __entries=row.__entries.map(
                        lambda e:
                        hl.struct(
                            DP=e.DP,
                            END=row.info.END,
                            GQ=e.GQ,
                            LA=hl.range(0, alleles_len - hl.cond(has_non_ref, 1, 0)),
                            LAD=hl.cond(has_non_ref, e.AD[:-1], e.AD),
                            LGT=e.GT,
                            LPGT=e.PGT,
                            LPL=hl.cond(has_non_ref,
                                        hl.cond(alleles_len > 2,
                                                e.PL[:-alleles_len],
                                                hl.null(e.PL.dtype)),
                                        hl.cond(alleles_len > 1,
                                                e.PL,
                                                hl.null(e.PL.dtype))),
                            MIN_DP=e.MIN_DP,
                            PID=e.PID,
                            RGQ=hl.cond(
                                has_non_ref,
                                e.PL[hl.call(0, alleles_len - 1).unphased_diploid_gt_index()],
                                hl.null(e.PL.dtype.element_type)),
                            SB=e.SB,
                            gvcf_info=hl.case()
                                .when(hl.is_missing(row.info.END),
                                      hl.struct(
                                          ClippingRankSum=row.info.ClippingRankSum,
                                          BaseQRankSum=row.info.BaseQRankSum,
                                          MQ=row.info.MQ,
                                          MQRankSum=row.info.MQRankSum,
                                          MQ_DP=row.info.MQ_DP,
                                          QUALapprox=row.info.QUALapprox,
                                          RAW_MQ=row.info.RAW_MQ,
                                          ReadPosRankSum=row.info.ReadPosRankSum,
                                          VarDP=hl.cond(row.info.VarDP > vardp_outlier,
                                                        row.info.DP, row.info.VarDP)))
                                .or_missing()
                        ))),
            ),
            mt.row.dtype)
        _transform_rows_function_map[mt.row.dtype] = f
    transform_row = _transform_rows_function_map[mt.row.dtype]
    return Table(TableMapRows(mt._tir, Apply(transform_row._name, TopLevelReference('row'))))

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
                    globl=hl.array([ref]).extend(hl.array(hl.set(hl.flatten(lal)).remove(ref))),
                    local=lal)))

    def renumber_entry(entry, old_to_new) -> StructExpression:
        # global index of alternate (non-ref) alleles
        return entry.annotate(LA=entry.LA.map(lambda lak: old_to_new[lak]))

    if (ts.row.dtype, ts.globals.dtype) not in _merge_function_map:
        f = hl.experimental.define_function(
            lambda row, gbl:
            hl.rbind(
                merge_alleles(row.data.map(lambda d: d.alleles)),
                lambda alleles:
                hl.struct(
                    locus=row.locus,
                    alleles=alleles.globl,
                    rsid=hl.find(hl.is_defined, row.data.map(lambda d: d.rsid)),
                    __entries=hl.bind(
                        lambda combined_allele_index:
                        hl.range(0, hl.len(row.data)).flatmap(
                            lambda i:
                            hl.cond(hl.is_missing(row.data[i].__entries),
                                    hl.range(0, hl.len(gbl.g[i].__cols))
                                      .map(lambda _: hl.null(row.data[i].__entries.dtype.element_type)),
                                    hl.bind(
                                        lambda old_to_new: row.data[i].__entries.map(
                                            lambda e: renumber_entry(e, old_to_new)),
                                        hl.range(0, hl.len(alleles.local[i])).map(
                                            lambda j: combined_allele_index[alleles.local[i][j]])))),
                        hl.dict(hl.range(0, hl.len(alleles.globl)).map(
                            lambda j: hl.tuple([alleles.globl[j], j])))))),
            ts.row.dtype, ts.globals.dtype)
        _merge_function_map[(ts.row.dtype, ts.globals.dtype)] = f
    merge_function = _merge_function_map[(ts.row.dtype, ts.globals.dtype)]
    ts = Table(TableMapRows(ts._tir, Apply(merge_function._name,
                                           TopLevelReference('row'),
                                           TopLevelReference('global'))))
    return ts.transmute_globals(__cols=hl.flatten(ts.g.map(lambda g: g.__cols)))

def combine_gvcfs(mts):
    """merges vcfs using multi way join"""
    ts = hl.Table._multi_way_zip_join([localize(mt) for mt in mts], 'data', 'g')
    combined = combine(ts)
    return unlocalize(combined)

@typecheck(lgt=expr_call, la=expr_array(expr_int32))
def lgt_to_gt(lgt, la):
    """A method for transforming Local GT and Local Alleles into the true GT"""
    return hl.call(la[lgt[0]], la[lgt[1]])

def quick_summary(mt):
    """compute aggregate INFO fields that do not require densify"""
    return mt.annotate_rows(
        info=hl.struct(
            MQ_DP=hl.agg.sum(mt.entry.gvcf_info.MQ_DP),
            QUALapprox=hl.agg.sum(mt.entry.gvcf_info.QUALapprox),
            RAW_MQ=hl.agg.sum(mt.entry.gvcf_info.RAW_MQ),
            VarDP=hl.agg.sum(mt.entry.gvcf_info.VarDP),
            SB_TABLE=hl.array([
                hl.agg.sum(mt.entry.SB[0]),
                hl.agg.sum(mt.entry.SB[1]),
                hl.agg.sum(mt.entry.SB[2]),
                hl.agg.sum(mt.entry.SB[3]),
            ])))

def summarize(mt):
    """Computes summary statistics

    Calls :func:`.quick_summary`. Calling both this and :func:`.quick_summary`, will lead
    to :func:`.quick_summary` being executed twice.

    Note
    ----
    You will not be able to run :func:`.combine_gvcfs` with the output of this
    function.
    """
    mt = quick_summary(mt)
    mt = hl.experimental.densify(mt)
    return mt.annotate_rows(info=hl.rbind(
        hl.agg.call_stats(lgt_to_gt(mt.LGT, mt.LA), mt.alleles),
        lambda gs: hl.struct(
            # here, we alphabetize the INFO fields by GATK convention
            AC=gs.AC[1:],  # The VCF spec indicates that AC and AF have Number=A, so we need
            AF=gs.AF[1:],  # to drop the first element from each of these.
            AN=gs.AN,
            BaseQRankSum=hl.median(hl.agg.collect(mt.entry.gvcf_info.BaseQRankSum)),
            ClippingRankSum=hl.median(hl.agg.collect(mt.entry.gvcf_info.ClippingRankSum)),
            DP=hl.agg.sum(mt.entry.DP),
            MQ=hl.median(hl.agg.collect(mt.entry.gvcf_info.MQ)),
            MQRankSum=hl.median(hl.agg.collect(mt.entry.gvcf_info.MQRankSum)),
            MQ_DP=mt.info.MQ_DP,
            QUALapprox=mt.info.QUALapprox,
            RAW_MQ=mt.info.RAW_MQ,
            ReadPosRankSum=hl.median(hl.agg.collect(mt.entry.gvcf_info.ReadPosRankSum)),
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
    return mt.drop('gvcf_info')

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
# As of 2019-03-25, the schema returned by the combiner is as follows:
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
# ----------------------------------------
# Entry fields:
#     'DP': int32
#     'END': int32
#     'GQ': int32
#     'LA': array<int32>
#     'LAD': array<int32>
#     'LGT': call
#     'LPGT': call
#     'LPL': array<int32>
#     'MIN_DP': int32
#     'PID': str
#     'RGQ': int32
#     'SB': array<int32>
#     'gvcf_info': struct {
#         ClippingRankSum: float64,
#         BaseQRankSum: float64,
#         MQ: float64,
#         MQRankSum: float64,
#         MQ_DP: int32,
#         QUALapprox: int32,
#         RAW_MQ: float64,
#         ReadPosRankSum: float64,
#         VarDP: int32
#     }
# ----------------------------------------
# Column key: ['s']
# Row key: ['locus']
# ----------------------------------------
