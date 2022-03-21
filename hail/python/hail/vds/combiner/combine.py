from typing import Collection, List, Optional, Set

import hail as hl
from hail import MatrixTable, Table
from hail.ir import Apply, TableMapRows, TopLevelReference
from hail.experimental.vcf_combiner.vcf_combiner import combine_gvcfs, localize, parse_as_fields, unlocalize
from ..variant_dataset import VariantDataset

_transform_variant_function_map = {}
_transform_reference_fuction_map = {}


def make_variants_matrix_table(mt: MatrixTable,
                               info_to_keep: Optional[Collection[str]] = None
                               ) -> MatrixTable:
    if info_to_keep is None:
        info_to_keep = []
    if not info_to_keep:
        info_to_keep = [name for name in mt.info if name not in ['END', 'DP']]
    info_key = tuple(sorted(info_to_keep))  # hashable stable value
    mt = localize(mt)
    mt = mt.filter(hl.is_missing(mt.info.END))

    if (mt.row.dtype, info_key) not in _transform_variant_function_map:
        def get_lgt(e, n_alleles, has_non_ref, row):
            index = e.GT.unphased_diploid_gt_index()
            n_no_nonref = n_alleles - hl.int(has_non_ref)
            triangle_without_nonref = hl.triangle(n_no_nonref)
            return (hl.case()
                    .when(e.GT.is_haploid(),
                          hl.or_missing(e.GT[0] < n_no_nonref, e.GT))
                    .when(index < triangle_without_nonref, e.GT)
                    .when(index < hl.triangle(n_alleles), hl.missing('call'))
                    .or_error('invalid GT ' + hl.str(e.GT) + ' at site ' + hl.str(row.locus)))

        def make_entry_struct(e, alleles_len, has_non_ref, row):
            handled_fields = dict()
            handled_names = {'LA', 'gvcf_info',
                             'LAD', 'AD',
                             'LGT', 'GT',
                             'LPL', 'PL',
                             'LPGT', 'PGT'}

            if 'GT' not in e:
                raise hl.utils.FatalError("the Hail GVCF combiner expects GVCFs to have a 'GT' field in FORMAT.")

            handled_fields['LA'] = hl.range(0, alleles_len - hl.if_else(has_non_ref, 1, 0))
            handled_fields['LGT'] = get_lgt(e, alleles_len, has_non_ref, row)
            if 'AD' in e:
                handled_fields['LAD'] = hl.if_else(has_non_ref, e.AD[:-1], e.AD)
            if 'PGT' in e:
                handled_fields['LPGT'] = e.PGT
            if 'PL' in e:
                handled_fields['LPL'] = hl.if_else(has_non_ref,
                                                   hl.if_else(alleles_len > 2,
                                                              e.PL[:-alleles_len],
                                                              hl.missing(e.PL.dtype)),
                                                   hl.if_else(alleles_len > 1,
                                                              e.PL,
                                                              hl.missing(e.PL.dtype)))
                handled_fields['RGQ'] = hl.if_else(
                    has_non_ref,
                    hl.if_else(e.GT.is_haploid(),
                               e.PL[alleles_len - 1],
                               e.PL[hl.call(0, alleles_len - 1).unphased_diploid_gt_index()]),
                    hl.missing(e.PL.dtype.element_type))

            handled_fields['gvcf_info'] = (hl.case()
                                           .when(hl.is_missing(row.info.END),
                                                 hl.struct(**(
                                                     parse_as_fields(
                                                         row.info.select(*info_to_keep),
                                                         has_non_ref)
                                                 )))
                                           .or_missing())

            pass_through_fields = {k: v for k, v in e.items() if k not in handled_names}
            return hl.struct(**handled_fields, **pass_through_fields)

        f = hl.experimental.define_function(
            lambda row: hl.rbind(
                hl.len(row.alleles), '<NON_REF>' == row.alleles[-1],
                lambda alleles_len, has_non_ref: hl.struct(
                    locus=row.locus,
                    alleles=hl.if_else(has_non_ref, row.alleles[:-1], row.alleles),
                    rsid=row.rsid,
                    __entries=row.__entries.map(
                        lambda e: make_entry_struct(e, alleles_len, has_non_ref, row)))),
            mt.row.dtype)
        _transform_variant_function_map[mt.row.dtype, info_key] = f
    transform_row = _transform_variant_function_map[mt.row.dtype, info_key]
    return unlocalize(Table(TableMapRows(mt._tir, Apply(transform_row._name, transform_row._ret_type, TopLevelReference('row')))))


def defined_entry_fields(mt: MatrixTable, sample=None) -> Set[str]:
    if sample is not None:
        mt = mt.head(sample)
    used = mt.aggregate_entries(hl.struct(**{
        k: hl.agg.any(hl.is_defined(v)) for k, v in mt.entry.items()
    }))
    return set(k for k in mt.entry if used[k])


def make_reference_matrix_table(mt: MatrixTable,
                                entry_to_keep: Collection[str]
                                ) -> MatrixTable:
    mt = mt.filter_rows(hl.is_defined(mt.info.END))
    entry_key = tuple(sorted(entry_to_keep))  # hashable stable value

    def make_entry_struct(e, row):
        handled_fields = dict()
        # we drop PL by default, but if `entry_to_keep` has it then PL needs to be
        # turned into LPL
        handled_names = {'AD', 'PL'}

        if 'AD' in entry_to_keep:
            handled_fields['LAD'] = e['AD'][:1]
        if 'PL' in entry_to_keep:
            handled_fields['LPL'] = e['PL'][:1]

        reference_fields = {k: v for k, v in e.items()
                            if k in entry_to_keep and k not in handled_names}
        return (hl.case()
                  .when(e.GT.is_hom_ref(),
                        hl.struct(END=row.info.END, **reference_fields, **handled_fields))
                  .or_error('found END with non reference-genotype at' + hl.str(row.locus)))

    mt = localize(mt)
    if (mt.row.dtype, entry_key) not in _transform_reference_fuction_map:
        f = hl.experimental.define_function(
            lambda row: hl.struct(
                locus=row.locus,
                ref_allele=row.alleles[0][0],
                __entries=row.__entries.map(
                    lambda e: make_entry_struct(e, row))),
            mt.row.dtype)
        _transform_reference_fuction_map[mt.row.dtype, entry_key] = f

    transform_row = _transform_reference_fuction_map[mt.row.dtype, entry_key]
    return unlocalize(Table(TableMapRows(mt._tir, Apply(transform_row._name, transform_row._ret_type, TopLevelReference('row')))))


def transform_gvcf(mt: MatrixTable,
                   reference_entry_fields_to_keep: Collection[str],
                   info_to_keep: Optional[Collection[str]] = None) -> VariantDataset:
    """Transforms a gvcf into a sparse matrix table

    The input to this should be some result of either :func:`.import_vcf` or
    :func:`.import_gvcfs` with ``array_elements_required=False``.

    There is an assumption that this function will be called on a matrix table
    with one column (or a localized table version of the same).

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        The gvcf being transformed.
    reference_entry_fields_to_keep : :class:`list` of :class:`str`
        Genotype fields to keep in the reference table. If empty, the first
        10,000 reference block rows of ``mt`` will be sampled and all fields
        found to be defined other than ``GT``, ``AD``, and ``PL`` will be entry
        fields in the resulting reference matrix in the dataset.
    info_to_keep : :class:`list` of :class:`str`
        Any ``INFO`` fields in the gvcf that are to be kept and put in the ``gvcf_info`` entry
        field. By default, all ``INFO`` fields except ``END`` and ``DP`` are kept.

    Returns
    -------
    :obj:`.Table`
        A localized matrix table that can be used as part of the input to `combine_gvcfs`

    Notes
    -----
    This function will parse the following allele specific annotations from
    pipe delimited strings into proper values. ::

        AS_QUALapprox
        AS_RAW_MQ
        AS_RAW_MQRankSum
        AS_RAW_ReadPosRankSum
        AS_SB_TABLE
        AS_VarDP

    """
    ref_mt = make_reference_matrix_table(mt, reference_entry_fields_to_keep)
    var_mt = make_variants_matrix_table(mt, info_to_keep)
    return VariantDataset(ref_mt, var_mt._key_rows_by_assert_sorted('locus', 'alleles'))


_merge_function_map = {}


def combine_r(ts):
    if (ts.row.dtype, ts.globals.dtype) not in _merge_function_map:
        f = hl.experimental.define_function(
            lambda row, gbl:
            hl.struct(
                locus=row.locus,
                ref_allele=hl.find(hl.is_defined, row.data.map(lambda d: d.ref_allele)),
                __entries=hl.range(0, hl.len(row.data)).flatmap(
                    lambda i:
                    hl.if_else(hl.is_missing(row.data[i]),
                               hl.range(0, hl.len(gbl.g[i].__cols))
                               .map(lambda _: hl.missing(row.data[i].__entries.dtype.element_type)),
                               row.data[i].__entries))),
            ts.row.dtype, ts.globals.dtype)
        _merge_function_map[(ts.row.dtype, ts.globals.dtype)] = f
    merge_function = _merge_function_map[(ts.row.dtype, ts.globals.dtype)]
    ts = Table(TableMapRows(ts._tir, Apply(merge_function._name,
                                           merge_function._ret_type,
                                           TopLevelReference('row'),
                                           TopLevelReference('global'))))
    return ts.transmute_globals(__cols=hl.flatten(ts.g.map(lambda g: g.__cols)))


def combine_references(mts: List[MatrixTable]) -> MatrixTable:
    ts = hl.Table.multi_way_zip_join([localize(mt) for mt in mts], 'data', 'g')
    combined = combine_r(ts)
    return unlocalize(combined)


def combine_variant_datasets(vdss: List[VariantDataset]) -> VariantDataset:
    reference = combine_references([vds.reference_data for vds in vdss])
    no_variant_key = [vds.variant_data.key_rows_by('locus') for vds in vdss]

    variants = combine_gvcfs(no_variant_key)
    return VariantDataset(reference, variants._key_rows_by_assert_sorted('locus', 'alleles'))
