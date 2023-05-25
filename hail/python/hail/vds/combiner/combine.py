import math
from typing import Collection, Optional, Set
from typing import List, Tuple, Dict

import hail as hl
from hail import MatrixTable, Table
from hail.experimental.function import Function
from hail.expr import StructExpression, unify_all, construct_expr
from hail.expr.expressions import expr_bool, expr_str
from hail.genetics.reference_genome import reference_genome_type
from hail.ir import Apply, TableMapRows
from hail.typecheck import oneof, sequenceof, typecheck
from ..variant_dataset import VariantDataset

_transform_variant_function_map: Dict[Tuple[hl.HailType, Tuple[str, ...]], Function] = {}
_transform_reference_fuction_map: Dict[Tuple[hl.HailType, Tuple[str, ...]], Function] = {}
_merge_function_map: Dict[Tuple[hl.HailType, hl.HailType], Function] = {}


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

    transform_row = _transform_variant_function_map.get((mt.row.dtype, info_key))
    if transform_row is None or not hl.current_backend()._is_registered_ir_function_name(transform_row._name):
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

        transform_row = hl.experimental.define_function(
            lambda row: hl.rbind(
                hl.len(row.alleles), '<NON_REF>' == row.alleles[-1],
                lambda alleles_len, has_non_ref: hl.struct(
                    locus=row.locus,
                    alleles=hl.if_else(has_non_ref, row.alleles[:-1], row.alleles),
                    **({'rsid': row.rsid} if 'rsid' in row else {}),
                    __entries=row.__entries.map(
                        lambda e: make_entry_struct(e, alleles_len, has_non_ref, row)))),
            mt.row.dtype)
        _transform_variant_function_map[mt.row.dtype, info_key] = transform_row
    return unlocalize(Table(TableMapRows(mt._tir, Apply(transform_row._name, transform_row._ret_type, mt.row._ir))))


def defined_entry_fields(mt: MatrixTable, sample=None) -> Set[str]:
    if sample is not None:
        mt = mt.head(sample)
    used = mt.aggregate_entries(hl.struct(**{
        k: hl.agg.any(hl.is_defined(v)) for k, v in mt.entry.items()
    }))
    return set(k for k in mt.entry if used[k])


def make_reference_stream(stream, entry_to_keep: Collection[str]):
    stream = stream.filter(lambda elt: hl.is_defined(elt.info.END))
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

    row_type = stream.dtype.element_type
    transform_row = _transform_reference_fuction_map.get((row_type, entry_key))
    if transform_row is None or not hl.current_backend()._is_registered_ir_function_name(transform_row._name):
        transform_row = hl.experimental.define_function(
            lambda row: hl.struct(
                locus=row.locus,
                __entries=row.__entries.map(
                    lambda e: make_entry_struct(e, row))),
            row_type)
        _transform_reference_fuction_map[row_type, entry_key] = transform_row

    return stream.map(lambda row: hl.struct(
        locus=row.locus,
        __entries=row.__entries.map(
            lambda e: make_entry_struct(e, row))))


def make_variant_stream(stream, info_to_keep):
    info_t = stream.dtype.element_type['info']
    if info_to_keep is None:
        info_to_keep = []
    if not info_to_keep:
        info_to_keep = [name for name in info_t if name not in ['END', 'DP']]
    info_key = tuple(sorted(info_to_keep))  # hashable stable value
    stream = stream.filter(lambda elt: hl.is_missing(elt.info.END))

    row_type = stream.dtype.element_type

    transform_row = _transform_variant_function_map.get((row_type, info_key))
    if transform_row is None or not hl.current_backend()._is_registered_ir_function_name(transform_row._name):
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

        transform_row = hl.experimental.define_function(
            lambda row: hl.rbind(
                hl.len(row.alleles), '<NON_REF>' == row.alleles[-1],
                lambda alleles_len, has_non_ref: hl.struct(
                    locus=row.locus,
                    alleles=hl.if_else(has_non_ref, row.alleles[:-1], row.alleles),
                    **({'rsid': row.rsid} if 'rsid' in row else {}),
                    __entries=row.__entries.map(
                        lambda e: make_entry_struct(e, alleles_len, has_non_ref, row)))),
            row_type)
        _transform_variant_function_map[row_type, info_key] = transform_row

    from hail.expr import construct_expr
    from hail.utils.java import Env
    uid = Env.get_uid()
    map_ir = hl.ir.ToArray(hl.ir.StreamMap(hl.ir.ToStream(stream._ir), uid,
                                           Apply(transform_row._name, transform_row._ret_type,
                                                 hl.ir.Ref(uid, type=row_type))))
    return construct_expr(map_ir, map_ir.typ, stream._indices, stream._aggregations)


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

    mt = localize(mt).key_by('locus')
    transform_row = _transform_reference_fuction_map.get((mt.row.dtype, entry_key))
    if transform_row is None or not hl.current_backend()._is_registered_ir_function_name(transform_row._name):
        transform_row = hl.experimental.define_function(
            lambda row: hl.struct(
                locus=row.locus,
                __entries=row.__entries.map(
                    lambda e: make_entry_struct(e, row))),
            mt.row.dtype)
        _transform_reference_fuction_map[mt.row.dtype, entry_key] = transform_row

    return unlocalize(Table(TableMapRows(mt._tir, Apply(transform_row._name, transform_row._ret_type, mt.row._ir))))


def transform_gvcf(mt: MatrixTable,
                   reference_entry_fields_to_keep: Collection[str],
                   info_to_keep: Optional[Collection[str]] = None) -> VariantDataset:
    """Transforms a GVCF into a sparse matrix table

    The input to this should be some result of either :func:`.import_vcf` or
    :func:`.import_gvcfs` with ``array_elements_required=False``.

    There is an assumption that this function will be called on a matrix table
    with one column (or a localized table version of the same).

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        The GVCF being transformed.
    reference_entry_fields_to_keep : :class:`list` of :class:`str`
        Genotype fields to keep in the reference table. If empty, the first
        10,000 reference block rows of ``mt`` will be sampled and all fields
        found to be defined other than ``GT``, ``AD``, and ``PL`` will be entry
        fields in the resulting reference matrix in the dataset.
    info_to_keep : :class:`list` of :class:`str`
        Any ``INFO`` fields in the GVCF that are to be kept and put in the ``gvcf_info`` entry
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


def combine_r(ts, ref_block_max_len_field):
    ts = Table(TableMapRows(ts._tir, combine_reference_row(ts.row, ts.globals)._ir))

    global_fds = {'__cols': hl.flatten(ts.g.map(lambda g: g.__cols))}
    if ref_block_max_len_field is not None:
        global_fds[ref_block_max_len_field] = hl.max(ts.g.map(lambda g: g[ref_block_max_len_field]))
    return ts.transmute_globals(**global_fds)


def combine_reference_row(row, globals):
    merge_function = _merge_function_map.get((row.dtype, globals))
    if merge_function is None or not hl.current_backend()._is_registered_ir_function_name(merge_function._name):
        merge_function = hl.experimental.define_function(
            lambda row, gbl:
            hl.struct(
                locus=row.locus,
                __entries=hl.range(0, hl.len(row.data)).flatmap(
                    lambda i:
                    hl.if_else(hl.is_missing(row.data[i]),
                               hl.range(0, hl.len(gbl.g[i].__cols))
                               .map(lambda _: hl.missing(row.data[i].__entries.dtype.element_type)),
                               row.data[i].__entries))),
            row.dtype, globals.dtype)
        _merge_function_map[(row.dtype, globals.dtype)] = merge_function
    apply_ir = Apply(merge_function._name,
                     merge_function._ret_type,
                     row._ir,
                     globals._ir)
    indices, aggs = unify_all(row, globals)
    return construct_expr(apply_ir, apply_ir.typ, indices, aggs)


def combine_references(mts: List[MatrixTable]) -> MatrixTable:
    fd = hl.vds.VariantDataset.ref_block_max_length_field
    n_with_ref_max_len = len([mt for mt in mts if fd in mt.globals])
    any_ref_max = n_with_ref_max_len > 0
    all_ref_max = n_with_ref_max_len == len(mts)

    # if some mts have max ref len but not all, drop it
    if any_ref_max and not all_ref_max:
        mts = [mt.drop(fd) if fd in mt.globals else mt for mt in mts]

    mts = [mt.drop('ref_allele') if 'ref_allele' in mt.row else mt for mt in mts]

    ts = hl.Table.multi_way_zip_join([localize(mt) for mt in mts], 'data', 'g')
    combined = combine_r(ts, fd if all_ref_max else None)
    return unlocalize(combined)


def combine_variant_datasets(vdss: List[VariantDataset]) -> VariantDataset:
    reference = combine_references([vds.reference_data for vds in vdss])
    no_variant_key = [vds.variant_data.key_rows_by('locus') for vds in vdss]

    variants = combine_gvcfs(no_variant_key)
    return VariantDataset(reference, variants._key_rows_by_assert_sorted('locus', 'alleles'))


_transform_rows_function_map: Dict[Tuple[hl.HailType], Function] = {}
_merge_function_map: Dict[Tuple[hl.HailType, hl.HailType], Function] = {}


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_ints(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.if_else(has_non_ref, ints[:-1], ints)
    return ints.map(lambda i: hl.if_else((hl.len(i) == 0) | (i == '.'), hl.missing(hl.tint32), hl.int32(i)))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_doubles(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.if_else(has_non_ref, ints[:-1], ints)
    return ints.map(lambda i: hl.if_else((hl.len(i) == 0) | (i == '.'), hl.missing(hl.tfloat64), hl.float64(i)))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_sb_table(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.if_else(has_non_ref, ints[:-1], ints)
    return ints.map(lambda xs: xs.split(",").map(hl.int32))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_ranksum(string, has_non_ref):
    typ = hl.ttuple(hl.tfloat64, hl.tint32)
    items = string.split(r'\|')
    items = hl.if_else(has_non_ref, items[:-1], items)
    return items.map(lambda s: hl.if_else(
        (hl.len(s) == 0) | (s == '.'),
        hl.missing(typ),
        hl.rbind(s.split(','), lambda ss: hl.if_else(
            hl.len(ss) != 2,  # bad field, possibly 'NaN', just set it null
            hl.missing(hl.ttuple(hl.tfloat64, hl.tint32)),
            hl.tuple([hl.float64(ss[0]), hl.int32(ss[1])])))))


_as_function_map = {
    'AS_QUALapprox': parse_as_ints,
    'AS_RAW_MQ': parse_as_doubles,
    'AS_RAW_MQRankSum': parse_as_ranksum,
    'AS_RAW_ReadPosRankSum': parse_as_ranksum,
    'AS_SB_TABLE': parse_as_sb_table,
    'AS_VarDP': parse_as_ints,
}


def parse_as_fields(info, has_non_ref):
    return hl.struct(**{f: info[f] if f not in _as_function_map
    else _as_function_map[f](info[f], has_non_ref) for f in info})


def localize(mt):
    if isinstance(mt, MatrixTable):
        return mt._localize_entries('__entries', '__cols')
    return mt


def unlocalize(mt):
    if isinstance(mt, Table):
        return mt._unlocalize_entries('__entries', '__cols', ['s'])
    return mt


def merge_alleles(alleles):
    from hail.expr.functions import _num_allele_type, _allele_ints
    return hl.rbind(
        alleles.map(lambda a: hl.or_else(a[0], ''))
        .fold(lambda s, t: hl.if_else(hl.len(s) > hl.len(t), s, t), ''),
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
                                hl.if_else(
                                    (_allele_ints['SNP'] == at)
                                    | (_allele_ints['Insertion'] == at)
                                    | (_allele_ints['Deletion'] == at)
                                    | (_allele_ints['MNP'] == at)
                                    | (_allele_ints['Complex'] == at),
                                    a + ref[hl.len(r):],
                                    a)))))),
            lambda lal:
            hl.struct(
                globl=hl.array([ref]).extend(hl.array(hl.set(hl.flatten(lal)).remove(ref))),
                local=lal)))


def combine_variant_rows(row, globals):
    def renumber_entry(entry, old_to_new) -> StructExpression:
        # global index of alternate (non-ref) alleles
        return entry.annotate(LA=entry.LA.map(lambda lak: old_to_new[lak]))

    merge_function = _merge_function_map.get((row.dtype, globals.dtype))
    if merge_function is None or not hl.current_backend()._is_registered_ir_function_name(merge_function._name):
        merge_function = hl.experimental.define_function(
            lambda row, gbl:
            hl.rbind(
                merge_alleles(row.data.map(lambda d: d.alleles)),
                lambda alleles:
                hl.struct(
                    locus=row.locus,
                    alleles=alleles.globl,
                    **({'rsid': hl.find(hl.is_defined, row.data.map(
                        lambda d: d.rsid))} if 'rsid' in row.data.dtype.element_type else {}),
                    __entries=hl.bind(
                        lambda combined_allele_index:
                        hl.range(0, hl.len(row.data)).flatmap(
                            lambda i:
                            hl.if_else(hl.is_missing(row.data[i].__entries),
                                       hl.range(0, hl.len(gbl.g[i].__cols))
                                       .map(lambda _: hl.missing(row.data[i].__entries.dtype.element_type)),
                                       hl.bind(
                                           lambda old_to_new: row.data[i].__entries.map(
                                               lambda e: renumber_entry(e, old_to_new)),
                                           hl.range(0, hl.len(alleles.local[i])).map(
                                               lambda j: combined_allele_index[alleles.local[i][j]])))),
                        hl.dict(hl.range(0, hl.len(alleles.globl)).map(
                            lambda j: hl.tuple([alleles.globl[j], j])))))),
            row.dtype, globals.dtype)
        _merge_function_map[(row.dtype, globals.dtype)] = merge_function
    indices, aggs = unify_all(row, globals)
    apply_ir = Apply(merge_function._name,
                     merge_function._ret_type,
                     row._ir,
                     globals._ir)
    return construct_expr(apply_ir, apply_ir.typ, indices, aggs)


def combine(ts):
    ts = Table(TableMapRows(ts._tir, combine_variant_rows(
        ts.row,
        ts.globals)._ir))
    return ts.transmute_globals(__cols=hl.flatten(ts.g.map(lambda g: g.__cols)))


@typecheck(mts=sequenceof(oneof(Table, MatrixTable)))
def combine_gvcfs(mts):
    """Merges gvcfs and/or sparse matrix tables

    Parameters
    ----------
    mts : :obj:`List[Union[Table, MatrixTable]]`
        The matrix tables (or localized versions) to combine

    Returns
    -------
    :class:`.MatrixTable`

    Notes
    -----
    All of the input tables/matrix tables must have the same partitioning. This
    module provides no method of repartitioning data.
    """
    ts = hl.Table.multi_way_zip_join([localize(mt) for mt in mts], 'data', 'g')
    combined = combine(ts)
    return unlocalize(combined)


@typecheck(mt=hl.MatrixTable, desired_average_partition_size=int, tmp_path=str)
def calculate_new_intervals(mt, desired_average_partition_size: int, tmp_path: str):
    """takes a table, keyed by ['locus', ...] and produces a list of intervals suitable
    for repartitioning a combiner matrix table.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        Sparse MT intermediate.
    desired_average_partition_size : :obj:`int`
        Average target number of rows for each partition.
    tmp_path : :obj:`str`
        Temporary path for scan checkpointing.

    Returns
    -------
    (:obj:`List[Interval]`, :obj:`.Type`)
    """
    assert list(mt.row_key) == ['locus']
    assert isinstance(mt.locus.dtype, hl.tlocus)
    reference_genome = mt.locus.dtype.reference_genome
    end = hl.Locus(reference_genome.contigs[-1],
                   reference_genome.lengths[reference_genome.contigs[-1]],
                   reference_genome=reference_genome)

    (n_rows, n_cols) = mt.count()

    if n_rows == 0:
        raise ValueError('empty table!')

    # split by a weight function that takes into account the number of
    # dense entries per row. However, give each row some base weight
    # to prevent densify computations from becoming unbalanced (these
    # scale roughly linearly with N_ROW * N_COL)
    ht = mt.select_rows(weight=hl.agg.count() + (n_cols // 25) + 1).rows().checkpoint(tmp_path)

    total_weight = ht.aggregate(hl.agg.sum(ht.weight))
    partition_weight = int(total_weight / (n_rows / desired_average_partition_size))

    ht = ht.annotate(cumulative_weight=hl.scan.sum(ht.weight),
                     last_weight=hl.scan._prev_nonnull(ht.weight),
                     row_idx=hl.scan.count())

    def partition_bound(x):
        return x - (x % hl.int64(partition_weight))

    at_partition_bound = partition_bound(ht.cumulative_weight) != partition_bound(ht.cumulative_weight - ht.last_weight)

    ht = ht.filter(at_partition_bound | (ht.row_idx == n_rows - 1))
    ht = ht.annotate(start=hl.or_else(
        hl.scan._prev_nonnull(hl.locus_from_global_position(ht.locus.global_position() + 1,
                                                            reference_genome=reference_genome)),
        hl.locus_from_global_position(0, reference_genome=reference_genome)))
    ht = ht.select(
        interval=hl.interval(start=hl.struct(locus=ht.start), end=hl.struct(locus=ht.locus), includes_end=True))

    intervals_dtype = hl.tarray(ht.interval.dtype)
    intervals = ht.aggregate(hl.agg.collect(ht.interval))
    last_st = hl.eval(
        hl.locus_from_global_position(hl.literal(intervals[-1].end.locus).global_position() + 1,
                                      reference_genome=reference_genome))
    interval = hl.Interval(start=hl.Struct(locus=last_st), end=hl.Struct(locus=end), includes_end=True)
    intervals.append(interval)
    return intervals, intervals_dtype


@typecheck(reference_genome=reference_genome_type, interval_size=int)
def calculate_even_genome_partitioning(reference_genome, interval_size) -> List[hl.utils.Interval]:
    """create a list of locus intervals suitable for importing and merging gvcfs.

    Parameters
    ----------
    reference_genome: :class:`str` or :class:`.ReferenceGenome`,
        Reference genome to use. NOTE: only GRCh37 and GRCh38 references
        are supported.
    interval_size: :obj:`int` The ceiling and rough target of interval size.
        Intervals will never be larger than this, but may be smaller.

    Returns
    -------
    :obj:`List[Interval]`
    """

    def calc_parts(contig):
        def locus_interval(start, end):
            return hl.Interval(
                start=hl.Locus(contig=contig, position=start, reference_genome=reference_genome),
                end=hl.Locus(contig=contig, position=end, reference_genome=reference_genome),
                includes_end=True)

        contig_length = reference_genome.lengths[contig]
        n_parts = math.ceil(contig_length / interval_size)
        real_size = math.ceil(contig_length / n_parts)
        n = 1
        intervals = []
        while n < contig_length:
            start = n
            end = min(n + real_size, contig_length)
            intervals.append(locus_interval(start, end))
            n = end + 1

        return intervals

    if reference_genome.name == 'GRCh37':
        contigs = [f'{i}' for i in range(1, 23)] + ['X', 'Y', 'MT']
    elif reference_genome.name == 'GRCh38':
        contigs = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
    else:
        raise ValueError(
            f"Unsupported reference genome '{reference_genome.name}', "
            "only 'GRCh37' and 'GRCh38' are supported")

    intervals = []
    for ctg in contigs:
        intervals.extend(calc_parts(ctg))
    return intervals
