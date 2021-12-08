from typing import Sequence

import hail as hl
from hail import ir
from hail.expr import ArrayExpression, expr_any, expr_array, expr_interval, expr_locus
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_first_key_field_locus
from hail.table import Table
from hail.typecheck import sequenceof, typecheck, nullable, oneof
from hail.utils.java import Env, info
from hail.utils.misc import divide_null, new_temp_file
from hail.vds.variant_dataset import VariantDataset


def write_variant_datasets(vdss, paths, *,
                           overwrite=False, stage_locally=False,
                           codec_spec=None):
    """Write many `vdses` to their corresponding path in `paths`."""
    ref_writer = ir.MatrixNativeMultiWriter([f"{p}/reference_data" for p in paths], overwrite, stage_locally,
                                            codec_spec)
    var_writer = ir.MatrixNativeMultiWriter([f"{p}/variant_data" for p in paths], overwrite, stage_locally, codec_spec)
    Env.backend().execute(ir.MatrixMultiWrite([vds.reference_data._mir for vds in vdss], ref_writer))
    Env.backend().execute(ir.MatrixMultiWrite([vds.variant_data._mir for vds in vdss], var_writer))


@typecheck(vds=VariantDataset)
def to_dense_mt(vds: 'VariantDataset') -> 'MatrixTable':
    """Creates a single, dense :class:`.MatrixTable` from the split
    :class:`.VariantDataset` representation.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset in dense MatrixTable representation.
    """
    ref = vds.reference_data
    ref = ref.drop(*(x for x in ('alleles', 'rsid') if x in ref.row))
    var = vds.variant_data
    refl = ref.localize_entries('_ref_entries')
    varl = var.localize_entries('_var_entries', '_var_cols')
    varl = varl.annotate(_variant_defined=True)
    joined = refl.join(varl.key_by('locus'), how='outer')
    dr = joined.annotate(
        dense_ref=hl.or_missing(
            joined._variant_defined,
            hl.scan._densify(hl.len(joined._var_cols), joined._ref_entries)
        )
    )
    dr = dr.filter(dr._variant_defined)

    def coalesce_join(ref, var):

        call_field = 'GT' if 'GT' in var else 'LGT'
        assert call_field in var, var.dtype

        merged_fields = {}
        merged_fields[call_field] = hl.coalesce(var[call_field], hl.call(0, 0))
        for field in ref.dtype:
            if field in var:
                merged_fields[field] = hl.coalesce(var[field], ref[field])

        return hl.struct(**merged_fields).annotate(**{f: var[f] for f in var if f not in merged_fields})

    dr = dr.annotate(
        _dense=hl.zip(dr._var_entries, dr.dense_ref).map(
            lambda tuple: coalesce_join(hl.or_missing(tuple[1].END <= dr.locus.position, tuple[1]), tuple[0])
        )
    )
    dr = dr._key_by_assert_sorted('locus', 'alleles')
    dr = dr.drop('_var_entries', '_ref_entries', 'dense_ref', '_variant_defined')
    return dr._unlocalize_entries('_dense', '_var_cols', list(var.col_key))


@typecheck(vds=VariantDataset)
def to_merged_sparse_mt(vds: 'VariantDataset') -> 'MatrixTable':
    """Creates a single, merged sparse :class:'.MatrixTable' from the split
    :class:`.VariantDataset` representation.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset in the merged sparse MatrixTable representation.
    """
    rht = vds.reference_data.localize_entries('_ref_entries', '_ref_cols')
    vht = vds.variant_data.localize_entries('_var_entries', '_var_cols')

    # drop 'alleles' key for join
    vht = vht.key_by('locus')

    merged_schema = {}
    for e in vds.variant_data.entry:
        merged_schema[e] = vds.variant_data[e].dtype

    for e in vds.reference_data.entry:
        if e in merged_schema:
            if not merged_schema[e] == vds.reference_data[e].dtype:
                raise TypeError(f"cannot unify field {e!r}: {merged_schema[e]}, {vds.reference_data[e].dtype}")
        else:
            merged_schema[e] = vds.reference_data[e].dtype

    ht = rht.join(vht, how='outer').drop('_ref_cols')

    def merge_arrays(r_array, v_array):

        def rewrite_ref(r):
            ref_block_selector = {}
            for k, t in merged_schema.items():
                if k == 'LA':
                    ref_block_selector[k] = hl.literal([0])
                elif k in ('LGT', 'GT'):
                    ref_block_selector[k] = hl.call(0, 0)
                else:
                    ref_block_selector[k] = r[k] if k in r else hl.missing(t)
            return r.select(**ref_block_selector)

        def rewrite_var(v):
            return v.select(**{
                k: v[k] if k in v else hl.missing(t)
                for k, t in merged_schema.items()
            })

        return hl.case() \
            .when(hl.is_missing(r_array), v_array.map(rewrite_var)) \
            .when(hl.is_missing(v_array), r_array.map(rewrite_ref)) \
            .default(hl.zip(r_array, v_array).map(lambda t: hl.coalesce(rewrite_var(t[1]), rewrite_ref(t[0]))))

    ht = ht.select(
        alleles=hl.coalesce(ht['alleles'], hl.array([ht['ref_allele']])),
        # handle cases where vmt is not keyed by alleles
        **{k: ht[k] for k in vds.variant_data.row_value if k != 'alleles'},
        _entries=merge_arrays(ht['_ref_entries'], ht['_var_entries'])
    )
    ht = ht._key_by_assert_sorted('locus', 'alleles')
    return ht._unlocalize_entries('_entries', '_var_cols', list(vds.variant_data.col_key))


@typecheck(vds=VariantDataset, name=str, gq_bins=sequenceof(int))
def sample_qc(vds: 'VariantDataset', *, name='sample_qc', gq_bins: 'Sequence[int]' = (0, 20, 60)) -> 'Table':
    """Run sample_qc on dataset in the split :class:`.VariantDataset` representation.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    name : :obj:`str`
        Name for resulting field.
    gq_bins : :class:`tuple` of :obj:`int`
        Tuple containing cutoffs for genotype quality (GQ) scores.

    Returns
    -------
    :class:`.Table`
        Hail Table of results, keyed by sample.
    """

    require_first_key_field_locus(vds.reference_data, 'sample_qc')
    require_first_key_field_locus(vds.variant_data, 'sample_qc')

    from hail.expr.functions import _num_allele_type, _allele_types

    allele_types = _allele_types[:]
    allele_types.extend(['Transition', 'Transversion'])
    allele_enum = {i: v for i, v in enumerate(allele_types)}
    allele_ints = {v: k for k, v in allele_enum.items()}

    def allele_type(ref, alt):
        return hl.bind(
            lambda at: hl.if_else(at == allele_ints['SNP'],
                                  hl.if_else(hl.is_transition(ref, alt),
                                             allele_ints['Transition'],
                                             allele_ints['Transversion']),
                                  at),
            _num_allele_type(ref, alt)
        )

    variant_ac = Env.get_uid()
    variant_atypes = Env.get_uid()

    vmt = vds.variant_data
    if 'GT' not in vmt.entry:
        vmt = vmt.annotate_entries(GT=hl.experimental.lgt_to_gt(vmt.LGT, vmt.LA))

    vmt = vmt.annotate_rows(**{
        variant_ac: hl.agg.call_stats(vmt.GT, vmt.alleles).AC,
        variant_atypes: vmt.alleles[1:].map(lambda alt: allele_type(vmt.alleles[0], alt))
    })

    bound_exprs = {}

    bound_exprs['n_het'] = hl.agg.count_where(vmt['GT'].is_het())
    bound_exprs['n_hom_var'] = hl.agg.count_where(vmt['GT'].is_hom_var())
    bound_exprs['n_singleton'] = hl.agg.sum(
        hl.sum(hl.range(0, vmt['GT'].ploidy).map(lambda i: vmt[variant_ac][vmt['GT'][i]] == 1))
    )

    bound_exprs['allele_type_counts'] = hl.agg.explode(
        lambda allele_type: hl.tuple(
            hl.agg.count_where(allele_type == i) for i in range(len(allele_ints))
        ),
        (hl.range(0, vmt['GT'].ploidy)
         .map(lambda i: vmt['GT'][i])
         .filter(lambda allele_idx: allele_idx > 0)
         .map(lambda allele_idx: vmt[variant_atypes][allele_idx - 1]))
    )

    gq_exprs = hl.agg.filter(
        hl.is_defined(vmt.GT),
        hl.struct(**{f'gq_over_{x}': hl.agg.count_where(vmt.GQ > x) for x in gq_bins})
    )

    result_struct = hl.rbind(
        hl.struct(**bound_exprs),
        lambda x: hl.rbind(
            hl.struct(**{
                'gq_exprs': gq_exprs,
                'n_het': x.n_het,
                'n_hom_var': x.n_hom_var,
                'n_non_ref': x.n_het + x.n_hom_var,
                'n_singleton': x.n_singleton,
                'n_snp': (x.allele_type_counts[allele_ints['Transition']]
                          + x.allele_type_counts[allele_ints['Transversion']]),
                'n_insertion': x.allele_type_counts[allele_ints['Insertion']],
                'n_deletion': x.allele_type_counts[allele_ints['Deletion']],
                'n_transition': x.allele_type_counts[allele_ints['Transition']],
                'n_transversion': x.allele_type_counts[allele_ints['Transversion']],
                'n_star': x.allele_type_counts[allele_ints['Star']]
            }),
            lambda s: s.annotate(
                r_ti_tv=divide_null(hl.float64(s.n_transition), s.n_transversion),
                r_het_hom_var=divide_null(hl.float64(s.n_het), s.n_hom_var),
                r_insertion_deletion=divide_null(hl.float64(s.n_insertion), s.n_deletion)
            )
        )
    )
    variant_results = vmt.select_cols(**result_struct).cols()

    rmt = vds.reference_data
    ref_results = rmt.select_cols(
        gq_exprs=hl.struct(**{
            f'gq_over_{x}': hl.agg.filter(rmt.GQ > x, hl.agg.sum(1 + rmt.END - rmt.locus.position)) for x in gq_bins
        })
    ).cols()

    joined = ref_results[variant_results.key].gq_exprs
    joined_results = variant_results.transmute(**{
        f'gq_over_{x}': variant_results.gq_exprs[f'gq_over_{x}'] + joined[f'gq_over_{x}'] for x in gq_bins
    })
    return joined_results


@typecheck(vds=VariantDataset, samples_table=Table, keep=bool, remove_dead_alleles=bool)
def filter_samples(vds: 'VariantDataset', samples_table: 'Table', *,
                   keep: bool = True,
                   remove_dead_alleles: bool = False) -> 'VariantDataset':
    """Filter samples in a :class:`.VariantDataset`.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    samples_table : :class:`.Table`
        Samples to filter on.
    keep : :obj:`bool`
        Whether to keep (default), or filter out the samples from `samples_table`.
    remove_dead_alleles : :obj:`bool`
        If true, remove alleles observed in no samples. Alleles with AC == 0 will be
        removed, and LA values recalculated.

    Returns
    -------
    :class:`.VariantDataset`
    """
    if not list(samples_table[x].dtype for x in samples_table.key) == [hl.tstr]:
        raise TypeError(f'invalid key: {samples_table.key.dtype}')
    samples_to_keep = samples_table.aggregate(hl.agg.collect_as_set(samples_table.key[0]), _localize=False)._persist()
    reference_data = vds.reference_data.filter_cols(samples_to_keep.contains(vds.reference_data.col_key[0]), keep=keep)
    reference_data = reference_data.filter_rows(hl.agg.count() > 0)
    variant_data = vds.variant_data.filter_cols(samples_to_keep.contains(vds.variant_data.col_key[0]), keep=keep)

    if remove_dead_alleles:
        vd = variant_data
        vd = vd.annotate_rows(__allele_counts=hl.agg.explode(lambda x: hl.agg.counter(x), vd.LA), __n=hl.agg.count())
        vd = vd.filter_rows(vd.__n > 0)
        vd = vd.drop('__n')

        vd = vd.annotate_rows(__kept_indices=hl.dict(
            hl.enumerate(
                hl.range(hl.len(vd.alleles)).filter(lambda idx: (idx == 0) | (vd.__allele_counts.get(idx, 0) > 0)),
                index_first=False)))

        vd = vd.annotate_rows(
            __old_to_new_LA=hl.range(hl.len(vd.alleles)).map(lambda idx: vd.__kept_indices.get(idx, -1)))

        def new_la_index(old_idx):
            raw_idx = vd.__old_to_new_LA[old_idx]
            return hl.case().when(raw_idx >= 0, raw_idx) \
                .or_error("'filter_samples': unexpected local allele: old index=" + hl.str(old_idx))

        vd = vd.annotate_entries(LA=vd.LA.map(lambda la: new_la_index(la)))
        vd = vd.key_rows_by('locus')
        vd = vd.annotate_rows(alleles=vd.__kept_indices.keys().map(lambda i: vd.alleles[i]))
        vd = vd._key_rows_by_assert_sorted('locus', 'alleles')
        vd = vd.drop('__allele_counts', '__kept_indices', '__old_to_new_LA')
        return VariantDataset(reference_data, vd)

    variant_data = variant_data.filter_rows(hl.agg.count() > 0)
    return VariantDataset(reference_data, variant_data)


@typecheck(vds=VariantDataset,
           calling_intervals=oneof(Table, expr_array(expr_interval(expr_locus()))),
           normalization_contig=str
           )
def impute_sex_chromosome_ploidy(
        vds: VariantDataset,
        calling_intervals,
        normalization_contig: str
) -> hl.Table:
    """Impute sex chromosome ploidy from depth of reference data within calling intervals.

    Returns a :class:`.Table` with sample ID keys, with the following fields:

     -  ``autosomal_mean_dp`` (*float64*): Mean depth on calling intervals on normalization contig.
     -  ``x_mean_dp`` (*float64*): Mean depth on calling intervals on X chromosome.
     -  ``x_ploidy`` (*float64*): Estimated ploidy on X chromosome. Equal to ``2 * autosomal_mean_dp / x_mean_dp``.
     -  ``y_mean_dp`` (*float64*): Mean depth on calling intervals on  chromosome.
     -  ``y_ploidy`` (*float64*): Estimated ploidy on Y chromosome. Equal to ``2 * autosomal_mean_dp / y_mean_dp``.

    Parameters
    ----------
    vds : vds: :class:`.VariantDataset`
        Dataset.
    calling_intervals : :class:`.Table` or :class:`.ArrayExpression`
        Calling intervals with consistent read coverage (for exomes, trim the capture intervals).
    normalization_contig : str
        Autosomal contig for depth comparison.

    Returns
    -------
    :class:`.Table`
    """

    if not isinstance(calling_intervals, Table):
        calling_intervals = hl.Table.parallelize(hl.map(lambda i: hl.struct(interval=i), calling_intervals),
                                                 schema=hl.tstruct(interval=calling_intervals.dtype.element_type),
                                                 key='interval')
    else:
        key_dtype = calling_intervals.key.dtype
        if len(key_dtype) != 1 or not isinstance(calling_intervals.key[0].dtype, hl.tinterval) or calling_intervals.key[0].dtype.point_type != vds.reference_data.locus.dtype:
            raise ValueError(f"'impute_sex_chromosome_ploidy': expect calling_intervals to be list of intervals or"
                             f" table with single key of type interval<locus>, found table with key: {key_dtype}")

    rg = vds.reference_data.locus.dtype.reference_genome

    par_boundaries = []
    for par_interval in rg.par:
        par_boundaries.append(par_interval.start)
        par_boundaries.append(par_interval.end)

    # segment on PAR interval boundaries
    calling_intervals = hl.segment_intervals(calling_intervals, par_boundaries)

    # remove intervals overlapping PAR
    calling_intervals = calling_intervals.filter(hl.all(lambda x: ~x.overlaps(calling_intervals.interval), hl.literal(rg.par)))

    # checkpoint for efficient multiple downstream usages
    info("'impute_sex_chromosome_ploidy': checkpointing calling intervals")
    calling_intervals = calling_intervals.checkpoint(new_temp_file(extension='ht'))

    interval = calling_intervals.key[0]
    (any_bad_intervals, chrs_represented) = calling_intervals.aggregate(
        (hl.agg.any(interval.start.contig != interval.end.contig), hl.agg.collect_as_set(interval.start.contig)))
    if any_bad_intervals:
        raise ValueError("'impute_sex_chromosome_ploidy' does not support calling intervals that span chromosome boundaries")

    if len(rg.x_contigs) != 1:
        raise NotImplementedError(
            f"reference genome {rg.name!r} has multiple X contigs, this is not supported in 'impute_sex_chromosome_ploidy'"
        )
    chr_x = rg.x_contigs[0]
    if len(rg.y_contigs) != 1:
        raise NotImplementedError(
            f"reference genome {rg.name!r} has multiple Y contigs, this is not supported in 'impute_sex_chromosome_ploidy'"
        )
    chr_y = rg.y_contigs[0]

    kept_contig_filter = hl.array(chrs_represented).map(lambda x: hl.parse_locus_interval(x, reference_genome=rg))
    vds = VariantDataset(hl.filter_intervals(vds.reference_data, kept_contig_filter),
                         hl.filter_intervals(vds.variant_data, kept_contig_filter))

    coverage = interval_coverage(vds, calling_intervals, gq_thresholds=()).drop('gq_thresholds')

    coverage = coverage.annotate_rows(contig=coverage.interval.start.contig)
    coverage = coverage.annotate_cols(
        __mean_dp=hl.agg.group_by(coverage.contig, hl.agg.sum(coverage.sum_dp) / hl.agg.sum(coverage.interval_size)))

    mean_dp_dict = coverage.__mean_dp
    auto_dp = mean_dp_dict.get(normalization_contig)
    x_dp = mean_dp_dict.get(chr_x)
    y_dp = mean_dp_dict.get(chr_y)
    per_sample = coverage.transmute_cols(autosomal_mean_dp=auto_dp,
                                         x_mean_dp=x_dp,
                                         x_ploidy=x_dp / auto_dp * 2,
                                         y_mean_dp=y_dp,
                                         y_ploidy=y_dp / auto_dp * 2)
    info("'impute_sex_chromosome_ploidy': computing and checkpointing coverage and karyotype metrics")
    return per_sample.cols().checkpoint(new_temp_file('impute_sex_karyotype', extension='ht'))


@typecheck(vds=VariantDataset, variants_table=Table, keep=bool)
def filter_variants(vds: 'VariantDataset', variants_table: 'Table', *, keep: bool = True) -> 'VariantDataset':
    """Filter variants in a :class:`.VariantDataset`, without removing reference
    data.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    variants_table : :class:`.Table`
        Variants to filter on.
    keep: :obj:`bool`
        Whether to keep (default), or filter out the variants from `variants_table`.

    Returns
    -------
    :class:`.VariantDataset`.
    """
    if keep:
        variant_data = vds.variant_data.semi_join_rows(variants_table)
    else:
        variant_data = vds.variant_data.anti_join_rows(variants_table)
    return VariantDataset(vds.reference_data, variant_data)


@typecheck(vds=VariantDataset, intervals=expr_array(expr_interval(expr_any)), keep=bool)
def filter_intervals(vds: 'VariantDataset', intervals: 'ArrayExpression', *, keep: bool = False) -> 'VariantDataset':
    """Filter intervals in a :class:`.VariantDataset` (only on variant data for
    now).

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    intervals : :class:`.ArrayExpression` of type :class:`.tinterval`
        Intervals to filter on.
    keep : :obj:`bool`
        Whether to keep, or filter out (default) rows that fall within any
        interval in `intervals`.

    Returns
    -------
    :class:`.VariantDataset`
    """
    # for now, don't touch reference data.
    # should remove large regions and scan forward ref blocks to the start of the next kept region
    variant_data = hl.filter_intervals(vds.variant_data, intervals, keep)
    return VariantDataset(vds.reference_data, variant_data)


@typecheck(vds=VariantDataset, filter_changed_loci=bool)
def split_multi(vds: 'VariantDataset', *, filter_changed_loci: bool = False) -> 'VariantDataset':
    """Split the multiallelic variants in a :class:`.VariantDataset`.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    filter_changed_loci : :obj:`bool`
        If any REF/ALT pair changes locus under :func:`.min_rep`, filter that
        variant instead of throwing an error.

    Returns
    -------
    :class:`.VariantDataset`
    """
    variant_data = hl.experimental.sparse_split_multi(vds.variant_data, filter_changed_loci=filter_changed_loci)
    return VariantDataset(vds.reference_data, variant_data)


@typecheck(ref=MatrixTable, intervals=Table)
def segment_reference_blocks(ref: 'MatrixTable', intervals: 'Table') -> 'MatrixTable':
    """Returns a matrix table of reference blocks segmented according to intervals.

    Loci outside the given intervals are discarded. Reference blocks that start before
    but span an interval will appear at the interval start locus.

    Note
    ----
        Assumes disjoint intervals which do not span contigs.

        Requires start-inclusive intervals.

    Parameters
    ----------
    ref : :class:`.MatrixTable`
        MatrixTable of reference blocks.
    intervals : :class:`.Table`
        Table of intervals at which to segment reference blocks.

    Returns
    -------
    :class:`.MatrixTable`
    """
    interval_field = list(intervals.key)[0]
    if not intervals[interval_field].dtype == hl.tinterval(ref.locus.dtype):
        raise ValueError(f"expect intervals to be keyed by intervals of loci matching the VariantDataset:"
                         f" found {intervals[interval_field].dtype} / {ref.locus.dtype}")
    intervals = intervals.select(_interval_dup=intervals[interval_field])

    if not intervals.aggregate(
            hl.agg.all(intervals[interval_field].includes_start & (
                intervals[interval_field].start.contig == intervals[interval_field].end.contig))):
        raise ValueError("expect intervals to be start-inclusive")

    starts = intervals.key_by(_start_locus=intervals[interval_field].start)
    starts = starts.annotate(_include_locus=True)
    refl = ref.localize_entries('_ref_entries', '_ref_cols')
    joined = refl.join(starts, how='outer')
    rg = ref.locus.dtype.reference_genome
    contigs = rg.contigs
    contig_idx_map = hl.literal({contigs[i]: i for i in range(len(contigs))}, 'dict<str, int32>')
    joined = joined.annotate(__contig_idx=contig_idx_map[joined.locus.contig])
    joined = joined.annotate(
        _ref_entries=joined._ref_entries.map(lambda e: e.annotate(__contig_idx=joined.__contig_idx)))
    dense = joined.annotate(
        dense_ref=hl.or_missing(
            joined._include_locus,
            hl.rbind(joined.locus.position,
                     lambda pos: hl.enumerate(hl.scan._densify(hl.len(joined._ref_cols), joined._ref_entries))
                     .map(lambda idx_and_e: hl.rbind(idx_and_e[0], idx_and_e[1],
                                                     lambda idx, e: hl.coalesce(joined._ref_entries[idx], hl.or_missing(
                                                         (e.__contig_idx == joined.__contig_idx) & (e.END >= pos),
                                                         e))).drop('__contig_idx'))
                     ))
    )
    dense = dense.filter(dense._include_locus).drop('_interval_dup', '_include_locus', '__contig_idx')

    # at this point, 'dense' is a table with dense rows of reference blocks, keyed by locus

    refl_filtered = refl.annotate(**{interval_field: intervals[refl.locus]._interval_dup})

    # remove rows that are not contained in an interval, and rows that are the start of an
    # interval (interval starts come from the 'dense' table)
    refl_filtered = refl_filtered.filter(
        hl.is_defined(refl_filtered[interval_field]) & (refl_filtered.locus != refl_filtered[interval_field].start))

    # union dense interval starts with filtered table
    refl_filtered = refl_filtered.union(dense.transmute(_ref_entries=dense.dense_ref))

    # rewrite reference blocks to end at the first of (interval end, reference block end)
    refl_filtered = refl_filtered.annotate(
        interval_end=refl_filtered[interval_field].end.position - ~refl_filtered[interval_field].includes_end)
    refl_filtered = refl_filtered.annotate(
        _ref_entries=refl_filtered._ref_entries.map(
            lambda entry: entry.annotate(END=hl.min(entry.END, refl_filtered.interval_end))))

    return refl_filtered._unlocalize_entries('_ref_entries', '_ref_cols', list(ref.col_key))


@typecheck(vds=VariantDataset, intervals=Table, gq_thresholds=sequenceof(int), dp_field=nullable(str))
def interval_coverage(vds: VariantDataset, intervals: hl.Table, gq_thresholds=(0, 20,), dp_field=None) -> 'MatrixTable':
    """Compute statistics about base coverage by interval.

    Returns a :class:`.MatrixTable` with interval row keys and sample column keys.

    Contains the following row fields:
     - ``interval`` (*interval*): Genomic interval of interest.
     - ``interval_size`` (*int32*): Size of interval, in bases.


    Computes the following entry fields:

     -  ``bases_over_gq_threshold`` (*tuple of int64*): Number of bases in the interval
        over each GQ threshold.
     -  ``fraction_over_gq_threshold`` (*tuple of float64*): Fraction of interval (in bases)
        above each GQ threshold. Computed by dividing each member of *bases_over_gq_threshold*
        by *interval_size*.
     -  ``sum_dp`` (*int64*): Sum of depth values by base across the interval.
     -  ``mean_dp`` (*float64*): Mean depth of bases across the interval. Computed by dividing
        *mean_dp* by *interval_size*.

    Note
    ----
    The metrics computed by this method are computed **only from reference blocks**. Most
    variant callers produce data where non-reference calls interrupt reference blocks, and
    so the metrics computed here are slight underestimates of the true values (which would
    include the quality/depth of non-reference calls). This is likely a negligible difference,
    but is something to be aware of, especially as it interacts with samples of
    ancestral backgrounds with more or fewer non-reference calls.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
    intervals : :class:`.Table`
        Table of intervals. Must be start-inclusive, and cannot span contigs.
    gq_thresholds : tuple of int
        GQ thresholds.
    dp_field : str, optional
        Field for depth calculation. Uses DP or MIN_DP by default (with priority for DP if present).

    Returns
    -------
    :class:`.MatrixTable`
        Interval-by-sample matrix
    """
    ref = vds.reference_data
    split = segment_reference_blocks(ref, intervals)
    intervals = intervals.annotate(interval_dup=intervals.key[0])

    if 'DP' in ref.entry:
        dp_field_to_use = 'DP'
    elif 'MIN_DP' in ref.entry:
        dp_field_to_use = 'MIN_DP'
    else:
        dp_field_to_use = dp_field

    if dp_field_to_use is not None:
        dp_field_dict = {'sum_dp': hl.agg.sum((split.END - split.locus.position + 1) * split[dp_field_to_use])}
    else:
        dp_field_dict = dict()

    per_interval = split.group_rows_by(interval=intervals[split.row_key[0]].interval_dup) \
        .aggregate(
        bases_over_gq_threshold=tuple(
            hl.agg.filter(split.GQ > gq_threshold, hl.agg.sum(split.END - split.locus.position + 1)) for gq_threshold in
            gq_thresholds),
        **dp_field_dict
    )

    interval = per_interval.interval
    interval_size = interval.end.position + interval.includes_end - interval.start.position - 1 + interval.includes_start
    per_interval = per_interval.annotate_rows(interval_size=interval_size)
    per_interval = per_interval.annotate_entries(
        fraction_over_gq_threshold=tuple(
            hl.float(x) / per_interval.interval_size for x in per_interval.bases_over_gq_threshold))

    if dp_field_to_use is not None:
        per_interval = per_interval.annotate_entries(mean_dp=per_interval.sum_dp / per_interval.interval_size)

    per_interval = per_interval.annotate_globals(gq_thresholds=hl.tuple(gq_thresholds))

    return per_interval
