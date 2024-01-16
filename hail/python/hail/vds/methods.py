from typing import Sequence

import hail as hl
from hail import ir
from hail.expr import expr_any, expr_array, expr_bool, expr_interval, expr_locus, expr_str
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_first_key_field_locus
from hail.table import Table
from hail.typecheck import dictof, enumeration, func_spec, nullable, oneof, sequenceof, typecheck
from hail.utils.java import Env, info, warning
from hail.utils.misc import divide_null, new_temp_file, wrap_to_list
from hail.vds.variant_dataset import VariantDataset


def write_variant_datasets(vdss, paths, *, overwrite=False, stage_locally=False, codec_spec=None):
    """Write many `vdses` to their corresponding path in `paths`."""
    ref_writer = ir.MatrixNativeMultiWriter(
        [f"{p}/reference_data" for p in paths], overwrite, stage_locally, codec_spec
    )
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
    # FIXME(chrisvittal) consider changing END semantics on VDS to make this better
    # see https://github.com/hail-is/hail/issues/13183 for why this is here and more discussion
    # we assume that END <= contig.length
    ref = ref.annotate_rows(_locus_global_pos=ref.locus.global_position(), _locus_pos=ref.locus.position)
    ref = ref.transmute_entries(_END_GLOBAL=ref._locus_global_pos + (ref.END - ref._locus_pos))

    to_drop = 'alleles', 'rsid', 'ref_allele', '_locus_global_pos', '_locus_pos'
    ref = ref.drop(*(x for x in to_drop if x in ref.row))
    var = vds.variant_data
    refl = ref.localize_entries('_ref_entries')
    varl = var.localize_entries('_var_entries', '_var_cols')
    varl = varl.annotate(_variant_defined=True)
    joined = varl.key_by('locus').join(refl, how='outer')
    dr = joined.annotate(
        dense_ref=hl.or_missing(
            joined._variant_defined, hl.scan._densify(hl.len(joined._var_cols), joined._ref_entries)
        )
    )
    dr = dr.filter(dr._variant_defined)

    def coalesce_join(ref, var):
        call_field = 'GT' if 'GT' in var else 'LGT'
        assert call_field in var, var.dtype

        shared_fields = [call_field, *list(f for f in ref.dtype if f in var.dtype)]
        shared_field_set = set(shared_fields)
        var_fields = [f for f in var.dtype if f not in shared_field_set]

        return hl.if_else(
            hl.is_defined(var),
            var.select(*shared_fields, *var_fields),
            ref.annotate(**{call_field: hl.call(0, 0)}).select(
                *shared_fields, **{f: hl.missing(var[f].dtype) for f in var_fields}
            ),
        )

    dr = dr.annotate(
        _dense=hl.rbind(
            dr._ref_entries,
            lambda refs_at_this_row: hl.enumerate(hl.zip(dr._var_entries, dr.dense_ref)).map(
                lambda tup: coalesce_join(
                    hl.coalesce(
                        refs_at_this_row[tup[0]],
                        hl.or_missing(tup[1][1]._END_GLOBAL >= dr.locus.global_position(), tup[1][1]),
                    ),
                    tup[1][0],
                )
            ),
        ),
    )

    dr = dr._key_by_assert_sorted('locus', 'alleles')
    fields_to_drop = ['_var_entries', '_ref_entries', 'dense_ref', '_variant_defined']

    if hl.vds.VariantDataset.ref_block_max_length_field in dr.globals:
        fields_to_drop.append(hl.vds.VariantDataset.ref_block_max_length_field)

    if 'ref_allele' in dr.row:
        fields_to_drop.append('ref_allele')
    dr = dr.drop(*fields_to_drop)
    return dr._unlocalize_entries('_dense', '_var_cols', list(var.col_key))


@typecheck(vds=VariantDataset, ref_allele_function=nullable(func_spec(1, expr_str)))
def to_merged_sparse_mt(vds: 'VariantDataset', *, ref_allele_function=None) -> 'MatrixTable':
    """Creates a single, merged sparse :class:`.MatrixTable` from the split
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
    for e in vds.reference_data.entry:
        merged_schema[e] = vds.reference_data[e].dtype

    for e in vds.variant_data.entry:
        if e in merged_schema:
            if not merged_schema[e] == vds.variant_data[e].dtype:
                raise TypeError(f"cannot unify field {e!r}: {merged_schema[e]}, {vds.variant_data[e].dtype}")
        else:
            merged_schema[e] = vds.variant_data[e].dtype

    ht = vht.join(rht, how='outer').drop('_ref_cols')

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
            return v.select(**{k: v[k] if k in v else hl.missing(t) for k, t in merged_schema.items()})

        return (
            hl.case()
            .when(hl.is_missing(r_array), v_array.map(rewrite_var))
            .when(hl.is_missing(v_array), r_array.map(rewrite_ref))
            .default(hl.zip(r_array, v_array).map(lambda t: hl.coalesce(rewrite_var(t[1]), rewrite_ref(t[0]))))
        )

    if ref_allele_function is None:
        rg = ht.locus.dtype.reference_genome
        if 'ref_allele' in ht.row:

            def ref_allele_function(ht):
                return ht.ref_allele

        elif rg.has_sequence():

            def ref_allele_function(ht):
                return ht.locus.sequence_context()

            info("to_merged_sparse_mt: using locus sequence context to fill in reference alleles at monomorphic loci.")
        else:
            raise ValueError(
                "to_merged_sparse_mt: in order to construct a ref allele for reference-only sites, "
                "either pass a function to fill in reference alleles (e.g. ref_allele_function=lambda locus: hl.missing('str'))"
                " or add a sequence file with 'hl.get_reference(RG_NAME).add_sequence(FASTA_PATH)'."
            )
    ht = ht.select(
        alleles=hl.coalesce(ht['alleles'], hl.array([ref_allele_function(ht)])),
        # handle cases where vmt is not keyed by alleles
        **{k: ht[k] for k in vds.variant_data.row_value if k != 'alleles'},
        _entries=merge_arrays(ht['_ref_entries'], ht['_var_entries']),
    )
    ht = ht._key_by_assert_sorted('locus', 'alleles')
    return ht._unlocalize_entries('_entries', '_var_cols', list(vds.variant_data.col_key))


@typecheck(vds=VariantDataset, gq_bins=sequenceof(int), dp_bins=sequenceof(int), dp_field=nullable(str))
def sample_qc(
    vds: 'VariantDataset',
    *,
    gq_bins: 'Sequence[int]' = (0, 20, 60),
    dp_bins: 'Sequence[int]' = (0, 1, 10, 20, 30),
    dp_field=None,
) -> 'Table':
    """Compute sample quality metrics about a :class:`.VariantDataset`.

    If the `dp_field` parameter is not specified, the ``DP`` is used for depth
    if present. If no ``DP`` field is present, the ``MIN_DP`` field is used. If no ``DP``
    or ``MIN_DP`` field is present, no depth statistics will be calculated.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    name : :obj:`str`
        Name for resulting field.
    gq_bins : :class:`tuple` of :obj:`int`
        Tuple containing cutoffs for genotype quality (GQ) scores.
    dp_bins : :class:`tuple` of :obj:`int`
        Tuple containing cutoffs for depth (DP) scores.
    dp_field : :obj:`str`
        Name of depth field. If not supplied, DP or MIN_DP will be used, in that order.

    Returns
    -------
    :class:`.Table`
        Hail Table of results, keyed by sample.
    """

    require_first_key_field_locus(vds.reference_data, 'sample_qc')
    require_first_key_field_locus(vds.variant_data, 'sample_qc')

    ref = vds.reference_data

    if 'DP' in ref.entry:
        ref_dp_field_to_use = 'DP'
    elif 'MIN_DP' in ref.entry:
        ref_dp_field_to_use = 'MIN_DP'
    else:
        ref_dp_field_to_use = dp_field

    from hail.expr.functions import _allele_types, _num_allele_type

    allele_types = _allele_types[:]
    allele_types.extend(['Transition', 'Transversion'])
    allele_enum = {i: v for i, v in enumerate(allele_types)}
    allele_ints = {v: k for k, v in allele_enum.items()}

    def allele_type(ref, alt):
        return hl.bind(
            lambda at: hl.if_else(
                at == allele_ints['SNP'],
                hl.if_else(hl.is_transition(ref, alt), allele_ints['Transition'], allele_ints['Transversion']),
                at,
            ),
            _num_allele_type(ref, alt),
        )

    variant_ac = Env.get_uid()
    variant_atypes = Env.get_uid()

    vmt = vds.variant_data
    if 'GT' not in vmt.entry:
        vmt = vmt.annotate_entries(GT=hl.vds.lgt_to_gt(vmt.LGT, vmt.LA))

    vmt = vmt.annotate_rows(**{
        variant_ac: hl.agg.call_stats(vmt.GT, vmt.alleles).AC,
        variant_atypes: vmt.alleles[1:].map(lambda alt: allele_type(vmt.alleles[0], alt)),
    })

    bound_exprs = {}

    bound_exprs['n_het'] = hl.agg.count_where(vmt['GT'].is_het())
    bound_exprs['n_hom_var'] = hl.agg.count_where(vmt['GT'].is_hom_var())
    bound_exprs['n_singleton'] = hl.agg.sum(
        hl.rbind(
            vmt['GT'],
            lambda gt: hl.sum(
                hl.range(0, gt.ploidy).map(
                    lambda i: hl.rbind(gt[i], lambda gti: (gti != 0) & (vmt[variant_ac][gti] == 1))
                )
            ),
        )
    )
    bound_exprs['n_singleton_ti'] = hl.agg.sum(
        hl.rbind(
            vmt['GT'],
            lambda gt: hl.sum(
                hl.range(0, gt.ploidy).map(
                    lambda i: hl.rbind(
                        gt[i],
                        lambda gti: (gti != 0)
                        & (vmt[variant_ac][gti] == 1)
                        & (vmt[variant_atypes][gti - 1] == allele_ints['Transition']),
                    )
                )
            ),
        )
    )
    bound_exprs['n_singleton_tv'] = hl.agg.sum(
        hl.rbind(
            vmt['GT'],
            lambda gt: hl.sum(
                hl.range(0, gt.ploidy).map(
                    lambda i: hl.rbind(
                        gt[i],
                        lambda gti: (gti != 0)
                        & (vmt[variant_ac][gti] == 1)
                        & (vmt[variant_atypes][gti - 1] == allele_ints['Transversion']),
                    )
                )
            ),
        )
    )

    bound_exprs['allele_type_counts'] = hl.agg.explode(
        lambda allele_type: hl.tuple(hl.agg.count_where(allele_type == i) for i in range(len(allele_ints))),
        (
            hl.range(0, vmt['GT'].ploidy)
            .map(lambda i: vmt['GT'][i])
            .filter(lambda allele_idx: allele_idx > 0)
            .map(lambda allele_idx: vmt[variant_atypes][allele_idx - 1])
        ),
    )

    dp_exprs = {}
    if ref_dp_field_to_use is not None and 'DP' in vmt.entry:
        dp_exprs['dp'] = hl.tuple(hl.agg.count_where(vmt.DP >= x) for x in dp_bins)

    gq_dp_exprs = hl.struct(**{'gq': hl.tuple(hl.agg.count_where(vmt.GQ >= x) for x in gq_bins)}, **dp_exprs)

    result_struct = hl.rbind(
        hl.struct(**bound_exprs),
        lambda x: hl.rbind(
            hl.struct(**{
                'gq_dp_exprs': gq_dp_exprs,
                'n_het': x.n_het,
                'n_hom_var': x.n_hom_var,
                'n_non_ref': x.n_het + x.n_hom_var,
                'n_singleton': x.n_singleton,
                'n_singleton_ti': x.n_singleton_ti,
                'n_singleton_tv': x.n_singleton_tv,
                'n_snp': (
                    x.allele_type_counts[allele_ints['Transition']] + x.allele_type_counts[allele_ints['Transversion']]
                ),
                'n_insertion': x.allele_type_counts[allele_ints['Insertion']],
                'n_deletion': x.allele_type_counts[allele_ints['Deletion']],
                'n_transition': x.allele_type_counts[allele_ints['Transition']],
                'n_transversion': x.allele_type_counts[allele_ints['Transversion']],
                'n_star': x.allele_type_counts[allele_ints['Star']],
            }),
            lambda s: s.annotate(
                r_ti_tv=divide_null(hl.float64(s.n_transition), s.n_transversion),
                r_ti_tv_singleton=divide_null(hl.float64(s.n_singleton_ti), s.n_singleton_tv),
                r_het_hom_var=divide_null(hl.float64(s.n_het), s.n_hom_var),
                r_insertion_deletion=divide_null(hl.float64(s.n_insertion), s.n_deletion),
            ),
        ),
    )
    variant_results = vmt.select_cols(**result_struct).cols()

    rmt = vds.reference_data

    ref_dp_expr = {}
    if ref_dp_field_to_use is not None:
        ref_dp_expr['ref_bases_over_dp_threshold'] = hl.tuple(
            hl.agg.filter(rmt[ref_dp_field_to_use] >= x, hl.agg.sum(1 + rmt.END - rmt.locus.position)) for x in dp_bins
        )
    ref_results = rmt.select_cols(
        ref_bases_over_gq_threshold=hl.tuple(
            hl.agg.filter(rmt.GQ >= x, hl.agg.sum(1 + rmt.END - rmt.locus.position)) for x in gq_bins
        ),
        **ref_dp_expr,
    ).cols()

    joined = ref_results[variant_results.key]

    joined_dp_expr = {}
    dp_bins_field = {}
    if ref_dp_field_to_use is not None:
        joined_dp_expr['bases_over_dp_threshold'] = hl.tuple(
            x + y for x, y in zip(variant_results.gq_dp_exprs.dp, joined.ref_bases_over_dp_threshold)
        )
        dp_bins_field['dp_bins'] = hl.tuple(dp_bins)

    joined_results = variant_results.transmute(
        bases_over_gq_threshold=hl.tuple(
            x + y for x, y in zip(variant_results.gq_dp_exprs.gq, joined.ref_bases_over_gq_threshold)
        ),
        **joined_dp_expr,
    )

    joined_results = joined_results.annotate_globals(gq_bins=hl.tuple(gq_bins), **dp_bins_field)
    return joined_results


@typecheck(vds=VariantDataset, samples=oneof(Table, expr_array(expr_str)), keep=bool, remove_dead_alleles=bool)
def filter_samples(
    vds: 'VariantDataset', samples, *, keep: bool = True, remove_dead_alleles: bool = False
) -> 'VariantDataset':
    """Filter samples in a :class:`.VariantDataset`.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    samples : :class:`.Table` or list of str
        Samples to keep or remove.
    keep : :obj:`bool`
        Whether to keep (default), or filter out the samples from `samples_table`.
    remove_dead_alleles : :obj:`bool`
        If true, remove alleles observed in no samples. Alleles with AC == 0 will be
        removed, and LA values recalculated.

    Returns
    -------
    :class:`.VariantDataset`
    """
    if not isinstance(samples, hl.Table):
        samples = hl.Table.parallelize(samples.map(lambda s: hl.struct(s=s)), key='s')
    if not list(samples[x].dtype for x in samples.key) == [hl.tstr]:
        raise TypeError(f'invalid key: {samples.key.dtype}')
    samples_to_keep = samples.aggregate(hl.agg.collect_as_set(samples.key[0]), _localize=False)._persist()
    reference_data = vds.reference_data.filter_cols(samples_to_keep.contains(vds.reference_data.col_key[0]), keep=keep)
    reference_data = reference_data.filter_rows(hl.agg.count() > 0)
    variant_data = vds.variant_data.filter_cols(samples_to_keep.contains(vds.variant_data.col_key[0]), keep=keep)

    if remove_dead_alleles:
        vd = variant_data
        vd = vd.annotate_rows(__allele_counts=hl.agg.explode(lambda x: hl.agg.counter(x), vd.LA), __n=hl.agg.count())
        vd = vd.filter_rows(vd.__n > 0)
        vd = vd.drop('__n')

        vd = vd.annotate_rows(
            __kept_indices=hl.dict(
                hl.enumerate(
                    hl.range(hl.len(vd.alleles)).filter(lambda idx: (idx == 0) | (vd.__allele_counts.get(idx, 0) > 0)),
                    index_first=False,
                )
            )
        )

        vd = vd.annotate_rows(
            __old_to_new_LA=hl.range(hl.len(vd.alleles)).map(lambda idx: vd.__kept_indices.get(idx, -1))
        )

        def new_la_index(old_idx):
            raw_idx = vd.__old_to_new_LA[old_idx]
            return (
                hl.case()
                .when(raw_idx >= 0, raw_idx)
                .or_error("'filter_samples': unexpected local allele: old index=" + hl.str(old_idx))
            )

        vd = vd.annotate_entries(LA=vd.LA.map(lambda la: new_la_index(la)))
        vd = vd.key_rows_by('locus')
        vd = vd.annotate_rows(alleles=vd.__kept_indices.keys().map(lambda i: vd.alleles[i]))
        vd = vd._key_rows_by_assert_sorted('locus', 'alleles')
        vd = vd.drop('__allele_counts', '__kept_indices', '__old_to_new_LA')
        return VariantDataset(reference_data, vd)

    variant_data = variant_data.filter_rows(hl.agg.count() > 0)
    return VariantDataset(reference_data, variant_data)


@typecheck(mt=MatrixTable, normalization_contig=str)
def impute_sex_chr_ploidy_from_interval_coverage(
    mt: 'MatrixTable',
    normalization_contig: str,
) -> 'Table':
    """Impute sex chromosome ploidy from a precomputed interval coverage MatrixTable.

    The input MatrixTable must have the following row fields:

     - ``interval`` (*interval*): Genomic interval of interest.
     - ``interval_size`` (*int32*): Size of interval, in bases.

    And the following entry fields:

     -  ``sum_dp`` (*int64*): Sum of depth values by base across the interval.

    Returns a :class:`.Table` with sample ID keys, with the following fields:

     -  ``autosomal_mean_dp`` (*float64*): Mean depth on calling intervals on normalization contig.
     -  ``x_mean_dp`` (*float64*): Mean depth on calling intervals on X chromosome.
     -  ``x_ploidy`` (*float64*): Estimated ploidy on X chromosome. Equal to ``2 * x_mean_dp / autosomal_mean_dp``.
     -  ``y_mean_dp`` (*float64*): Mean depth on calling intervals on  chromosome.
     -  ``y_ploidy`` (*float64*): Estimated ploidy on Y chromosome. Equal to ``2 * y_mean_db / autosomal_mean_dp``.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        Interval-by-sample MatrixTable with sum of depth values across the interval.
    normalization_contig : str
        Autosomal contig for depth comparison.

    Returns
    -------
    :class:`.Table`
    """

    rg = mt.interval.start.dtype.reference_genome

    if len(rg.x_contigs) != 1:
        raise NotImplementedError(
            f"reference genome {rg.name!r} has multiple X contigs, this is not supported in 'impute_sex_chr_ploidy_from_interval_coverage'"
        )
    chr_x = rg.x_contigs[0]
    if len(rg.y_contigs) != 1:
        raise NotImplementedError(
            f"reference genome {rg.name!r} has multiple Y contigs, this is not supported in 'impute_sex_chr_ploidy_from_interval_coverage'"
        )
    chr_y = rg.y_contigs[0]

    mt = mt.annotate_rows(contig=mt.interval.start.contig)
    mt = mt.annotate_cols(__mean_dp=hl.agg.group_by(mt.contig, hl.agg.sum(mt.sum_dp) / hl.agg.sum(mt.interval_size)))

    mean_dp_dict = mt.__mean_dp
    auto_dp = mean_dp_dict.get(normalization_contig, 0.0)
    x_dp = mean_dp_dict.get(chr_x, 0.0)
    y_dp = mean_dp_dict.get(chr_y, 0.0)
    per_sample = mt.transmute_cols(
        autosomal_mean_dp=auto_dp,
        x_mean_dp=x_dp,
        x_ploidy=2 * x_dp / auto_dp,
        y_mean_dp=y_dp,
        y_ploidy=2 * y_dp / auto_dp,
    )
    info("'impute_sex_chromosome_ploidy': computing and checkpointing coverage and karyotype metrics")
    return per_sample.cols().checkpoint(new_temp_file('impute_sex_karyotype', extension='ht'))


@typecheck(
    vds=VariantDataset,
    calling_intervals=oneof(Table, expr_array(expr_interval(expr_locus()))),
    normalization_contig=str,
    use_variant_dataset=bool,
)
def impute_sex_chromosome_ploidy(
    vds: VariantDataset, calling_intervals, normalization_contig: str, use_variant_dataset: bool = False
) -> hl.Table:
    """Impute sex chromosome ploidy from depth of reference or variant data within calling intervals.

    Returns a :class:`.Table` with sample ID keys, with the following fields:

     -  ``autosomal_mean_dp`` (*float64*): Mean depth on calling intervals on normalization contig.
     -  ``x_mean_dp`` (*float64*): Mean depth on calling intervals on X chromosome.
     -  ``x_ploidy`` (*float64*): Estimated ploidy on X chromosome. Equal to ``2 * x_mean_dp / autosomal_mean_dp``.
     -  ``y_mean_dp`` (*float64*): Mean depth on calling intervals on  chromosome.
     -  ``y_ploidy`` (*float64*): Estimated ploidy on Y chromosome. Equal to ``2 * y_mean_db / autosomal_mean_dp``.

    Parameters
    ----------
    vds : vds: :class:`.VariantDataset`
        Dataset.
    calling_intervals : :class:`.Table` or :class:`.ArrayExpression`
        Calling intervals with consistent read coverage (for exomes, trim the capture intervals).
    normalization_contig : str
        Autosomal contig for depth comparison.
    use_variant_dataset : bool
        Whether to use depth of variant data within calling intervals instead of reference data. Default will use reference data.

    Returns
    -------
    :class:`.Table`
    """

    if not isinstance(calling_intervals, Table):
        calling_intervals = hl.Table.parallelize(
            hl.map(lambda i: hl.struct(interval=i), calling_intervals),
            schema=hl.tstruct(interval=calling_intervals.dtype.element_type),
            key='interval',
        )
    else:
        key_dtype = calling_intervals.key.dtype
        if (
            len(key_dtype) != 1
            or not isinstance(calling_intervals.key[0].dtype, hl.tinterval)
            or calling_intervals.key[0].dtype.point_type != vds.reference_data.locus.dtype
        ):
            raise ValueError(
                f"'impute_sex_chromosome_ploidy': expect calling_intervals to be list of intervals or"
                f" table with single key of type interval<locus>, found table with key: {key_dtype}"
            )

    rg = vds.reference_data.locus.dtype.reference_genome

    par_boundaries = []
    for par_interval in rg.par:
        par_boundaries.append(par_interval.start)
        par_boundaries.append(par_interval.end)

    # segment on PAR interval boundaries
    calling_intervals = hl.segment_intervals(calling_intervals, par_boundaries)

    # remove intervals overlapping PAR
    calling_intervals = calling_intervals.filter(
        hl.all(lambda x: ~x.overlaps(calling_intervals.interval), hl.literal(rg.par))
    )

    # checkpoint for efficient multiple downstream usages
    info("'impute_sex_chromosome_ploidy': checkpointing calling intervals")
    calling_intervals = calling_intervals.checkpoint(new_temp_file(extension='ht'))

    interval = calling_intervals.key[0]
    (any_bad_intervals, chrs_represented) = calling_intervals.aggregate((
        hl.agg.any(interval.start.contig != interval.end.contig),
        hl.agg.collect_as_set(interval.start.contig),
    ))
    if any_bad_intervals:
        raise ValueError(
            "'impute_sex_chromosome_ploidy' does not support calling intervals that span chromosome boundaries"
        )

    if len(rg.x_contigs) != 1:
        raise NotImplementedError(
            f"reference genome {rg.name!r} has multiple X contigs, this is not supported in 'impute_sex_chromosome_ploidy'"
        )
    if len(rg.y_contigs) != 1:
        raise NotImplementedError(
            f"reference genome {rg.name!r} has multiple Y contigs, this is not supported in 'impute_sex_chromosome_ploidy'"
        )

    kept_contig_filter = hl.array(chrs_represented).map(lambda x: hl.parse_locus_interval(x, reference_genome=rg))
    vds = VariantDataset(
        hl.filter_intervals(vds.reference_data, kept_contig_filter),
        hl.filter_intervals(vds.variant_data, kept_contig_filter),
    )

    if use_variant_dataset:
        mt = vds.variant_data
        calling_intervals = calling_intervals.annotate(interval_dup=interval)
        mt = mt.annotate_rows(interval=calling_intervals[mt.locus].interval_dup)
        mt = mt.filter_rows(hl.is_defined(mt.interval))
        coverage = mt.select_entries(sum_dp=mt.DP, interval_size=hl.is_defined(mt.DP))
    else:
        coverage = interval_coverage(vds, calling_intervals, gq_thresholds=()).drop('gq_thresholds')

    return impute_sex_chr_ploidy_from_interval_coverage(coverage, normalization_contig)


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


@typecheck(
    vds=VariantDataset,
    intervals=oneof(Table, expr_array(expr_interval(expr_any))),
    keep=bool,
    mode=enumeration('variants_only', 'split_at_boundaries', 'unchecked_filter_both'),
)
def _parameterized_filter_intervals(vds: 'VariantDataset', intervals, keep: bool, mode: str) -> 'VariantDataset':
    intervals_table = None
    if isinstance(intervals, Table):
        expected = hl.tinterval(hl.tlocus(vds.reference_genome))
        if len(intervals.key) != 1 or intervals.key[0].dtype != hl.tinterval(hl.tlocus(vds.reference_genome)):
            raise ValueError(
                f"'filter_intervals': expect a table with a single key of type {expected}; "
                f"found {list(intervals.key.dtype.values())}"
            )
        intervals_table = intervals
        intervals = hl.literal(intervals.aggregate(hl.agg.collect(intervals.key[0]), _localize=False))

    if mode == 'unchecked_filter_both':
        return VariantDataset(
            hl.filter_intervals(vds.reference_data, intervals, keep),
            hl.filter_intervals(vds.variant_data, intervals, keep),
        )

    reference_data = vds.reference_data
    if keep:
        rbml = hl.vds.VariantDataset.ref_block_max_length_field
        if rbml in vds.reference_data.globals:
            max_len = hl.eval(vds.reference_data.index_globals()[rbml])
            ref_intervals = intervals.map(
                lambda interval: hl.interval(
                    interval.start - (max_len - 1), interval.end, interval.includes_start, interval.includes_end
                )
            )
            reference_data = hl.filter_intervals(reference_data, ref_intervals, keep)
        else:
            warning(
                "'hl.vds.filter_intervals': filtering intervals without a known max reference block length"
                "\n  (computed by `hl.vds.store_ref_block_max_length` or 'hl.vds.truncate_reference_blocks')"
                "\n  requires a full pass over the reference data (expensive!)"
            )

    if mode == 'variants_only':
        variant_data = hl.filter_intervals(vds.variant_data, intervals, keep)
        return VariantDataset(reference_data, variant_data)
    if mode == 'split_at_boundaries':
        if not keep:
            raise ValueError("filter_intervals mode 'split_at_boundaries' not implemented for keep=False")
        par_intervals = intervals_table or hl.Table.parallelize(
            intervals.map(lambda x: hl.struct(interval=x)),
            schema=hl.tstruct(interval=intervals.dtype.element_type),
            key='interval',
        )
        ref = segment_reference_blocks(reference_data, par_intervals).drop('interval_end', next(iter(par_intervals.key)))
        return VariantDataset(ref, hl.filter_intervals(vds.variant_data, intervals, keep))


@typecheck(
    vds=VariantDataset,
    keep=nullable(oneof(str, sequenceof(str))),
    remove=nullable(oneof(str, sequenceof(str))),
    keep_autosomes=bool,
)
def filter_chromosomes(vds: 'VariantDataset', *, keep=None, remove=None, keep_autosomes=False) -> 'VariantDataset':
    """Filter chromosomes of a :class:`.VariantDataset` in several possible modes.

    Notes
    -----
    There are three modes for :func:`filter_chromosomes`, based on which argument is passed
    to the function. Exactly one of the below arguments must be passed by keyword.

     - ``keep``: This argument expects a single chromosome identifier or a list of chromosome
       identifiers, and the function returns a :class:`.VariantDataset` with only those
       chromosomes.
     - ``remove``: This argument expects a single chromosome identifier or a list of chromosome
       identifiers, and the function returns a :class:`.VariantDataset` with those chromosomes
       removed.
     - ``keep_autosomes``: This argument expects the value ``True``, and returns a dataset without
       sex and mitochondrial chromosomes.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset.
    keep
        Keep a specified list of contigs.
    remove
        Remove a specified list of contigs
    keep_autosomes
        If true, keep only autosomal chromosomes.

    Returns
    -------
    :class:`.VariantDataset`.
    """

    n_args_passed = (keep is not None) + (remove is not None) + keep_autosomes
    if n_args_passed == 0:
        raise ValueError("filter_chromosomes: expect one of 'keep', 'remove', or 'keep_autosomes' arguments")
    if n_args_passed > 1:
        raise ValueError(
            "filter_chromosomes: expect ONLY one of 'keep', 'remove', or 'keep_autosomes' arguments"
            "\n  In order use 'keep_autosomes' with 'keep' or 'remove', call the function twice"
        )

    rg = vds.reference_genome

    to_keep = []

    if keep is not None:
        keep = wrap_to_list(keep)
        to_keep.extend(keep)
    elif remove is not None:
        remove = set(wrap_to_list(remove))
        for c in rg.contigs:
            if c not in remove:
                to_keep.append(c)
    elif keep_autosomes:
        to_remove = set(rg.x_contigs + rg.y_contigs + rg.mt_contigs)
        for c in rg.contigs:
            if c not in to_remove:
                to_keep.append(c)

    parsed_intervals = hl.literal(to_keep, hl.tarray(hl.tstr)).map(
        lambda c: hl.parse_locus_interval(c, reference_genome=rg)
    )
    return _parameterized_filter_intervals(vds, intervals=parsed_intervals, keep=True, mode='unchecked_filter_both')


@typecheck(
    vds=VariantDataset,
    intervals=oneof(Table, expr_array(expr_interval(expr_any))),
    split_reference_blocks=bool,
    keep=bool,
)
def filter_intervals(
    vds: 'VariantDataset', intervals, *, split_reference_blocks: bool = False, keep: bool = True
) -> 'VariantDataset':
    """Filter intervals in a :class:`.VariantDataset`.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    intervals : :class:`.Table` or :class:`.ArrayExpression` of type :class:`.tinterval`
        Intervals to filter on.
    split_reference_blocks: :obj:`bool`
        If true, remove reference data outside the given intervals by segmenting reference
        blocks at interval boundaries. Results in a smaller result, but this filter mode
        is more computationally expensive to evaluate.
    keep : :obj:`bool`
        Whether to keep, or filter out (default) rows that fall within any
        interval in `intervals`.

    Returns
    -------
    :class:`.VariantDataset`
    """
    if split_reference_blocks and not keep:
        raise ValueError("'filter_intervals': cannot use 'split_reference_blocks' with keep=False")
    return _parameterized_filter_intervals(
        vds, intervals, keep=keep, mode='split_at_boundaries' if split_reference_blocks else 'variants_only'
    )


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
    interval_field = next(iter(intervals.key))
    if not intervals[interval_field].dtype == hl.tinterval(ref.locus.dtype):
        raise ValueError(
            f"expect intervals to be keyed by intervals of loci matching the VariantDataset:"
            f" found {intervals[interval_field].dtype} / {ref.locus.dtype}"
        )
    intervals = intervals.select(_interval_dup=intervals[interval_field])

    if not intervals.aggregate(
        hl.agg.all(
            intervals[interval_field].includes_start
            & (intervals[interval_field].start.contig == intervals[interval_field].end.contig)
        )
    ):
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
        _ref_entries=joined._ref_entries.map(lambda e: e.annotate(__contig_idx=joined.__contig_idx))
    )
    dense = joined.annotate(
        dense_ref=hl.or_missing(
            joined._include_locus,
            hl.rbind(
                joined.locus.position,
                lambda pos: hl.enumerate(hl.scan._densify(hl.len(joined._ref_cols), joined._ref_entries)).map(
                    lambda idx_and_e: hl.rbind(
                        idx_and_e[0],
                        idx_and_e[1],
                        lambda idx, e: hl.coalesce(
                            joined._ref_entries[idx],
                            hl.or_missing((e.__contig_idx == joined.__contig_idx) & (e.END >= pos), e),
                        ),
                    ).drop('__contig_idx')
                ),
            ),
        )
    )
    dense = dense.filter(dense._include_locus).drop('_interval_dup', '_include_locus', '__contig_idx')

    # at this point, 'dense' is a table with dense rows of reference blocks, keyed by locus

    refl_filtered = refl.annotate(**{interval_field: intervals[refl.locus]._interval_dup})

    # remove rows that are not contained in an interval, and rows that are the start of an
    # interval (interval starts come from the 'dense' table)
    refl_filtered = refl_filtered.filter(
        hl.is_defined(refl_filtered[interval_field]) & (refl_filtered.locus != refl_filtered[interval_field].start)
    )

    # union dense interval starts with filtered table
    refl_filtered = refl_filtered.union(dense.transmute(_ref_entries=dense.dense_ref))

    # rewrite reference blocks to end at the first of (interval end, reference block end)
    refl_filtered = refl_filtered.annotate(
        interval_end=refl_filtered[interval_field].end.position - ~refl_filtered[interval_field].includes_end
    )
    refl_filtered = refl_filtered.annotate(
        _ref_entries=refl_filtered._ref_entries.map(
            lambda entry: entry.annotate(END=hl.min(entry.END, refl_filtered.interval_end))
        )
    )

    return refl_filtered._unlocalize_entries('_ref_entries', '_ref_cols', list(ref.col_key))


@typecheck(
    vds=VariantDataset,
    intervals=Table,
    gq_thresholds=sequenceof(int),
    dp_thresholds=sequenceof(int),
    dp_field=nullable(str),
)
def interval_coverage(
    vds: VariantDataset,
    intervals: hl.Table,
    gq_thresholds=(
        0,
        10,
        20,
    ),
    dp_thresholds=(0, 1, 10, 20, 30),
    dp_field=None,
) -> 'MatrixTable':
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
     -  ``bases_over_dp_threshold`` (*tuple of int64*): Number of bases in the interval
        over each DP threshold.
     -  ``fraction_over_dp_threshold`` (*tuple of float64*): Fraction of interval (in bases)
        above each DP threshold. Computed by dividing each member of *bases_over_dp_threshold*
        by *interval_size*.
     -  ``sum_dp`` (*int64*): Sum of depth values by base across the interval.
     -  ``mean_dp`` (*float64*): Mean depth of bases across the interval. Computed by dividing
        *sum_dp* by *interval_size*.

    If the `dp_field` parameter is not specified, the ``DP`` is used for depth
    if present. If no ``DP`` field is present, the ``MIN_DP`` field is used. If no ``DP``
    or ``MIN_DP`` field is present, no depth statistics will be calculated.

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

    ref_block_length = split.END - split.locus.position + 1
    if dp_field_to_use is not None:
        dp = split[dp_field_to_use]
        dp_field_dict = {
            'sum_dp': hl.agg.sum(ref_block_length * dp),
            'bases_over_dp_threshold': tuple(
                hl.agg.filter(dp >= dp_threshold, hl.agg.sum(ref_block_length)) for dp_threshold in dp_thresholds
            ),
        }
    else:
        dp_field_dict = dict()

    per_interval = split.group_rows_by(interval=intervals[split.row_key[0]].interval_dup).aggregate(
        bases_over_gq_threshold=tuple(
            hl.agg.filter(split.GQ >= gq_threshold, hl.agg.sum(ref_block_length)) for gq_threshold in gq_thresholds
        ),
        **dp_field_dict,
    )

    interval = per_interval.interval
    interval_size = (
        interval.end.position + interval.includes_end - interval.start.position - 1 + interval.includes_start
    )
    per_interval = per_interval.annotate_rows(interval_size=interval_size)

    dp_mod_dict = {}
    if dp_field_to_use is not None:
        dp_mod_dict['fraction_over_dp_threshold'] = tuple(
            hl.float(x) / per_interval.interval_size for x in per_interval.bases_over_dp_threshold
        )
        dp_mod_dict['mean_dp'] = per_interval.sum_dp / per_interval.interval_size

    per_interval = per_interval.annotate_entries(
        fraction_over_gq_threshold=tuple(
            hl.float(x) / per_interval.interval_size for x in per_interval.bases_over_gq_threshold
        ),
        **dp_mod_dict,
    )

    per_interval = per_interval.annotate_globals(gq_thresholds=hl.tuple(gq_thresholds))

    return per_interval


@typecheck(
    ds=oneof(MatrixTable, VariantDataset),
    max_ref_block_base_pairs=nullable(int),
    ref_block_winsorize_fraction=nullable(float),
)
def truncate_reference_blocks(ds, *, max_ref_block_base_pairs=None, ref_block_winsorize_fraction=None):
    """Cap reference blocks at a maximum length in order to permit faster interval filtering.

    Examples
    --------
    Truncate reference blocks to 5 kilobases:

    >>> vds2 = hl.vds.truncate_reference_blocks(vds, max_ref_block_base_pairs=5000) # doctest: +SKIP

    Truncate the longest 1% of reference blocks to the length of the 99th percentile block:

    >>> vds2 = hl.vds.truncate_reference_blocks(vds, ref_block_winsorize_fraction=0.01) # doctest: +SKIP

    Notes
    -----
    After this function has been run, the reference blocks have a known maximum length `ref_block_max_length`,
    stored in the global fields, which permits :func:`.vds.filter_intervals` to filter to intervals of the reference
    data by reading `ref_block_max_length` bases ahead of each interval. This allows narrow interval queries
    to run in roughly O(data kept) work rather than O(all reference data) work.

    It is also possible to patch an existing VDS to store the max reference block length with :func:`.vds.store_ref_block_max_length`.

    See Also
    --------
    :func:`.vds.store_ref_block_max_length`.

    Parameters
    ----------
    vds : :class:`.VariantDataset` or :class:`.MatrixTable`
    max_ref_block_base_pairs
        Maximum size of reference blocks, in base pairs.
    ref_block_winsorize_fraction
        Fraction of reference block length distribution to truncate / winsorize.

    Returns
    -------
    :class:`.VariantDataset` or :class:`.MatrixTable`
    """
    if isinstance(ds, VariantDataset):
        rd = ds.reference_data
    else:
        rd = ds

    fd_name = hl.vds.VariantDataset.ref_block_max_length_field
    if fd_name in rd.globals:
        rd = rd.drop(fd_name)

    if int(ref_block_winsorize_fraction is None) + int(max_ref_block_base_pairs is None) != 1:
        raise ValueError(
            'truncate_reference_blocks: require exactly one of "max_ref_block_base_pairs", "ref_block_winsorize_fraction"'
        )

    if ref_block_winsorize_fraction is not None:
        assert (
            ref_block_winsorize_fraction > 0 and ref_block_winsorize_fraction < 1
        ), 'truncate_reference_blocks: "ref_block_winsorize_fraction" must be between 0 and 1 (e.g. 0.01 to truncate the top 1% of reference blocks)'
        if ref_block_winsorize_fraction > 0.1:
            warning(
                f"'truncate_reference_blocks': ref_block_winsorize_fraction of {ref_block_winsorize_fraction} will lead to significant data duplication,"
                f" recommended values are <0.05."
            )
        max_ref_block_base_pairs = rd.aggregate_entries(
            hl.agg.approx_quantiles(rd.END - rd.locus.position + 1, 1 - ref_block_winsorize_fraction, k=200)
        )

    assert (
        max_ref_block_base_pairs > 0
    ), 'truncate_reference_blocks: "max_ref_block_base_pairs" must be between greater than zero'
    info(f"splitting VDS reference blocks at {max_ref_block_base_pairs} base pairs")

    rd_under_limit = rd.filter_entries(rd.END - rd.locus.position < max_ref_block_base_pairs).localize_entries(
        'fixed_blocks', 'cols'
    )

    rd_over_limit = rd.filter_entries(rd.END - rd.locus.position >= max_ref_block_base_pairs).key_cols_by(
        col_idx=hl.scan.count()
    )
    rd_over_limit = rd_over_limit.select_rows().select_cols().key_rows_by().key_cols_by()
    es = rd_over_limit.entries()
    es = es.annotate(new_start=hl.range(es.locus.position, es.END + 1, max_ref_block_base_pairs))
    es = es.explode('new_start')
    es = es.transmute(
        locus=hl.locus(es.locus.contig, es.new_start, reference_genome=es.locus.dtype.reference_genome),
        END=hl.min(es.new_start + max_ref_block_base_pairs - 1, es.END),
    )
    es = es.key_by(es.locus).collect_by_key("new_blocks")
    es = es.transmute(moved_blocks_dict=hl.dict(es.new_blocks.map(lambda x: (x.col_idx, x.drop('col_idx')))))

    joined = rd_under_limit.join(es, how='outer')
    joined = joined.transmute(
        merged_blocks=hl.range(hl.len(joined.cols)).map(
            lambda idx: hl.coalesce(joined.moved_blocks_dict.get(idx), joined.fixed_blocks[idx])
        )
    )
    new_rd = joined._unlocalize_entries(
        entries_field_name='merged_blocks', cols_field_name='cols', col_key=list(rd.col_key)
    )
    new_rd = new_rd.annotate_globals(**{fd_name: max_ref_block_base_pairs})

    if isinstance(ds, hl.vds.VariantDataset):
        return VariantDataset(reference_data=new_rd, variant_data=ds.variant_data)
    return new_rd


@typecheck(
    ds=oneof(MatrixTable, VariantDataset),
    equivalence_function=func_spec(2, expr_bool),
    merge_functions=nullable(dictof(str, oneof(str, func_spec(1, expr_any)))),
)
def merge_reference_blocks(ds, equivalence_function, merge_functions=None):
    """Merge adjacent reference blocks according to user equivalence criteria.

    Examples
    --------
    Coarsen GQ granularity into bins of 10 and merges blocks with the same GQ in order to
    compress reference data.

    >>> rd = vds.reference_data # doctest: +SKIP
    >>> vds.reference_data = rd.annotate_entries(GQ = rd.GQ - rd.GQ % 10) # doctest: +SKIP
    >>> vds2 = hl.vds.merge_reference_blocks(vds,
    ...                                      equivalence_function=lambda block1, block2: block1.GQ == block2.GQ),
    ...                                      merge_functions={'MIN_DP': 'min'}) # doctest: +SKIP

    Notes
    -----
    The `equivalence_function` argument expects a function from two reference blocks to a
    boolean value indicating whether they should be combined. Adjacency checks are builtin
    to the method (two reference blocks are 'adjacent' if the END of one block is one base
    before the beginning of the next).

    The `merge_functions`

    Parameters
    ----------
    ds : :class:`.VariantDataset` or :class:`.MatrixTable`
        Variant dataset or reference block matrix table.
    Returns
    -------
    :class:`.VariantDataset` or :class:`.MatrixTable`
    """
    if isinstance(ds, VariantDataset):
        rd = ds.reference_data
    else:
        rd = ds
    rd = rd.annotate_rows(contig_idx_row=rd.locus.contig_idx, start_pos_row=rd.locus.position)
    rd = rd.annotate_entries(contig_idx=rd.contig_idx_row, start_pos=rd.start_pos_row)
    ht = rd.localize_entries('entries', 'cols')

    def merge(block1, block2):
        new_fields = {'END': block2.END}
        if merge_functions:
            for k, f in merge_functions.items():
                if isinstance(f, str):
                    f = f.lower()
                    if f == 'min':

                        def f(b1, b2):
                            return hl.min(block1[k], block2[k])

                    elif f == 'max':

                        def f(b1, b2):
                            return hl.max(block1[k], block2[k])

                    elif f == 'sum':

                        def f(b1, b2):
                            return block1[k] + block2[k]

                    else:
                        raise ValueError(
                            f"merge_reference_blocks: unknown merge function {f!r},"
                            f" support 'min', 'max', and 'sum' in addition to custom lambdas"
                        )
                new_value = f(block1, block2)
                if new_value.dtype != block1[k].dtype:
                    raise ValueError(
                        f'merge_reference_blocks: merge_function for {k!r}: new type {new_value.dtype!r} '
                        f'differs from original type {block1[k].dtype!r}'
                    )
                new_fields[k] = new_value
        return block1.annotate(**new_fields)

    def keep_last(t1, t2):
        e1 = t1[0]
        e2 = t2[0]
        are_adjacent = (e1.contig_idx == e2.contig_idx) & (e1.END + 1 == e2.start_pos)
        return hl.if_else(
            hl.is_defined(e1) & hl.is_defined(e2) & are_adjacent & equivalence_function(e1, e2),
            (merge(e1, e2), True),
            t2,
        )

    # approximate a scan that merges before result
    ht = ht.annotate(
        prev_block=hl.zip(
            hl.scan.array_agg(
                lambda elt: hl.scan.fold(
                    (hl.missing(rd.entry.dtype), False), lambda acc: keep_last(acc, (elt, False)), keep_last
                ),
                ht.entries,
            ),
            ht.entries,
        ).map(lambda tup: keep_last(tup[0], (tup[1], False)))
    )
    ht_join = ht

    ht = ht.key_by()
    ht = ht.select(
        to_shuffle=hl.enumerate(ht.prev_block).filter(
            lambda idx_and_elt: hl.is_defined(idx_and_elt[1]) & idx_and_elt[1][1]
        )
    )
    ht = ht.explode('to_shuffle')
    rg = rd.locus.dtype.reference_genome
    ht = ht.transmute(col_idx=ht.to_shuffle[0], entry=ht.to_shuffle[1][0])
    ht_shuf = ht.key_by(
        locus=hl.locus(hl.literal(rg.contigs)[ht.entry.contig_idx], ht.entry.start_pos, reference_genome=rg)
    )

    ht_shuf = ht_shuf.collect_by_key("new_starts")
    # new_starts can contain multiple records for a collapsed ref block, one for each folded block.
    # We want to keep the one with the highest END
    ht_shuf = ht_shuf.select(
        moved_blocks_dict=hl.group_by(lambda elt: elt.col_idx, ht_shuf.new_starts).map_values(
            lambda arr: arr[hl.argmax(arr.map(lambda x: x.entry.END))].entry.drop('contig_idx', 'start_pos')
        )
    )

    ht_joined = ht_join.join(ht_shuf.select_globals(), 'left')

    def merge_f(tup):
        (idx, original_entry) = tup

        return (
            hl.case()
            .when(
                ~(hl.coalesce(ht_joined.prev_block[idx][1], False)),
                hl.coalesce(ht_joined.moved_blocks_dict.get(idx), original_entry.drop('contig_idx', 'start_pos')),
            )
            .or_missing()
        )

    ht_joined = ht_joined.annotate(new_entries=hl.enumerate(ht_joined.entries).map(lambda tup: merge_f(tup)))
    ht_joined = ht_joined.drop('moved_blocks_dict', 'entries', 'prev_block', 'contig_idx_row', 'start_pos_row')
    new_rd = ht_joined._unlocalize_entries(
        entries_field_name='new_entries', cols_field_name='cols', col_key=list(rd.col_key)
    )

    rbml = hl.vds.VariantDataset.ref_block_max_length_field
    if rbml in new_rd.globals:
        new_rd = new_rd.drop(rbml)

    if isinstance(ds, VariantDataset):
        return VariantDataset(reference_data=new_rd, variant_data=ds.variant_data)
    return new_rd
