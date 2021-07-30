from typing import Sequence

import hail as hl
from hail.expr import ArrayExpression, expr_any, expr_array, expr_interval
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_first_key_field_locus
from hail.table import Table
from hail.typecheck import sequenceof, typecheck
from hail.utils.java import Env
from hail.utils.misc import divide_null
from hail.vds.variant_dataset import VariantDataset


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
    gq_bins : :class:`tup` of :obj:`int`
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


@typecheck(vds=VariantDataset, samples_table=Table, keep=bool)
def filter_samples(vds: 'VariantDataset', samples_table: 'Table', *, keep: bool = True) -> 'VariantDataset':
    """Filter samples in a :class:`.VariantDataset`.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in VariantDataset representation.
    samples_table : :class:`.Table`
        Samples to filter on.
    keep : :obj:`bool`
        Whether to keep (default), or filter out the samples from `samples_table`.

    Returns
    -------
    :class:`.VariantDataset`
    """
    if not list(samples_table[x].dtype for x in samples_table.key) == [hl.tstr]:
        raise TypeError(f'invalid key: {samples_table.key.dtype}')
    samples_to_keep = samples_table.aggregate(hl.agg.collect_as_set(samples_table.key[0]), _localize=False)._persist()
    reference_data = vds.reference_data.filter_cols(samples_to_keep.contains(vds.reference_data.key[0]), keep=keep)
    variant_data = vds.variant_data.filter_cols(samples_to_keep.contains(vds.variant_data.key[0]), keep=keep)
    return VariantDataset(reference_data, variant_data)


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
