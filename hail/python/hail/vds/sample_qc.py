from typing import Optional, Sequence

import hail as hl
from hail.expr.expressions import Expression
from hail.expr.expressions.typed_expressions import (
    ArrayExpression,
    CallExpression,
    LocusExpression,
    NumericExpression,
    StructExpression,
)
from hail.methods.misc import require_first_key_field_locus
from hail.table import Table
from hail.typecheck import sequenceof, typecheck, nullable
from hail.utils.java import Env
from hail.utils.misc import divide_null
from hail.vds.variant_dataset import VariantDataset


@typecheck(global_gt=Expression, alleles=ArrayExpression)
def vmt_sample_qc_variant_annotations(
    *,
    global_gt: 'Expression',
    alleles: 'ArrayExpression',
) -> 'StructExpression':
    from hail.expr.functions import _num_allele_type, _allele_types

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

    return hl.struct(
        ac=hl.agg.call_stats(global_gt, alleles).AC,
        atypes=alleles[1:].map(lambda alt: allele_type(alleles[0], alt)),
    )


@typecheck(
    global_gt=Expression,
    gq=Expression,
    variant_ac=ArrayExpression,
    variant_atypes=ArrayExpression,
    dp=nullable(Expression),
    gq_bins=sequenceof(int),
    dp_bins=sequenceof(int),
)
def vmt_sample_qc(
    *,
    global_gt: 'CallExpression',
    gq: 'Expression',
    variant_ac: 'ArrayExpression',
    variant_atypes: 'ArrayExpression',
    dp: Optional['Expression'] = None,
    gq_bins: 'Sequence[int]' = (0, 20, 60),
    dp_bins: 'Sequence[int]' = (0, 1, 10, 20, 30),
) -> 'Expression':
    from hail.expr.functions import _allele_types

    allele_types = _allele_types[:]
    allele_types.extend(['Transition', 'Transversion'])
    allele_enum = dict(enumerate(allele_types))
    allele_ints = {v: k for k, v in allele_enum.items()}

    bound_exprs = {}

    bound_exprs['n_het'] = hl.agg.count_where(global_gt.is_het())
    bound_exprs['n_hom_var'] = hl.agg.count_where(global_gt.is_hom_var())
    bound_exprs['n_singleton'] = hl.agg.sum(
        hl.rbind(
            global_gt,
            lambda global_gt: hl.sum(
                hl.range(0, global_gt.ploidy).map(
                    lambda i: hl.rbind(global_gt[i], lambda gti: (gti != 0) & (variant_ac[gti] == 1))
                )
            ),
        )
    )
    bound_exprs['n_singleton_ti'] = hl.agg.sum(
        hl.rbind(
            global_gt,
            lambda global_gt: hl.sum(
                hl.range(0, global_gt.ploidy).map(
                    lambda i: hl.rbind(
                        global_gt[i],
                        lambda gti: (gti != 0)
                        & (variant_ac[gti] == 1)
                        & (variant_atypes[gti - 1] == allele_ints['Transition']),
                    )
                )
            ),
        )
    )
    bound_exprs['n_singleton_tv'] = hl.agg.sum(
        hl.rbind(
            global_gt,
            lambda global_gt: hl.sum(
                hl.range(0, global_gt.ploidy).map(
                    lambda i: hl.rbind(
                        global_gt[i],
                        lambda gti: (gti != 0)
                        & (variant_ac[gti] == 1)
                        & (variant_atypes[gti - 1] == allele_ints['Transversion']),
                    )
                )
            ),
        )
    )

    bound_exprs['allele_type_counts'] = hl.agg.explode(
        lambda allele_type: hl.tuple(hl.agg.count_where(allele_type == i) for i in range(len(allele_ints))),
        (
            hl.range(0, global_gt.ploidy)
            .map(lambda i: global_gt[i])
            .filter(lambda allele_idx: allele_idx > 0)
            .map(lambda allele_idx: variant_atypes[allele_idx - 1])
        ),
    )

    dp_exprs = {}
    if dp is not None:
        dp_exprs['dp'] = hl.tuple(hl.agg.count_where(dp >= x) for x in dp_bins)

    gq_dp_exprs = hl.struct(**{'gq': hl.tuple(hl.agg.count_where(gq >= x) for x in gq_bins)}, **dp_exprs)

    return hl.rbind(
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


@typecheck(
    locus=LocusExpression,
    gq=NumericExpression,
    end=NumericExpression,
    dp=nullable(Expression),
    gq_bins=sequenceof(int),
    dp_bins=sequenceof(int),
)
def rmt_sample_qc(
    *,
    locus: 'LocusExpression',
    end: 'NumericExpression',
    gq: 'NumericExpression',
    dp: Optional['Expression'] = None,
    gq_bins: 'Sequence[int]' = (0, 20, 60),
    dp_bins: 'Sequence[int]' = (0, 1, 10, 20, 30),
) -> 'StructExpression':
    ref_dp_expr = {}
    if dp is not None:
        ref_dp_expr['ref_bases_over_dp_threshold'] = hl.tuple(
            hl.agg.filter(dp >= x, hl.agg.sum(1 + end - locus.position)) for x in dp_bins
        )
    return hl.struct(
        bases_over_gq_threshold=hl.tuple(hl.agg.filter(gq >= x, hl.agg.sum(1 + end - locus.position)) for x in gq_bins),
        **ref_dp_expr,
    )


def combine_sample_qc(
    rmt_sample_qc: Expression,
    vmt_sample_qc: Expression,
) -> Expression:
    assert 'gq_dp_exprs' in vmt_sample_qc
    assert 'bases_over_gq_threshold' in rmt_sample_qc

    joined_dp_expr = {}
    if 'bases_over_dp_threshold' in vmt_sample_qc:
        joined_dp_expr['bases_over_dp_threshold'] = hl.tuple(
            x + y for x, y in zip(vmt_sample_qc.gq_dp_exprs.dp, rmt_sample_qc.bases_over_dp_threshold)
        )

    return hl.struct(
        bases_over_gq_threshold=hl.tuple(
            x + y for x, y in zip(vmt_sample_qc.gq_dp_exprs.gq, rmt_sample_qc.bases_over_gq_threshold)
        ),
        **joined_dp_expr,
    )


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

    if 'DP' in vds.reference_data.entry:
        ref_dp_field_to_use = 'DP'
    elif 'MIN_DP' in vds.reference_data.entry:
        ref_dp_field_to_use = 'MIN_DP'
    else:
        ref_dp_field_to_use = dp_field

    vmt = vds.variant_data
    if 'GT' not in vmt.entry:
        vmt = vmt.annotate_entries(GT=hl.vds.lgt_to_gt(vmt.LGT, vmt.LA))
    ac_and_atypes = vmt_sample_qc_variant_annotations(global_gt=vmt.GT, alleles=vmt.alleles)
    variant_ac = Env.get_uid()
    variant_atypes = Env.get_uid()
    vmt = vmt.annotate_rows(**{variant_ac: ac_and_atypes.ac, variant_atypes: ac_and_atypes.atypes})
    vmt_dp = vmt['DP'] if ref_dp_field_to_use is not None and 'DP' in vmt.entry else None
    variant_results = vmt.select_cols(
        **vmt_sample_qc(
            global_gt=vmt.GT,
            gq=vmt.GQ,
            variant_ac=vmt[variant_ac],
            variant_atypes=vmt[variant_atypes],
            dp=vmt_dp,
            gq_bins=gq_bins,
            dp_bins=dp_bins,
        )
    ).cols()

    rmt = vds.reference_data
    rmt_dp = rmt[ref_dp_field_to_use] if ref_dp_field_to_use is not None else None
    reference_results = rmt.select_cols(
        **rmt_sample_qc(
            locus=rmt.locus,
            gq=rmt.GQ,
            end=rmt.END,
            dp=rmt_dp,
            gq_bins=gq_bins,
            dp_bins=dp_bins,
        )
    ).cols()

    # TODO factor this out?
    joined = reference_results[variant_results.key]

    joined_dp_expr = {}
    dp_bins_field = {}
    if ref_dp_field_to_use is not None:
        joined_dp_expr['bases_over_dp_threshold'] = hl.tuple(
            x + y for x, y in zip(variant_results.gq_dp_exprs.dp, joined.ref_bases_over_dp_threshold)
        )
        dp_bins_field['dp_bins'] = hl.tuple(dp_bins)

    joined_results = variant_results.transmute(
        bases_over_gq_threshold=hl.tuple(
            x + y for x, y in zip(variant_results.gq_dp_exprs.gq, joined.bases_over_gq_threshold)
        ),
        **joined_dp_expr,
    )

    joined_results = joined_results.annotate_globals(gq_bins=hl.tuple(gq_bins), **dp_bins_field)
    return joined_results
