from typing import Sequence

import hail as hl
from hail.methods.misc import require_first_key_field_locus
from hail.table import Table
from hail.typecheck import sequenceof, typecheck
from hail.utils.java import Env
from hail.utils.misc import divide_null
from hail.vds import VariantDataset


@typecheck(vds=VariantDataset, name=str, gq_bins=sequenceof(int))
def sample_qc(vds, *, name='sample_qc', gq_bins: 'Sequence[int]' = (0, 20, 60)) -> 'Table':
    """Run sample_qc on dataset in the sparse :class:`.VariantDataset` representation.

    Parameters
    ----------
    vds : :class:`.VariantDataset`
        Dataset in sparse variant dataset representation.
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
        return hl.bind(lambda at: hl.if_else(at == allele_ints['SNP'],
                                             hl.if_else(hl.is_transition(ref, alt),
                                                        allele_ints['Transition'],
                                                        allele_ints['Transversion']),
                                             at),
                       _num_allele_type(ref, alt))

    variant_ac = Env.get_uid()
    variant_atypes = Env.get_uid()

    vmt = vds.variant_data
    if 'GT' not in vmt.entry:
        vmt = vmt.annotate_entries(GT=hl.experimental.lgt_to_gt(vmt.LGT, vmt.LA))

    vmt = vmt.annotate_rows(**{variant_ac: hl.agg.call_stats(vmt.GT, vmt.alleles).AC,
                               variant_atypes: vmt.alleles[1:].map(lambda alt: allele_type(vmt.alleles[0], alt))})

    bound_exprs = {}

    bound_exprs['n_het'] = hl.agg.count_where(vmt['GT'].is_het())
    bound_exprs['n_hom_var'] = hl.agg.count_where(vmt['GT'].is_hom_var())
    bound_exprs['n_singleton'] = hl.agg.sum(hl.sum(hl.range(0, vmt['GT'].ploidy).map(lambda i: vmt[variant_ac][vmt['GT'][i]] == 1)))

    def get_allele_type(allele_idx):
        return hl.if_else(allele_idx > 0, vmt[variant_atypes][allele_idx - 1], hl.missing(hl.tint32))

    bound_exprs['allele_type_counts'] = hl.agg.explode(
        lambda elt: hl.agg.counter(elt),
        hl.range(0, vmt['GT'].ploidy).map(lambda i: get_allele_type(vmt['GT'][i])))

    zero = hl.int64(0)

    gq_exprs = hl.agg.filter(hl.is_defined(vmt.GT),
                             hl.struct(**{f'gq_over_{x}': hl.agg.count_where(vmt.GQ > x)
                                          for x in gq_bins}))

    result_struct = hl.rbind(
        hl.struct(**bound_exprs),
        lambda x: hl.rbind(
            hl.struct(**{
                'gq_exprs': gq_exprs,
                'n_het': x.n_het,
                'n_hom_var': x.n_hom_var,
                'n_non_ref': x.n_het + x.n_hom_var,
                'n_singleton': x.n_singleton,
                'n_snp': (x.allele_type_counts.get(allele_ints["Transition"], zero)
                          + x.allele_type_counts.get(allele_ints["Transversion"], zero)),
                'n_insertion': x.allele_type_counts.get(allele_ints["Insertion"], zero),
                'n_deletion': x.allele_type_counts.get(allele_ints["Deletion"], zero),
                'n_transition': x.allele_type_counts.get(allele_ints["Transition"], zero),
                'n_transversion': x.allele_type_counts.get(allele_ints["Transversion"], zero),
                'n_star': x.allele_type_counts.get(allele_ints["Star"], zero)
            }),
            lambda s: s.annotate(
                r_ti_tv=divide_null(hl.float64(s.n_transition), s.n_transversion),
                r_het_hom_var=divide_null(hl.float64(s.n_het), s.n_hom_var),
                r_insertion_deletion=divide_null(hl.float64(s.n_insertion), s.n_deletion)
            )))
    variant_results = vmt.select_cols(**result_struct).cols()

    rmt = vds.reference_data
    ref_results = rmt.select_cols(gq_exprs=hl.struct(**{
        f'gq_over_{x}': hl.agg.filter(rmt.GQ > x, hl.agg.sum(1 + rmt.END - rmt.locus.position))
        for x in gq_bins
    })).cols()

    joined = ref_results[variant_results.key].gq_exprs
    joined_results = variant_results.transmute(**{
        f'gq_over_{x}': variant_results.gq_exprs[f'gq_over_{x}'] + joined[f'gq_over_{x}']
        for x in gq_bins
    })
    return joined_results
