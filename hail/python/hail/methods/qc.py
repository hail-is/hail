import copy
import logging

import hail as hl
from collections import Counter
import os
from shlex import quote as shq

from typing import Any, Dict, Tuple, List, Optional, Union

from hailtop import pip_version
from hailtop.utils import Timings, async_to_blocking
import hailtop.batch_client as bc
from hailtop.config import configuration_of, get_deploy_config
from hailtop import yamlx

from hail.backend.service_backend import ServiceBackend
from hail.typecheck import typecheck, oneof, anytype, dictof, nullable, numeric, sequenceof
from hail.expr.expressions.expression_typecheck import expr_float64
from hail.utils import FatalError
from hail.utils.java import Env, info, warning
from hail.utils.misc import divide_null, guess_cloud_spark_provider
from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.ir import TableToTableApply
from .misc import require_biallelic, require_row_key_variant, require_col_key_str, require_table_key_variant, require_alleles_field

log = logging.getLogger('methods.qc')


HAIL_GENETICS_VEP_GRCH37_85_IMAGE = os.environ.get('HAIL_GENETICS_VEP_GRCH37_85_IMAGE', f'hailgenetics/vep/grch37/85/vep:{pip_version()}')
HAIL_GENETICS_VEP_GRCH38_95_IMAGE = os.environ.get('HAIL_GENETICS_VEP_GRCH38_95_IMAGE', f'hailgenetics/vep/grch38/95/vep:{pip_version()}')


@typecheck(mt=MatrixTable, name=str)
def sample_qc(mt, name='sample_qc') -> MatrixTable:
    """Compute per-sample metrics useful for quality control.

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------

    Compute sample QC metrics and remove low-quality samples:

    >>> dataset = hl.sample_qc(dataset, name='sample_qc')
    >>> filtered_dataset = dataset.filter_cols((dataset.sample_qc.dp_stats.mean > 20) & (dataset.sample_qc.r_ti_tv > 1.5))

    Notes
    -----

    This method computes summary statistics per sample from a genetic matrix and stores
    the results as a new column-indexed struct field in the matrix, named based on the
    `name` parameter.

    If `mt` contains an entry field `DP` of type :py:data:`.tint32`, then the
    field `dp_stats` is computed. If `mt` contains an entry field `GQ` of type
    :py:data:`.tint32`, then the field `gq_stats` is computed. Both `dp_stats`
    and `gq_stats` are structs with with four fields:

    - `mean` (``float64``) -- Mean value.
    - `stdev` (``float64``) -- Standard deviation (zero degrees of freedom).
    - `min` (``int32``) -- Minimum value.
    - `max` (``int32``) -- Maximum value.

    If the dataset does not contain an entry field `GT` of type
    :py:data:`.tcall`, then an error is raised. The following fields are always
    computed from `GT`:

    - `call_rate` (``float64``) -- Fraction of calls not missing or filtered.
      Equivalent to `n_called` divided by :meth:`.count_rows`.
    - `n_called` (``int64``) -- Number of non-missing calls.
    - `n_not_called` (``int64``) -- Number of missing calls.
    - `n_filtered` (``int64``) -- Number of filtered entries.
    - `n_hom_ref` (``int64``) -- Number of homozygous reference calls.
    - `n_het` (``int64``) -- Number of heterozygous calls.
    - `n_hom_var` (``int64``) -- Number of homozygous alternate calls.
    - `n_non_ref` (``int64``) -- Sum of `n_het` and `n_hom_var`.
    - `n_snp` (``int64``) -- Number of SNP alternate alleles.
    - `n_insertion` (``int64``) -- Number of insertion alternate alleles.
    - `n_deletion` (``int64``) -- Number of deletion alternate alleles.
    - `n_singleton` (``int64``) -- Number of private alleles. Reference alleles are never counted as singletons, even if
      every other allele at a site is non-reference.
    - `n_transition` (``int64``) -- Number of transition (A-G, C-T) alternate alleles.
    - `n_transversion` (``int64``) -- Number of transversion alternate alleles.
    - `n_star` (``int64``) -- Number of star (upstream deletion) alleles.
    - `r_ti_tv` (``float64``) -- Transition/Transversion ratio.
    - `r_het_hom_var` (``float64``) -- Het/HomVar call ratio.
    - `r_insertion_deletion` (``float64``) -- Insertion/Deletion allele ratio.

    Missing values ``NA`` may result from division by zero.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        Dataset.
    name : :class:`str`
        Name for resulting field.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with a new column-indexed field `name`.
    """

    require_row_key_variant(mt, 'sample_qc')

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
    mt = mt.annotate_rows(**{variant_ac: hl.agg.call_stats(mt.GT, mt.alleles).AC,
                             variant_atypes: mt.alleles[1:].map(lambda alt: allele_type(mt.alleles[0], alt))})

    bound_exprs = {}
    gq_dp_exprs = {}

    def has_field_of_type(name, dtype):
        return name in mt.entry and mt[name].dtype == dtype

    if has_field_of_type('DP', hl.tint32):
        gq_dp_exprs['dp_stats'] = hl.agg.stats(mt.DP).select('mean', 'stdev', 'min', 'max')

    if has_field_of_type('GQ', hl.tint32):
        gq_dp_exprs['gq_stats'] = hl.agg.stats(mt.GQ).select('mean', 'stdev', 'min', 'max')

    if not has_field_of_type('GT', hl.tcall):
        raise ValueError("'sample_qc': expect an entry field 'GT' of type 'call'")

    bound_exprs['n_called'] = hl.agg.count_where(hl.is_defined(mt['GT']))
    bound_exprs['n_not_called'] = hl.agg.count_where(hl.is_missing(mt['GT']))

    n_rows_ref = hl.expr.construct_expr(hl.ir.Ref('n_rows', hl.tint64), hl.tint64, mt._col_indices,
                                        hl.utils.LinkedList(hl.expr.expressions.Aggregation))
    bound_exprs['n_filtered'] = n_rows_ref - hl.agg.count()
    bound_exprs['n_hom_ref'] = hl.agg.count_where(mt['GT'].is_hom_ref())
    bound_exprs['n_het'] = hl.agg.count_where(mt['GT'].is_het())
    bound_exprs['n_singleton'] = hl.agg.sum(hl.rbind(mt['GT'], lambda gt: hl.sum(
        hl.range(0, gt.ploidy).map(lambda i: hl.rbind(gt[i], lambda gti: (gti != 0) & (mt[variant_ac][gti] == 1))))))

    bound_exprs['allele_type_counts'] = hl.agg.explode(
        lambda allele_type: hl.tuple(
            hl.agg.count_where(allele_type == i) for i in range(len(allele_ints))
        ),
        (hl.range(0, mt['GT'].ploidy)
         .map(lambda i: mt['GT'][i])
         .filter(lambda allele_idx: allele_idx > 0)
         .map(lambda allele_idx: mt[variant_atypes][allele_idx - 1]))
    )

    result_struct = hl.rbind(
        hl.struct(**bound_exprs),
        lambda x: hl.rbind(
            hl.struct(**{
                **gq_dp_exprs,
                'call_rate': hl.float64(x.n_called) / (x.n_called + x.n_not_called + x.n_filtered),
                'n_called': x.n_called,
                'n_not_called': x.n_not_called,
                'n_filtered': x.n_filtered,
                'n_hom_ref': x.n_hom_ref,
                'n_het': x.n_het,
                'n_hom_var': x.n_called - x.n_hom_ref - x.n_het,
                'n_non_ref': x.n_called - x.n_hom_ref,
                'n_singleton': x.n_singleton,
                'n_snp': (x.allele_type_counts[allele_ints["Transition"]]
                          + x.allele_type_counts[allele_ints["Transversion"]]),
                'n_insertion': x.allele_type_counts[allele_ints["Insertion"]],
                'n_deletion': x.allele_type_counts[allele_ints["Deletion"]],
                'n_transition': x.allele_type_counts[allele_ints["Transition"]],
                'n_transversion': x.allele_type_counts[allele_ints["Transversion"]],
                'n_star': x.allele_type_counts[allele_ints["Star"]],
            }),
            lambda s: s.annotate(
                r_ti_tv=divide_null(hl.float64(s.n_transition), s.n_transversion),
                r_het_hom_var=divide_null(hl.float64(s.n_het), s.n_hom_var),
                r_insertion_deletion=divide_null(hl.float64(s.n_insertion), s.n_deletion)
            )))

    mt = mt.annotate_cols(**{name: result_struct})
    mt = mt.drop(variant_ac, variant_atypes)

    return mt


@typecheck(mt=MatrixTable, name=str)
def variant_qc(mt, name='variant_qc') -> MatrixTable:
    """Compute common variant statistics (quality control metrics).

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------

    >>> dataset_result = hl.variant_qc(dataset)

    Notes
    -----
    This method computes variant statistics from the genotype data, returning
    a new struct field `name` with the following metrics based on the fields
    present in the entry schema.

    If `mt` contains an entry field `DP` of type :py:data:`.tint32`, then the
    field `dp_stats` is computed. If `mt` contains an entry field `GQ` of type
    :py:data:`.tint32`, then the field `gq_stats` is computed. Both `dp_stats`
    and `gq_stats` are structs with with four fields:

    - `mean` (``float64``) -- Mean value.
    - `stdev` (``float64``) -- Standard deviation (zero degrees of freedom).
    - `min` (``int32``) -- Minimum value.
    - `max` (``int32``) -- Maximum value.

    If the dataset does not contain an entry field `GT` of type
    :py:data:`.tcall`, then an error is raised. The following fields are always
    computed from `GT`:

    - `AF` (``array<float64>``) -- Calculated allele frequency, one element
      per allele, including the reference. Sums to one. Equivalent to
      `AC` / `AN`.
    - `AC` (``array<int32>``) -- Calculated allele count, one element per
      allele, including the reference. Sums to `AN`.
    - `AN` (``int32``) -- Total number of called alleles.
    - `homozygote_count` (``array<int32>``) -- Number of homozygotes per
      allele. One element per allele, including the reference.
    - `call_rate` (``float64``) -- Fraction of calls neither missing nor filtered.
      Equivalent to `n_called` / :meth:`.count_cols`.
    - `n_called` (``int64``) -- Number of samples with a defined `GT`.
    - `n_not_called` (``int64``) -- Number of samples with a missing `GT`.
    - `n_filtered` (``int64``) -- Number of filtered entries.
    - `n_het` (``int64``) -- Number of heterozygous samples.
    - `n_non_ref` (``int64``) -- Number of samples with at least one called
      non-reference allele.
    - `het_freq_hwe` (``float64``) -- Expected frequency of heterozygous
      samples under Hardy-Weinberg equilibrium. See
      :func:`.functions.hardy_weinberg_test` for details.
    - `p_value_hwe` (``float64``) -- p-value from two-sided test of Hardy-Weinberg
      equilibrium. See :func:`.functions.hardy_weinberg_test` for details.
    - `p_value_excess_het` (``float64``) -- p-value from one-sided test of
      Hardy-Weinberg equilibrium for excess heterozygosity.
      See :func:`.functions.hardy_weinberg_test` for details.

    Warning
    -------
    `het_freq_hwe` and `p_value_hwe` are calculated as in
    :func:`.functions.hardy_weinberg_test`, with non-diploid calls
    (``ploidy != 2``) ignored in the counts. As this test is only
    statistically rigorous in the biallelic setting, :func:`.variant_qc`
    sets both fields to missing for multiallelic variants. Consider using
    :func:`~hail.methods.split_multi` to split multi-allelic variants beforehand.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        Dataset.
    name : :class:`str`
        Name for resulting field.

    Returns
    -------
    :class:`.MatrixTable`
    """
    require_alleles_field(mt, 'variant_qc')

    bound_exprs = {}
    gq_dp_exprs = {}

    def has_field_of_type(name, dtype):
        return name in mt.entry and mt[name].dtype == dtype

    if has_field_of_type('DP', hl.tint32):
        gq_dp_exprs['dp_stats'] = hl.agg.stats(mt.DP).select('mean', 'stdev', 'min', 'max')

    if has_field_of_type('GQ', hl.tint32):
        gq_dp_exprs['gq_stats'] = hl.agg.stats(mt.GQ).select('mean', 'stdev', 'min', 'max')

    if not has_field_of_type('GT', hl.tcall):
        raise ValueError("'variant_qc': expect an entry field 'GT' of type 'call'")

    bound_exprs['n_called'] = hl.agg.count_where(hl.is_defined(mt['GT']))
    bound_exprs['n_not_called'] = hl.agg.count_where(hl.is_missing(mt['GT']))
    n_cols_ref = hl.expr.construct_expr(hl.ir.Ref('n_cols', hl.tint32), hl.tint32,
                                        mt._row_indices, hl.utils.LinkedList(hl.expr.expressions.Aggregation))
    bound_exprs['n_filtered'] = hl.int64(n_cols_ref) - hl.agg.count()
    bound_exprs['call_stats'] = hl.agg.call_stats(mt.GT, mt.alleles)

    result = hl.rbind(hl.struct(**bound_exprs),
                      lambda e1: hl.rbind(
                          hl.case().when(
                              hl.len(mt.alleles) == 2,
                              (hl.hardy_weinberg_test(e1.call_stats.homozygote_count[0],
                                                      e1.call_stats.AC[1] - 2
                                                      * e1.call_stats.homozygote_count[1],
                                                      e1.call_stats.homozygote_count[1]),
                               hl.hardy_weinberg_test(e1.call_stats.homozygote_count[0],
                                                      e1.call_stats.AC[1] - 2
                                                      * e1.call_stats.homozygote_count[1],
                                                      e1.call_stats.homozygote_count[1],
                                                      one_sided=True))
                          ).or_missing(),
                          lambda hwe: hl.struct(**{
                              **gq_dp_exprs,
                              **e1.call_stats,
                              'call_rate': hl.float(e1.n_called) / (e1.n_called + e1.n_not_called + e1.n_filtered),
                              'n_called': e1.n_called,
                              'n_not_called': e1.n_not_called,
                              'n_filtered': e1.n_filtered,
                              'n_het': e1.n_called - hl.sum(e1.call_stats.homozygote_count),
                              'n_non_ref': e1.n_called - e1.call_stats.homozygote_count[0],
                              'het_freq_hwe': hwe[0].het_freq_hwe,
                              'p_value_hwe': hwe[0].p_value,
                              'p_value_excess_het': hwe[1].p_value})))

    return mt.annotate_rows(**{name: result})


@typecheck(left=MatrixTable,
           right=MatrixTable,
           _localize_global_statistics=bool)
def concordance(left, right, *, _localize_global_statistics=True) -> Tuple[List[List[int]], Table, Table]:
    """Calculate call concordance with another dataset.

    .. include:: ../_templates/req_tstring.rst

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    .. include:: ../_templates/req_unphased_diploid_gt.rst

    Examples
    --------

    Compute concordance between two datasets and output the global concordance
    statistics and two tables with concordance computed per column key and per
    row key:

    >>> global_conc, cols_conc, rows_conc = hl.concordance(dataset, dataset2)

    Notes
    -----

    This method computes the genotype call concordance (from the entry
    field **GT**) between two biallelic variant datasets.  It requires
    unique sample IDs and performs an inner join on samples (only
    samples in both datasets will be considered). In addition, all genotype
    calls must be **diploid** and **unphased**.

    It performs an ordered zip join of the variants.  That means the
    variants of each dataset are sorted, with duplicate variants
    appearing in some random relative order, and then zipped together.
    When a variant appears a different number of times between the two
    datasets, the dataset with the fewer number of instances is padded
    with "no data".  For example, if a variant is only in one dataset,
    then each genotype is treated as "no data" in the other.

    This method returns a tuple of three objects: a nested list of
    list of int with global concordance summary statistics, a table
    with concordance statistics per column key, and a table with
    concordance statistics per row key.

    **Using the global summary result**

    The global summary is a list of list of int (conceptually a 5 by 5 matrix),
    where the indices have special meaning:

    0. No Data (missing variant or filtered entry)
    1. No Call (missing genotype call)
    2. Hom Ref
    3. Heterozygous
    4. Hom Var

    The first index is the state in the left dataset and the second index is
    the state in the right dataset. Typical uses of the summary list are shown
    below.

    >>> summary, samples, variants = hl.concordance(dataset, dataset2)
    >>> left_homref_right_homvar = summary[2][4]
    >>> left_het_right_missing = summary[3][1]
    >>> left_het_right_something_else = sum(summary[3][:]) - summary[3][3]
    >>> total_concordant = summary[2][2] + summary[3][3] + summary[4][4]
    >>> total_discordant = sum([sum(s[2:]) for s in summary[2:]]) - total_concordant

    **Using the table results**

    Table 1: Concordance statistics by column

    This table contains the column key field of `left`, and the following fields:

        - `n_discordant` (:py:data:`.tint64`) -- Count of discordant calls (see below for
          full definition).
        - `concordance` (:class:`.tarray` of :class:`.tarray` of :py:data:`.tint64`) --
          Array of concordance per state on left and right, matching the structure of
          the global summary defined above.

    Table 2: Concordance statistics by row

    This table contains the row key fields of `left`, and the following fields:

        - `n_discordant` (:py:data:`.tfloat64`) -- Count of discordant calls (see below for
          full definition).
        - `concordance` (:class:`.tarray` of :class:`.tarray` of :py:data:`.tint64`) --
          Array of concordance per state on left and right, matching the structure of the
          global summary defined above.

    In these tables, the column **n_discordant** is provided as a convenience,
    because this is often one of the most useful concordance statistics. This
    value is the number of genotypes which were called (homozygous reference,
    heterozygous, or homozygous variant) in both datasets, but where the call
    did not match between the two.

    The column `concordance` matches the structure of the global summmary,
    which is detailed above. Once again, the first index into this array is the
    state on the left, and the second index is the state on the right. For
    example, ``concordance[1][4]`` is the number of "no call" genotypes on the
    left that were called homozygous variant on the right.

    Parameters
    ----------
    left : :class:`.MatrixTable`
        First dataset to compare.
    right : :class:`.MatrixTable`
        Second dataset to compare.

    Returns
    -------
    (list of list of int, :class:`.Table`, :class:`.Table`)
        The global concordance statistics, a table with concordance statistics
        per column key, and a table with concordance statistics per row key.

    """

    require_col_key_str(left, 'concordance, left')
    require_col_key_str(right, 'concordance, right')

    left_sample_counter = left.aggregate_cols(hl.agg.counter(left.col_key[0]))
    right_sample_counter = right.aggregate_cols(hl.agg.counter(right.col_key[0]))

    left_bad = [f'{k!r}: {v}' for k, v in left_sample_counter.items() if v > 1]
    right_bad = [f'{k!r}: {v}' for k, v in right_sample_counter.items() if v > 1]
    if left_bad or right_bad:
        raise ValueError(f"Found duplicate sample IDs:\n"
                         f"  left:  {', '.join(left_bad)}\n"
                         f"  right: {', '.join(right_bad)}")

    included = set(left_sample_counter.keys()).intersection(set(right_sample_counter.keys()))

    info(f"concordance: including {len(included)} shared samples "
         f"({len(left_sample_counter)} total on left, {len(right_sample_counter)} total on right)")

    left = require_biallelic(left, 'concordance, left')
    right = require_biallelic(right, 'concordance, right')

    lit = hl.literal(included, dtype=hl.tset(hl.tstr))
    left = left.filter_cols(lit.contains(left.col_key[0]))
    right = right.filter_cols(lit.contains(right.col_key[0]))

    left = left.select_entries('GT').select_rows().select_cols()
    right = right.select_entries('GT').select_rows().select_cols()

    joined = hl.experimental.full_outer_join_mt(left, right)

    def get_idx(struct):
        return hl.if_else(
            hl.is_missing(struct),
            0,
            hl.coalesce(2 + struct.GT.n_alt_alleles(), 1))

    aggr = hl.agg.counter(get_idx(joined.left_entry) + 5 * get_idx(joined.right_entry))

    def concordance_array(counter):
        return hl.range(0, 5).map(lambda i: hl.range(0, 5).map(lambda j: counter.get(i + 5 * j, 0)))

    discordant_indices = set()
    for i in range(5):
        for j in range(5):
            if i > 1 and j > 1 and i != j:
                discordant_indices.add(i + 5 * j)

    def n_discordant(counter):
        return hl.sum(
            hl.array(counter)
            .filter(lambda tup: hl.literal(discordant_indices).contains(tup[0]))
            .map(lambda tup: tup[1]))

    glob = joined.aggregate_entries(concordance_array(aggr), _localize=_localize_global_statistics)
    if _localize_global_statistics:
        total_conc = [x[1:] for x in glob[1:]]
        on_diag = sum(total_conc[i][i] for i in range(len(total_conc)))
        total_obs = sum(sum(x) for x in total_conc)
        pct = on_diag / total_obs * 100 if total_obs > 0 else float('nan')
        info(f"concordance: total concordance {pct:.2f}%")

    per_variant = joined.annotate_rows(concordance=aggr)
    per_variant = per_variant.select_rows(concordance=concordance_array(per_variant.concordance),
                                          n_discordant=n_discordant(per_variant.concordance))
    per_sample = joined.annotate_cols(concordance=aggr)
    per_sample = per_sample.select_cols(concordance=concordance_array(per_sample.concordance),
                                        n_discordant=n_discordant(per_sample.concordance))

    return glob, per_sample.cols(), per_variant.rows()


class VEPConfig:
    default_vep_json_typ = hl.tstruct(
        assembly_name=hl.tstr,
        allele_string=hl.tstr,
        ancestral=hl.tstr,
        colocated_variants=hl.tarray(hl.tstruct(
            aa_allele=hl.tstr,
            aa_maf=hl.tfloat,
            afr_allele=hl.tstr,
            afr_maf=hl.tfloat,
            allele_string=hl.tstr,
            amr_allele=hl.tstr,
            amr_maf=hl.tfloat,
            clin_sig=hl.tarray(hl.tstr),
            end=hl.tint32,
            eas_allele=hl.tstr,
            eas_maf=hl.tfloat,
            ea_allele=hl.tstr,
            ea_maf=hl.tfloat,
            eur_allele=hl.tstr,
            eur_maf=hl.tfloat,
            exac_adj_allele=hl.tstr,
            exac_adj_maf=hl.tfloat,
            exac_allele=hl.tstr,
            exac_afr_allele=hl.tstr,
            exac_afr_maf=hl.tfloat,
            exac_amr_allele=hl.tstr,
            exac_amr_maf=hl.tfloat,
            exac_eas_allele=hl.tstr,
            exac_eas_maf=hl.tfloat,
            exac_fin_allele=hl.tstr,
            exac_fin_maf=hl.tfloat,
            exac_maf=hl.tfloat,
            exac_nfe_allele=hl.tstr,
            exac_nfe_maf=hl.tfloat,
            exac_oth_allele=hl.tstr,
            exac_oth_maf=hl.tfloat,
            exac_sas_allele=hl.tstr,
            exac_sas_maf=hl.tfloat,
            id=hl.tstr,
            minor_allele=hl.tstr,
            minor_allele_freq=hl.tfloat,
            phenotype_or_disease=hl.tint32,
            pubmed=hl.tarray(hl.tint32),
            sas_allele=hl.tstr,
            sas_maf=hl.tfloat,
            somatic=hl.tint32,
            start=hl.tint32,
            strand=hl.tint32)),
        context=hl.tstr,
        end=hl.tint32,
        id=hl.tstr,
        input=hl.tstr,
        intergenic_consequences=hl.tarray(hl.tstruct(allele_num=hl.tint32,
                                                     consequence_terms=hl.tarray(hl.tstr),
                                                     impact=hl.tstr,
                                                     minimised=hl.tint32,
                                                     variant_allele=hl.tstr)),
        most_severe_consequence=hl.tstr,
        motif_feature_consequences=hl.tarray(hl.tstruct(allele_num=hl.tint32,
                                                        consequence_terms=hl.tarray(hl.tstr),
                                                        high_inf_pos=hl.tstr,
                                                        impact=hl.tstr,
                                                        minimised=hl.tint32,
                                                        motif_feature_id=hl.tstr,
                                                        motif_name=hl.tstr,
                                                        motif_pos=hl.tint32,
                                                        motif_score_change=hl.tfloat,
                                                        strand=hl.tint32,
                                                        variant_allele=hl.tstr)),
        regulatory_feature_consequences=hl.tarray(hl.tstruct(allele_num=hl.tint32,
                                                             biotype=hl.tstr,
                                                             consequence_terms=hl.tarray(hl.tstr),
                                                             impact=hl.tstr,
                                                             minimised=hl.tint32,
                                                             regulatory_feature_id=hl.tstr,
                                                             variant_allele=hl.tstr)),
        seq_region_name=hl.tstr,
        start=hl.tint32,
        strand=hl.tint32,
        transcript_consequences=hl.tarray(hl.tstruct(allele_num=hl.tint32,
                                                     amino_acids=hl.tstr,
                                                     biotype=hl.tstr,
                                                     canonical=hl.tint32,
                                                     ccds=hl.tstr,
                                                     cdna_start=hl.tint32,
                                                     cdna_end=hl.tint32,
                                                     cds_end=hl.tint32,
                                                     cds_start=hl.tint32,
                                                     codons=hl.tstr,
                                                     consequence_terms=hl.tarray(hl.tstr),
                                                     distance=hl.tint32,
                                                     domains=hl.tarray(hl.tstruct(db=hl.tstr,
                                                                                  name=hl.tstr)),
                                                     exon=hl.tstr,
                                                     gene_id=hl.tstr,
                                                     gene_pheno=hl.tint32,
                                                     gene_symbol=hl.tstr,
                                                     gene_symbol_source=hl.tstr,
                                                     hgnc_id=hl.tstr,
                                                     hgvsc=hl.tstr,
                                                     hgvsp=hl.tstr,
                                                     hgvs_offset=hl.tint32,
                                                     impact=hl.tstr,
                                                     intron=hl.tstr,
                                                     lof=hl.tstr,
                                                     lof_flags=hl.tstr,
                                                     lof_filter=hl.tstr,
                                                     lof_info=hl.tstr,
                                                     minimised=hl.tint32,
                                                     polyphen_prediction=hl.tstr,
                                                     polyphen_score=hl.tfloat,
                                                     protein_end=hl.tint32,
                                                     protein_start=hl.tint32,
                                                     protein_id=hl.tstr,
                                                     sift_prediction=hl.tstr,
                                                     sift_score=hl.tfloat,
                                                     strand=hl.tint32,
                                                     swissprot=hl.tstr,
                                                     transcript_id=hl.tstr,
                                                     trembl=hl.tstr,
                                                     uniparc=hl.tstr,
                                                     variant_allele=hl.tstr)),
        variant_class=hl.tstr)

    @staticmethod
    def from_dict(config: Dict[str, Any]):
        return VEPConfig(
            config['data_bucket'],
            config['regions'],
            config['image'],
            config['data_mount'],
            config['env'],
            config['vep_json_typ'],
            config['command'],
            config['csq_header_command'],
            config['reference_data_is_bucket_requester_pays'],
            config.get('cloud'),
        )

    def __init__(self,
                 data_bucket: str,
                 regions: List[str],
                 image: str,
                 data_mount: str,
                 env: Dict[str, str],
                 vep_json_typ: hl.expr.HailType,
                 command: List[str],
                 csq_header_command: List[str],
                 reference_data_is_bucket_requester_pays: bool,
                 cloud: Optional[str]):
        self.data_bucket = data_bucket
        self.regions = regions
        self.image = image
        self.data_mount = data_mount
        self.env = env
        self.vep_json_typ = vep_json_typ
        self.command = command
        self.csq_header_command = csq_header_command
        self.reference_data_is_bucket_requester_pays = reference_data_is_bucket_requester_pays
        self.cloud = cloud


def vep_85_grch37_run_cmd(run_typ: str) -> str:
    return f'''
cat >> run_vep.sh <<EOF
#!/bin/bash

if [ $VEP_CONSEQUENCE -ne 0 ]
then
  vcf_or_json="--vcf"
else
  vcf_or_json="--json"
fi

export VEP_COMMAND=/vep/vep \
${{VEP_INPUT_FILE:+--input_file $VEP_INPUT_FILE}} \
--format vcf \
${{vcf_or_json}} \
--everything \
--allele_number \
--no_stats \
--cache \
--offline \
--minimal \
--assembly GRCh37 \
--dir=${{VEP_DATA_DIR}} \
--plugin LoF,human_ancestor_fa:${{VEP_DATA_DIR}}/loftee_data/human_ancestor.fa.gz,filter_position:0.05,min_intron_size:15,conservation_file:${{VEP_DATA_DIR}}/loftee_data/phylocsf_gerp.sql,gerp_file:${{VEP_DATA_DIR}}/loftee_data/GERP_scores.final.sorted.txt.gz \
-o STDOUT

exec /vep.py "$@"
EOF

sh run_vep.sh {run_typ}
'''


def vep_95_grch38_run_cmd(run_typ: str) -> str:
    return f'''
cat >> run_vep.sh <<EOF
#!/bin/bash

if [ $VEP_CONSEQUENCE -ne 0 ]
then
  vcf_or_json="--vcf"
else
  vcf_or_json="--json"
fi

export VEP_COMMAND=/vep/vep \
${{VEP_INPUT_FILE:+--input_file $VEP_INPUT_FILE}} \
--format vcf \
${{if ($VEP_CONSEQUENCE -ne 0);then --vcf;--json;fi}} \
--everything \
--allele_number \
--no_stats \
--cache \
--offline \
--minimal \
--assembly GRCh38 \
--fasta ${{VEP_DATA_MOUNT}}/homo_sapiens/95_GRCh38/Homo_sapiens.GRCh38.dna.toplevel.fa.gz \
--plugin "LoF,loftee_path:/vep/ensembl-vep/Plugins/,gerp_bigwig:${{VEP_DATA_MOUNT}}/gerp_conservation_scores.homo_sapiens.GRCh38.bw,human_ancestor_fa:${{VEP_DATA_MOUNT}}/human_ancestor.fa.gz,conservation_file:${{VEP_DATA_MOUNT}}/loftee.sql" \
--dir_plugins /vep/ensembl-vep/Plugins/ \
--dir_cache ${{VEP_DATA_MOUNT}} \
-o STDOUT

exec /vep.py "$@"
EOF

sh run_vep.sh {run_typ}
'''


supported_vep_configs = {
    ('GRCh37', 'gcp', 'us-central1', 'hail.is'): VEPConfig(
        'hail-qob-vep-grch37-us-central1',
        ['us-central1'],
        HAIL_GENETICS_VEP_GRCH37_85_IMAGE,
        '/vep_data/',
        {},
        VEPConfig.default_vep_json_typ,
        ["/bin/bash", "-c", vep_85_grch37_run_cmd('vep')],
        ["/bin/bash", "-c", vep_85_grch37_run_cmd('csq_header')],
        True,
        'gcp',
    ),
    ('GRCh38', 'gcp', 'us-central1', 'hail.is'): VEPConfig(
        'hail-qob-vep-grch38-us-central1',
        ['us-central1'],
        HAIL_GENETICS_VEP_GRCH38_95_IMAGE,
        '/vep_data/',
        {},
        VEPConfig.default_vep_json_typ._insert_field('transcript_consequences', hl.tarray(
            VEPConfig.default_vep_json_typ['transcript_consequences'].element_type._insert_fields(
                appris=hl.tstr,
                tsl=hl.tint32,
            )
        )),
        ["/bin/bash", "-c", vep_95_grch38_run_cmd('vep')],
        ["/bin/bash", "-c", vep_95_grch38_run_cmd('csq_header')],
        True,
        'gcp',
    ),
}


def _supported_vep_config(cloud: str, reference_genome: str, *, regions: List[str]) -> VEPConfig:
    domain = get_deploy_config()._domain

    for region in regions:
        config_params = (reference_genome, cloud, region, domain)
        if config_params in supported_vep_configs:
            return supported_vep_configs[config_params]

    raise ValueError(f'could not find a supported vep configuration for reference genome {reference_genome}, '
                     f'cloud {cloud}, regions {regions}, and domain {domain}')


def _service_vep(backend: ServiceBackend,
                 ht: Table,
                 config: Optional[Dict[str, Any]],
                 regions: Optional[List[str]],
                 block_size: int,
                 csq: bool,
                 tolerate_parse_error: bool) -> hl.Table:
    reference_genome = ht.locus.dtype.reference_genome.name
    cloud = backend.bc.cloud()

    if regions is None:
        regions = backend.regions

    if config is not None:
        vep_config = VEPConfig.from_dict(config)
    else:
        vep_config = _supported_vep_config(cloud, reference_genome, regions=regions)

    requester_pays_project = backend.flags.get('gcs_requester_pays_project')
    if requester_pays_project is None and vep_config.is_maintained_by_hail and vep_config.cloud == 'gcp':
        raise ValueError("No requester pays project has been set. "
                         "Use hl.init(gcs_requester_pays_configuration='MY_PROJECT') "
                         "to set the requester pays project to use.")

    vep_input_path = hl.TemporaryDirectory(prefix='qob/vep/inputs/')
    vep_output_path = hl.TemporaryDirectory(prefix='qob/vep/outputs/')

    def get_env(part_id: int, input_file: Optional[str], output_file: str):
        local_env = copy.deepcopy(vep_config.env)
        local_env.update({
            'VEP_BLOCK_SIZE': str(block_size),
            'VEP_DATA_MOUNT': shq(vep_config.data_mount),
            'VEP_CONSEQUENCE': str(int(csq)),
            'VEP_TOLERATE_PARSE_ERROR': str(int(tolerate_parse_error)),
            'VEP_PART_ID': str(part_id),
            'VEP_OUTPUT_FILE': output_file,
        })
        if input_file:
            local_env['VEP_INPUT_FILE'] = input_file
        return local_env

    if csq:
        vep_typ = hl.tarray(hl.tstr)
    else:
        vep_typ = vep_config.vep_json_typ

    def build_vep_batch(bb: bc.aioclient.BatchBuilder):
        if csq:
            local_output_file = '/io/output'
            bb.create_job(vep_config.image,
                          vep_config.csq_header_command,
                          attributes={'name': 'csq-header'},
                          resources={'cpu': '1', 'memory': 'standard'},
                          cloudfuse=[(vep_config.data_bucket, vep_config.data_mount, True)],
                          output_files=[(local_output_file, f'{vep_output_path.name}/csq-header')],
                          regions=vep_config.regions,
                          requester_pays_project=requester_pays_project,
                          env=get_env(-1, None, local_output_file),
                          )

        for f in hl.hadoop_ls(vep_input_path.name):
            path = f['path']
            part_name = os.path.basename(path)
            if not part_name.startswith('part-'):
                continue
            part_id = int(part_name.split('-')[1])

            local_input_file = '/io/input'
            local_output_file = '/io/output.gz'

            bb.create_job(vep_config.image,
                          vep_config.command,
                          attributes={'name': f'vep-{part_id}'},
                          resources={'cpu': '1', 'memory': 'standard'},
                          input_files=[(path, local_input_file)],
                          output_files=[(local_output_file, f'{vep_output_path.name}/annotations/{part_name}.tsv.gz')],
                          cloudfuse=[(vep_config.data_bucket, vep_config.data_mount, True)],
                          regions=vep_config.regions,
                          requester_pays_project=requester_pays_project,
                          env=get_env(part_id, local_input_file, local_output_file),
                          )

    hl.export_vcf(ht, vep_input_path.name, parallel='header_per_shard')

    name = 'vep(...)'

    starting_job_id = async_to_blocking(backend._batch.status())['n_jobs'] + 1

    bb = backend.bc.update_batch(backend._batch.id)
    build_vep_batch(bb)
    b = bb.submit(disable_progress_bar=True)

    try:
        status = b.wait(description=name,
                        disable_progress_bar=backend.disable_progress_bar,
                        progress=None,
                        starting_job=starting_job_id)
    except BaseException:
        b.cancel()
        raise

    if status['n_succeeded'] != status['n_jobs']:
        failing_job = [job for job in b.jobs('!success')][0]
        failing_job = b.get_job(failing_job['job_id'])
        message = {
            'batch_status': status,
            'job_status': failing_job.status(),
            'log': failing_job.log()
        }
        raise FatalError(yamlx.dump(message))

    annotations = hl.import_table(f'{vep_output_path.name}/annotations/*',
                                  types={'variant': hl.tstr,
                                         'vep': vep_typ,
                                         'part_id': hl.tint,
                                         'block_id': hl.tint},
                                  force=True)

    annotations = annotations.annotate(vep_proc_id=hl.struct(
        part_id=annotations.part_id,
        block_id=annotations.block_id
    ))
    annotations = annotations.drop('part_id', 'block_id')
    annotations = annotations.key_by(**hl.parse_variant(annotations.variant, reference_genome=reference_genome))
    annotations = annotations.drop('variant')

    if csq:
        with hl.hadoop_open(f'{vep_output_path.name}/csq-header') as f:
            vep_csq_header = f.read().rstrip()
        annotations = annotations.annotate_globals(vep_csq_header=vep_csq_header)

    return annotations


@typecheck(dataset=oneof(Table, MatrixTable),
           config=nullable(oneof(str, dictof(str, anytype))),
           block_size=int,
           name=str,
           csq=bool,
           tolerate_parse_error=bool,
           regions=nullable(sequenceof(str)))
def vep(dataset: Union[Table, MatrixTable],
        config: Optional[Union[str, Dict[str, Any]]] = None,
        block_size: int = 1000,
        name: str = 'vep',
        csq: bool = False,
        tolerate_parse_error: bool = False,
        regions: Optional[List[str]] = None):
    """Annotate variants with VEP.

    .. include:: ../_templates/req_tvariant.rst

    :func:`.vep` runs `Variant Effect Predictor
    <http://www.ensembl.org/info/docs/tools/vep/index.html>`__ on the
    current dataset and adds the result as a row field.

    Examples
    --------

    Add VEP annotations to the dataset:

    >>> result = hl.vep(dataset, "data/vep-configuration.json") # doctest: +SKIP

    Notes
    -----

    **Installation**

    This VEP command only works if you have already installed VEP on your
    computing environment. If you use `hailctl dataproc` to start Hail clusters,
    installing VEP is achieved by specifying the `--vep` flag. For more detailed instructions,
    see :ref:`vep_dataproc`. If you use `hailctl hdinsight`, see :ref:`vep_hdinsight`.

    **Spark Configuration**

    :func:`.vep` needs a configuration file to tell it how to run VEP. This is the ``config`` argument
    to the VEP function. If you are using `hailctl dataproc` as mentioned above, you can just use the
    default argument for ``config`` and everything will work. If you need to run VEP with Hail in other environments,
    there are detailed instructions below.

    The format of the configuration file is JSON, and :func:`.vep`
    expects a JSON object with three fields:

    - `command` (array of string) -- The VEP command line to run.  The string literal `__OUTPUT_FORMAT_FLAG__` is replaced with `--json` or `--vcf` depending on `csq`.
    - `env` (object) -- A map of environment variables to values to add to the environment when invoking the command.  The value of each object member must be a string.
    - `vep_json_schema` (string): The type of the VEP JSON schema (as produced by the VEP when invoked with the `--json` option).  Note: This is the old-style 'parseable' Hail type syntax.  This will change.

    Here is an example configuration file for invoking VEP release 85
    installed in `/vep` with the Loftee plugin:

    .. code-block:: text

        {
            "command": [
                "/vep",
                "--format", "vcf",
                "__OUTPUT_FORMAT_FLAG__",
                "--everything",
                "--allele_number",
                "--no_stats",
                "--cache", "--offline",
                "--minimal",
                "--assembly", "GRCh37",
                "--plugin", "LoF,human_ancestor_fa:/root/.vep/loftee_data/human_ancestor.fa.gz,filter_position:0.05,min_intron_size:15,conservation_file:/root/.vep/loftee_data/phylocsf_gerp.sql,gerp_file:/root/.vep/loftee_data/GERP_scores.final.sorted.txt.gz",
                "-o", "STDOUT"
            ],
            "env": {
                "PERL5LIB": "/vep_data/loftee"
            },
            "vep_json_schema": "Struct{assembly_name:String,allele_string:String,ancestral:String,colocated_variants:Array[Struct{aa_allele:String,aa_maf:Float64,afr_allele:String,afr_maf:Float64,allele_string:String,amr_allele:String,amr_maf:Float64,clin_sig:Array[String],end:Int32,eas_allele:String,eas_maf:Float64,ea_allele:String,ea_maf:Float64,eur_allele:String,eur_maf:Float64,exac_adj_allele:String,exac_adj_maf:Float64,exac_allele:String,exac_afr_allele:String,exac_afr_maf:Float64,exac_amr_allele:String,exac_amr_maf:Float64,exac_eas_allele:String,exac_eas_maf:Float64,exac_fin_allele:String,exac_fin_maf:Float64,exac_maf:Float64,exac_nfe_allele:String,exac_nfe_maf:Float64,exac_oth_allele:String,exac_oth_maf:Float64,exac_sas_allele:String,exac_sas_maf:Float64,id:String,minor_allele:String,minor_allele_freq:Float64,phenotype_or_disease:Int32,pubmed:Array[Int32],sas_allele:String,sas_maf:Float64,somatic:Int32,start:Int32,strand:Int32}],context:String,end:Int32,id:String,input:String,intergenic_consequences:Array[Struct{allele_num:Int32,consequence_terms:Array[String],impact:String,minimised:Int32,variant_allele:String}],most_severe_consequence:String,motif_feature_consequences:Array[Struct{allele_num:Int32,consequence_terms:Array[String],high_inf_pos:String,impact:String,minimised:Int32,motif_feature_id:String,motif_name:String,motif_pos:Int32,motif_score_change:Float64,strand:Int32,variant_allele:String}],regulatory_feature_consequences:Array[Struct{allele_num:Int32,biotype:String,consequence_terms:Array[String],impact:String,minimised:Int32,regulatory_feature_id:String,variant_allele:String}],seq_region_name:String,start:Int32,strand:Int32,transcript_consequences:Array[Struct{allele_num:Int32,amino_acids:String,biotype:String,canonical:Int32,ccds:String,cdna_start:Int32,cdna_end:Int32,cds_end:Int32,cds_start:Int32,codons:String,consequence_terms:Array[String],distance:Int32,domains:Array[Struct{db:String,name:String}],exon:String,gene_id:String,gene_pheno:Int32,gene_symbol:String,gene_symbol_source:String,hgnc_id:String,hgvsc:String,hgvsp:String,hgvs_offset:Int32,impact:String,intron:String,lof:String,lof_flags:String,lof_filter:String,lof_info:String,minimised:Int32,polyphen_prediction:String,polyphen_score:Float64,protein_end:Int32,protein_start:Int32,protein_id:String,sift_prediction:String,sift_score:Float64,strand:Int32,swissprot:String,transcript_id:String,trembl:String,uniparc:String,variant_allele:String}],variant_class:String}"
        }

    The configuration files used by``hailctl dataproc`` can be found at the following locations:

     - ``GRCh37``: ``gs://hail-us-vep/vep85-loftee-gcloud.json``
     - ``GRCh38``: ``gs://hail-us-vep/vep95-GRCh38-loftee-gcloud.json``

    If no config file is specified, this function will check to see if environment variable `VEP_CONFIG_URI` is set with a path to a config file.

    **Batch Service Configuration**

    The config argument is a dictionary with the following expected fields:

     - `command` (:obj:`.list` of :obj:`.str`) -- The command line to run for a VEP job for a partition.
     - `csq_header_command` (:obj:`.list` of :obj:`.str`) -- The command line to run when generating the consequence header.
     - `env` (dict of :obj:`.str` to :obj:`.str`) -- A map of environment variables to values to add to the environment when invoking the command.
     - `vep_json_schema` (:class:`.HailType`): The type of the VEP JSON schema (as produced by VEP when invoked with the `--json` option).
     - `image` (:obj:`.str`) -- The docker image to run VEP.
     - `data_bucket` (:obj:`.str`) -- The location where the VEP data is stored.
     - `data_mount` (:obj:`.str`) -- The location in the container where the data should be mounted.
     - `reference_data_is_bucket_requester_pays` (:obj:`.bool`) -- True if the reference data is in a requester pays bucket.

    If no config is specified, Hail will use the user's Service configuration parameters to find a supported VEP configuration.

    The following environment variables are added to the job's environment based on the arguments to this function:

     - `VEP_BLOCK_SIZE` - The maximum number of variants provided as input to each invocation of VEP.
     - `VEP_PART_ID` - Partition ID.
     - `VEP_DATA_MOUNT` - Location where the vep data is mounted (same as `data_mount` in the config).
     - `VEP_CONSEQUENCE` - Integer equal to 0 or 1 on whether `csq` is False or True.
     - `VEP_TOLERATE_PARSE_ERROR` - Integer equal to 0 or 1 on whether `tolerate_parse_error` is False or True.
     - `VEP_OUTPUT_FILE` - String specifying the local path where the output TSV file with the VEP result should be located.
     - `VEP_INPUT_FILE` - String specifying the local path where the input VCF shard is located for all jobs.

    The `VEP_INPUT_FILE` environment variable is not available for the single job that computes the consequence header when
    ``csq=True``.

    **Annotations**

    A new row field is added in the location specified by `name` with type given
    by the type given by the `json_vep_schema` (if `csq` is ``False``) or
    :class:`.tarray` of :py:data:`.tstr` (if `csq` is ``True``).

    If csq is ``True``, then the CSQ header string is also added as a global
    field with name ``name + '_csq_header'``.

    Parameters
    ----------
    dataset : :class:`.MatrixTable` or :class:`.Table`
        Dataset.
    config : :class:`str` or ::obj:`dict` of :class:`str` to :class`str`, optional
        Path to VEP configuration file or a dictionary of configuration parameters.
    block_size : :obj:`int`
        Number of rows to process per VEP invocation.
    name : :class:`str`
        Name for resulting row field.
    csq : :obj:`bool`
        If ``True``, annotates with the VCF CSQ field as a :py:data:`.tstr`.
        If ``False``, annotates as the `vep_json_schema`.
    tolerate_parse_error : :obj:`bool`
        If ``True``, ignore invalid JSON produced by VEP and return a missing annotation.
    regions: :obj:`list` of :class:`str`, optional
        The list of regions to run jobs in when using the Service Backend. In most use cases, this
        should be a list containing a single region where the input data resides.

    Returns
    -------
    :class:`.MatrixTable` or :class:`.Table`
        Dataset with new row-indexed field `name` containing VEP annotations.

    """

    if isinstance(dataset, MatrixTable):
        require_row_key_variant(dataset, 'vep')
        ht = dataset.select_rows().rows()
    else:
        require_table_key_variant(dataset, 'vep')
        ht = dataset.select()

    ht = ht.distinct()

    backend = hl.current_backend()
    if isinstance(backend, ServiceBackend):
        annotations = _service_vep(backend, ht, config, regions, block_size, csq, tolerate_parse_error)
    else:
        if config is None:
            maybe_cloud_spark_provider = guess_cloud_spark_provider()
            maybe_config = os.getenv("VEP_CONFIG_URI")
            if maybe_config is not None:
                config = maybe_config
            elif maybe_cloud_spark_provider == 'hdinsight':
                warning(
                    'Assuming you are in a hailctl hdinsight cluster. If not, specify the config parameter to `hl.vep`.')
                config = 'file:/vep_data/vep-azure.json'
            else:
                raise ValueError("No config set and VEP_CONFIG_URI was not set.")

        annotations = Table(TableToTableApply(ht._tir,
                                              {'name': 'VEP',
                                               'config': config,
                                               'csq': csq,
                                               'blockSize': block_size,
                                               'tolerateParseError': tolerate_parse_error})).persist()

    if csq:
        dataset = dataset.annotate_globals(
            **{name + '_csq_header': annotations.index_globals()['vep_csq_header']})

    if isinstance(dataset, MatrixTable):
        vep = annotations[dataset.row_key]
        return dataset.annotate_rows(**{name: vep.vep, name + '_proc_id': vep.vep_proc_id})
    else:
        vep = annotations[dataset.key]
        return dataset.annotate(**{name: vep.vep, name + '_proc_id': vep.vep_proc_id})


@typecheck(dataset=oneof(Table, MatrixTable),
           config=str,
           block_size=int,
           name=str)
def nirvana(dataset: Union[MatrixTable, Table], config, block_size=500000, name='nirvana'):
    """Annotate variants using `Nirvana <https://github.com/Illumina/Nirvana>`_.

    .. include:: ../_templates/experimental.rst

    .. include:: ../_templates/req_tvariant.rst

    :func:`.nirvana` runs `Nirvana
    <https://github.com/Illumina/Nirvana>`_ on the current dataset and adds a
    new row field in the location specified by `name`.

    Examples
    --------

    Add Nirvana annotations to the dataset:

    >>> result = hl.nirvana(dataset, "data/nirvana.properties") # doctest: +SKIP

    **Configuration**

    :func:`.nirvana` requires a configuration file. The format is a
    `.properties file <https://en.wikipedia.org/wiki/.properties>`__, where each
    line defines a property as a key-value pair of the form ``key = value``.
    :func:`.nirvana` supports the following properties:

    - **hail.nirvana.dotnet** -- Location of dotnet. Optional, default: dotnet.
    - **hail.nirvana.path** -- Value of the PATH environment variable when
      invoking Nirvana. Optional, by default PATH is not set.
    - **hail.nirvana.location** -- Location of Nirvana.dll. Required.
    - **hail.nirvana.reference** -- Location of reference genome. Required.
    - **hail.nirvana.cache** -- Location of cache. Required.
    - **hail.nirvana.supplementaryAnnotationDirectory** -- Location of
      Supplementary Database. Optional, no supplementary database by default.

    Here is an example ``nirvana.properties`` configuration file:

    .. code-block:: text

        hail.nirvana.location = /path/to/dotnet/netcoreapp2.0/Nirvana.dll
        hail.nirvana.reference = /path/to/nirvana/References/Homo_sapiens.GRCh37.Nirvana.dat
        hail.nirvana.cache = /path/to/nirvana/Cache/GRCh37/Ensembl
        hail.nirvana.supplementaryAnnotationDirectory = /path/to/nirvana/SupplementaryDatabase/GRCh37

    **Annotations**

    A new row field is added in the location specified by `name` with the
    following schema:

    .. code-block:: text

        struct {
            chromosome: str,
            refAllele: str,
            position: int32,
            altAlleles: array<str>,
            cytogeneticBand: str,
            quality: float64,
            filters: array<str>,
            jointSomaticNormalQuality: int32,
            copyNumber: int32,
            strandBias: float64,
            recalibratedQuality: float64,
            variants: array<struct {
                altAllele: str,
                refAllele: str,
                chromosome: str,
                begin: int32,
                end: int32,
                phylopScore: float64,
                isReferenceMinor: bool,
                variantType: str,
                vid: str,
                hgvsg: str,
                isRecomposedVariant: bool,
                isDecomposedVariant: bool,
                regulatoryRegions: array<struct {
                    id: str,
                    type: str,
                    consequence: set<str>
                }>,
                clinvar: array<struct {
                    id: str,
                    reviewStatus: str,
                    isAlleleSpecific: bool,
                    alleleOrigins: array<str>,
                    refAllele: str,
                    altAllele: str,
                    phenotypes: array<str>,
                    medGenIds: array<str>,
                    omimIds: array<str>,
                    orphanetIds: array<str>,
                    significance: str,
                    lastUpdatedDate: str,
                    pubMedIds: array<str>
                }>,
                cosmic: array<struct {
                    id: str,
                    isAlleleSpecific: bool,
                    refAllele: str,
                    altAllele: str,
                    gene: str,
                    sampleCount: int32,
                    studies: array<struct {
                        id: int32,
                        histology: str,
                        primarySite: str
                    }>
                }>,
                dbsnp: struct {
                    ids: array<str>
                },
                globalAllele: struct {
                    globalMinorAllele: str,
                    globalMinorAlleleFrequency: float64
                },
                gnomad: struct {
                    coverage: str,
                    allAf: float64,
                    allAc: int32,
                    allAn: int32,
                    allHc: int32,
                    afrAf: float64,
                    afrAc: int32,
                    afrAn: int32,
                    afrHc: int32,
                    amrAf: float64,
                    amrAc: int32,
                    amrAn: int32,
                    amrHc: int32,
                    easAf: float64,
                    easAc: int32,
                    easAn: int32,
                    easHc: int32,
                    finAf: float64,
                    finAc: int32,
                    finAn: int32,
                    finHc: int32,
                    nfeAf: float64,
                    nfeAc: int32,
                    nfeAn: int32,
                    nfeHc: int32,
                    othAf: float64,
                    othAc: int32,
                    othAn: int32,
                    othHc: int32,
                    asjAf: float64,
                    asjAc: int32,
                    asjAn: int32,
                    asjHc: int32,
                    failedFilter: bool
                },
                gnomadExome: struct {
                    coverage: str,
                    allAf: float64,
                    allAc: int32,
                    allAn: int32,
                    allHc: int32,
                    afrAf: float64,
                    afrAc: int32,
                    afrAn: int32,
                    afrHc: int32,
                    amrAf: float64,
                    amrAc: int32,
                    amrAn: int32,
                    amrHc: int32,
                    easAf: float64,
                    easAc: int32,
                    easAn: int32,
                    easHc: int32,
                    finAf: float64,
                    finAc: int32,
                    finAn: int32,
                    finHc: int32,
                    nfeAf: float64,
                    nfeAc: int32,
                    nfeAn: int32,
                    nfeHc: int32,
                    othAf: float64,
                    othAc: int32,
                    othAn: int32,
                    othHc: int32,
                    asjAf: float64,
                    asjAc: int32,
                    asjAn: int32,
                    asjHc: int32,
                    sasAf: float64,
                    sasAc: int32,
                    sasAn: int32,
                    sasHc: int32,
                    failedFilter: bool
                },
                topmed: struct {
                    failedFilter: bool,
                    allAc: int32,
                    allAn: int32,
                    allAf: float64,
                    allHc: int32
                },
                oneKg: struct {
                    ancestralAllele: str,
                    allAf: float64,
                    allAc: int32,
                    allAn: int32,
                    afrAf: float64,
                    afrAc: int32,
                    afrAn: int32,
                    amrAf: float64,
                    amrAc: int32,
                    amrAn: int32,
                    easAf: float64,
                    easAc: int32,
                    easAn: int32,
                    eurAf: float64,
                    eurAc: int32,
                    eurAn: int32,
                    sasAf: float64,
                    sasAc: int32,
                    sasAn: int32
                },
                mitomap: array<struct {
                    refAllele: str,
                    altAllele: str,
                    diseases : array<str>,
                    hasHomoplasmy: bool,
                    hasHeteroplasmy: bool,
                    status: str,
                    clinicalSignificance: str,
                    scorePercentile: float64,
                    isAlleleSpecific: bool,
                    chromosome: str,
                    begin: int32,
                    end: int32,
                    variantType: str
                }
                transcripts: struct {
                    refSeq: array<struct {
                        transcript: str,
                        bioType: str,
                        aminoAcids: str,
                        cdnaPos: str,
                        codons: str,
                        cdsPos: str,
                        exons: str,
                        introns: str,
                        geneId: str,
                        hgnc: str,
                        consequence: array<str>,
                        hgvsc: str,
                        hgvsp: str,
                        isCanonical: bool,
                        polyPhenScore: float64,
                        polyPhenPrediction: str,
                        proteinId: str,
                        proteinPos: str,
                        siftScore: float64,
                        siftPrediction: str
                    }>,
                    ensembl: array<struct {
                        transcript: str,
                        bioType: str,
                        aminoAcids: str,
                        cdnaPos: str,
                        codons: str,
                        cdsPos: str,
                        exons: str,
                        introns: str,
                        geneId: str,
                        hgnc: str,
                        consequence: array<str>,
                        hgvsc: str,
                        hgvsp: str,
                        isCanonical: bool,
                        polyPhenScore: float64,
                        polyPhenPrediction: str,
                        proteinId: str,
                        proteinPos: str,
                        siftScore: float64,
                        siftPrediction: str
                    }>
                },
                overlappingGenes: array<str>
            }>
            genes: array<struct {
                name: str,
                omim: array<struct {
                    mimNumber: int32,
                    hgnc: str,
                    description: str,
                    phenotypes: array<struct {
                        mimNumber: int32,
                        phenotype: str,
                        mapping: str,
                        inheritance: array<str>,
                        comments: str
                    }>
                }>
                exac: struct {
                    pLi: float64,
                    pRec: float64,
                    pNull: float64
                }
            }>
        }

    Parameters
    ----------
    dataset : :class:`.MatrixTable` or :class:`.Table`
        Dataset.
    config : :class:`str`
        Path to Nirvana configuration file.
    block_size : :obj:`int`
        Number of rows to process per Nirvana invocation.
    name : :class:`str`
        Name for resulting row field.

    Returns
    -------
    :class:`.MatrixTable` or :class:`.Table`
        Dataset with new row-indexed field `name` containing Nirvana annotations.
    """
    if isinstance(dataset, MatrixTable):
        require_row_key_variant(dataset, 'nirvana')
        ht = dataset.select_rows().rows()
    else:
        require_table_key_variant(dataset, 'nirvana')
        ht = dataset.select()

    annotations = Table(TableToTableApply(ht._tir,
                                          {'name': 'Nirvana',
                                           'config': config,
                                           'blockSize': block_size}
                                          )).persist()

    if isinstance(dataset, MatrixTable):
        return dataset.annotate_rows(**{name: annotations[dataset.row_key].nirvana})
    else:
        return dataset.annotate(**{name: annotations[dataset.key].nirvana})


class _VariantSummary(object):
    def __init__(self, rg, n_variants, alleles_per_variant, variants_per_contig, allele_types, nti, ntv):
        self.rg = rg
        self.n_variants = n_variants
        self.alleles_per_variant = alleles_per_variant
        self.variants_per_contig = variants_per_contig
        self.allele_types = allele_types
        self.nti = nti
        self.ntv = ntv

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        return self._html_string()

    def __str__(self):
        contig_idx = {contig: i for i, contig in enumerate(self.rg.contigs)}
        max_contig_len = max(len(contig) for contig in self.variants_per_contig)
        contig_formatter = f'%{max_contig_len}s'

        max_allele_count_len = max(len(str(x)) for x in self.alleles_per_variant)
        allele_count_formatter = f'%{max_allele_count_len}s'

        max_allele_type_len = max(len(x) for x in self.allele_types)
        allele_type_formatter = f'%{max_allele_type_len}s'

        line_break = '=============================='

        builder = []
        builder.append(line_break)
        builder.append(f'Number of variants: {self.n_variants}')
        builder.append(line_break)
        builder.append('Alleles per variant')
        builder.append('-------------------')
        for n_alleles, count in sorted(self.alleles_per_variant.items(), key=lambda x: x[0]):
            builder.append(f'  {allele_count_formatter % n_alleles} alleles: {count} variants')
        builder.append(line_break)
        builder.append('Variants per contig')
        builder.append('-------------------')
        for contig, count in sorted(self.variants_per_contig.items(), key=lambda x: contig_idx[x[0]]):
            builder.append(f'  {contig_formatter % contig}: {count} variants')
        builder.append(line_break)
        builder.append('Allele type distribution')
        builder.append('------------------------')
        for allele_type, count in Counter(self.allele_types).most_common():
            summary = f'  {allele_type_formatter % allele_type}: {count} alternate alleles'
            if allele_type == 'SNP':
                nti = self.nti
                ntv = self.ntv
                summary += f' (Ti: {nti}, Tv: {ntv}, ratio: {nti / ntv:.2f})'
            builder.append(summary)
        builder.append(line_break)
        return '\n'.join(builder)

    def _html_string(self):
        contig_idx = {contig: i for i, contig in enumerate(self.rg.contigs)}

        import html
        builder = []
        builder.append('<p><b>Variant summary:</b></p>')
        builder.append('<ul>')
        builder.append(f'<li><p>Total variants: {self.n_variants}</p></li>')

        builder.append('<li><p>Alleles per variant:</p>')
        builder.append('<table><thead style="font-weight: bold;">')
        builder.append('<tr><th>Number of alleles</th><th>Count</th></tr></thead><tbody>')
        for n_alleles, count in sorted(self.alleles_per_variant.items(), key=lambda x: x[0]):
            builder.append('<tr>')
            builder.append(f'<td>{n_alleles}</td>')
            builder.append(f'<td>{count}</td>')
            builder.append('</tr>')
        builder.append('</tbody></table>')
        builder.append('</li>')

        builder.append('<li><p>Counts by allele type:</p>')
        builder.append('<table><thead style="font-weight: bold;">')
        builder.append('<tr><th>Allele type</th><th>Count</th></tr></thead><tbody>')
        for allele_type, count in Counter(self.allele_types).most_common():
            builder.append('<tr>')
            builder.append(f'<td>{html.escape(allele_type)}</td>')
            builder.append(f'<td>{count}</td>')
            builder.append('</tr>')
        builder.append('</tbody></table>')
        builder.append('</li>')

        builder.append('<li><p>Transitions/Transversions:</p>')
        builder.append('<table><thead style="font-weight: bold;">')
        builder.append('<tr><th>Metric</th><th>Value</th></tr></thead><tbody>')
        builder.append(f'<tr><td>Transitions</td><td>{self.nti}</td></tr>')
        builder.append(f'<tr><td>Transversions</td><td>{self.ntv}</td></tr>')
        builder.append(f'<tr><td>Ratio</td><td>{self.nti / self.ntv:.2f}</td></tr>')
        builder.append('</tbody></table>')
        builder.append('</li>')

        builder.append('<li><p>Variants per contig:</p>')
        builder.append('<table><thead style="font-weight: bold;">')
        builder.append('<tr><th>Contig</th><th>Count</th></tr></thead><tbody>')
        for contig, count in sorted(self.variants_per_contig.items(), key=lambda x: contig_idx[x[0]]):
            builder.append('<tr>')
            builder.append(f'<td>{html.escape(contig)}</td>')
            builder.append(f'<td>{count}</td>')
            builder.append('</tr>')
        builder.append('</tbody></table>')
        builder.append('</li>')

        builder.append('</ul>')
        return ''.join(builder)


@typecheck(mt=oneof(Table, MatrixTable), show=bool, handler=anytype)
def summarize_variants(mt: Union[MatrixTable, MatrixTable], show=True, *, handler=None):
    """Summarize the variants present in a dataset and print the results.

    Examples
    --------
    >>> hl.summarize_variants(dataset)  # doctest: +SKIP
    ==============================
    Number of variants: 346
    ==============================
    Alleles per variant
    -------------------
      2 alleles: 346 variants
    ==============================
    Variants per contig
    -------------------
      20: 346 variants
    ==============================
    Allele type distribution
    ------------------------
            SNP: 301 alleles
       Deletion: 27 alleles
      Insertion: 18 alleles
    ==============================

    Parameters
    ----------
    mt : :class:`.MatrixTable` or :class:`.Table`
        Matrix table with a variant (locus / alleles) row key.
    show : :obj:`bool`
        If ``True``, print results instead of returning them.
    handler

    Notes
    -----
    The result returned if `show` is ``False`` is a  :class:`.Struct` with
    five fields:

    - `n_variants` (:obj:`int`): Number of variants present in the matrix table.
    - `allele_types` (:obj:`dict` [:obj:`str`, :obj:`int`]): Number of alternate alleles in
      each allele allele category.
    - `contigs` (:obj:`dict` [:obj:`str`, :obj:`int`]): Number of variants on each contig.
    - `allele_counts` (:obj:`dict` [:obj:`int`, :obj:`int`]): Number of variants broken down
      by number of alleles (biallelic is 2, for example).
    - `r_ti_tv` (:obj:`float`): Ratio of transition alternate alleles to
      transversion alternate alleles.

    Returns
    -------
    :obj:`None` or :class:`.Struct`
        Returns ``None`` if `show` is ``True``, or returns results as a struct.
    """
    require_row_key_variant(mt, 'summarize_variants')
    if isinstance(mt, MatrixTable):
        ht = mt.rows()
    else:
        ht = mt
    allele_pairs = hl.range(1, hl.len(ht.alleles)).map(lambda i: (ht.alleles[0], ht.alleles[i]))

    def explode_result(alleles):
        ref, alt = alleles
        return (hl.agg.counter(hl.allele_type(ref, alt)),
                hl.agg.count_where(hl.is_transition(ref, alt)),
                hl.agg.count_where(hl.is_transversion(ref, alt)))

    (allele_types, nti, ntv), contigs, allele_counts, n_variants = ht.aggregate(
        (hl.agg.explode(explode_result, allele_pairs),
         hl.agg.counter(ht.locus.contig),
         hl.agg.counter(hl.len(ht.alleles)),
         hl.agg.count()))
    rg = ht.locus.dtype.reference_genome
    if show:
        summary = _VariantSummary(rg, n_variants, allele_counts, contigs, allele_types, nti, ntv)
        if handler is None:
            handler = hl.utils.default_handler()
        handler(summary)
    else:
        return hl.Struct(allele_types=allele_types,
                         contigs=contigs,
                         allele_counts=allele_counts,
                         n_variants=n_variants,
                         r_ti_tv=nti / ntv)


@typecheck(ds=oneof(hl.MatrixTable, hl.vds.VariantDataset),
           min_af=numeric,
           max_af=numeric,
           min_dp=int,
           max_dp=int,
           min_gq=int,
           ref_AF=nullable(expr_float64))
def compute_charr(
        ds: Union[hl.MatrixTable, hl.vds.VariantDataset],
        min_af: float = 0.05,
        max_af: float = 0.95,
        min_dp: int = 10,
        max_dp: int = 100,
        min_gq: int = 20,
        ref_AF: Optional[hl.Float64Expression] = None
):
    """Compute CHARR, the DNA sample contamination estimator.

    .. include:: _templates/experimental.rst

    Notes
    -----

    The returned table has the sample ID field, plus the field:

     - `charr` (float64): CHARR contamination estimation.

    Note
    -----
    It is possible to use gnomAD reference allele frequencies with the following:

    >>> gnomad_sites = hl.experimental.load_dataset('gnomad_genome_sites', version='3.1.2') # doctest: +SKIP
    >>> charr_result = hl.compute_charr(mt, ref_af=(1 - gnomad_sites[mt.row_key].freq[1])) # doctest: +SKIP

    Parameters
    ----------
    ds : :class:`.MatrixTable` or :class:`.VariantDataset`
        Dataset.
    min_af
        Minimum reference allele frequency to filter variants.
    max_af
        Maximum reference allele frequency to filter variants.
    min_dp
        Minimum sequencing depth to filter variants.
    max_dp
        Maximum sequencing depth to filter variants.
    min_gq
        Minimum genotype quality to filter variants
    ref_AF
        Reference AF expression. Necessary when the sample size is below 10,000.

    Returns
    -------
    :class:`.Table`
    """

    # Determine whether the input data is in the VDS format; if not, convert matrixtable to VDS and extract only the variant call information
    if isinstance(ds, hl.vds.VariantDataset):
        mt = ds.variant_data
    else:
        mt = ds

    if all(x in mt.entry for x in ['LA', 'LAD', 'LGT', 'GQ']):
        ad_field = 'LAD'
        gt_field = 'LGT'
    elif all(x in mt.entry for x in ['AD', 'GT', 'GQ']):
        ad_field = 'AD'
        gt_field = 'GT'
    else:
        raise ValueError(f"'compute_charr': require a VDS or MatrixTable with fields LAD/LAD/LGT/GQ/DP or AD/GT/GQ/DP,"
                         f" found entry fields {list(mt.entry)}")
    # Annotate reference allele frequency when it is not defined in the original data, and name it 'ref_AF'.
    ref_af_field = '__ref_af'
    if ref_AF is None:
        n_samples = mt.count_cols()
        if n_samples < 10000:
            raise ValueError("'compute_charr': with fewer than 10,000 samples, require a reference AF in 'reference_data_source'.")

        n_alleles = 2 * n_samples
        mt = mt.annotate_rows(
            **{ref_af_field: 1 - hl.agg.sum(mt[gt_field].n_alt_alleles()) / n_alleles}
        )
    else:
        mt = mt.annotate_rows(**{ref_af_field: ref_AF})

    # Filter to autosomal biallelic SNVs with reference allele frequency within the range (min_af, max_af)
    rg = mt.locus.dtype.reference_genome.name
    if rg == 'GRCh37':
        mt = hl.filter_intervals(mt, [hl.parse_locus_interval('1-22', reference_genome=rg)])
    elif rg == 'GRCh38':
        mt = hl.filter_intervals(mt, [hl.parse_locus_interval('chr1-chr22', reference_genome=rg)])
    else:
        mt = mt.filter_rows(mt.locus.in_autosome())

    mt = mt.filter_rows(
        (hl.len(mt.alleles) == 2)
        & hl.is_snp(mt.alleles[0], mt.alleles[1])
        & (mt[ref_af_field] > min_af)
        & (mt[ref_af_field] < max_af)
    )

    # Filter to variant calls with GQ above min_gq and DP within the range (min_dp, max_dp)
    ad_dp = mt['DP'] if 'DP' in mt.entry else hl.sum(mt[ad_field])
    mt = mt.filter_entries(
        mt[gt_field].is_hom_var() & (mt.GQ >= min_gq) & (ad_dp >= min_dp) & (ad_dp <= max_dp)
    )

    # Compute CHARR
    mt = mt.select_cols(
        charr=hl.agg.mean((mt[ad_field][0] / (mt[ad_field][0] + mt[ad_field][1])) / mt[ref_af_field])
    )

    mt = mt.select_globals(
        af_min=min_af,
        af_max=max_af,
        dp_min=min_dp,
        dp_max=max_dp,
        gq_min=min_gq,
    )

    return mt.cols()
