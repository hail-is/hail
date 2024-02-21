from typing import Union

import numpy as np

import hail as hl
from hail import (
    CallExpression,
    expr_call,
    king,
    NumericExpression,
    MatrixTable,
    expr_numeric,
    Struct,
    matrix_table_source,
    SetExpression,
    expr_any,
)
from hail.linalg import BlockMatrix, _eigh
from hail.typecheck import typecheck


@typecheck(genotypes=expr_call, relatedness_threshold=expr_numeric, divergence_threshold=expr_numeric)
def _partition_samples(
    genotypes: CallExpression,
    relatedness_threshold: Union[int, float, NumericExpression] = 0.025,
    divergence_threshold: Union[int, float, NumericExpression] = 0.025,
):
    """
    Identify a diverse subset of unrelated individuals that is representative
    of all ancestries in the sample using the PC-AiR algorithm for partitioning.

    Notes
    -----
    We say that two samples are **related** if their kinship coefficient is greater than the relatedness threshold.
    Otherwise, they are **unrelated**.
    We say that two samples are **divergent** if their ancestral divergence is less than
    the negation of the divergence threshold.

    This method estimates the kinship coefficient and the ancestral divergence between all samples
    using the `KING-robust, between-family kinship coefficient <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3025716/>`_
    estimator.

    This method returns an unrelated set and a related set.
    The intersection of these sets is empty, and the union of these sets is the set of all samples.
    Thus, the unrelated set and the related set are a **partition** of the set of all samples.

    The samples in the unrelated set are mutually unrelated.

    The partitioning algorithm is documented in the
    `PC-AiR paper <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4836868/#APP2title>`_.

    Parameters
    ----------
    genotypes : :class:`.CallExpression`
        A call expression representing the genotype calls.
    relatedness_threshold : :obj:`int` or :obj:`float` or :class:`.NumericExpression`
        The relatedness threshold. The default is 0.025.
    divergence_threshold : :obj:`int` or :obj:`float` or :class:`.NumericExpression`
        The divergence threshold. The default is 0.025.

    Returns
    -------
    :obj:`set` of :class:`.Struct`
        The keys of the samples in the unrelated set.
    :obj:`set` of :class:`.Struct`
        The keys of the samples in the related set.
    """
    # The variable names in this method are based on the notation in the PC-AiR paper.
    # TODO: The paper uses the within-family estimate for ancestral divergence
    # TODO: The paper suggests using the within-family estimate for relatedness as well
    # king returns the KING-robust, between-family kinship estimates for all sample pairs
    pairs: MatrixTable = king(genotypes)
    pairs = pairs.cache()

    assert len(pairs.row_key) == len(pairs.col_key)
    assert isinstance(pairs.row_key.dtype, hl.tstruct) and isinstance(pairs.col_key.dtype, hl.tstruct)
    assert pairs.row_key.dtype.types == pairs.col_key.dtype.types

    def keys_are_different():
        return hl.any(
            list(
                pairs[left_field] != pairs[right_field]
                for left_field, right_field in zip(pairs.row_key.dtype, pairs.col_key.dtype)
            )
        )

    pairs = pairs.annotate_cols(eta=hl.agg.count_where(keys_are_different() & (pairs.phi > relatedness_threshold)))
    pairs = pairs.annotate_cols(
        delta=hl.agg.count_where(
            keys_are_different() & (pairs.phi < relatedness_threshold) & (pairs.phi < -divergence_threshold)
        )
    )
    pairs = pairs.annotate_cols(
        gamma=hl.agg.sum(hl.if_else(keys_are_different() & (pairs.phi > relatedness_threshold), pairs.phi, 0))
    )

    unrelated = pairs.aggregate_cols(hl.agg.collect_as_set(pairs.col_key))
    related = set()
    samples = pairs.cols()
    samples_key = samples.key

    while True:
        samples = samples.order_by(hl.desc(samples.eta), samples.delta, samples.gamma)
        samples = samples.cache()
        selected_sample = samples.head(1).collect()[0]

        if selected_sample.eta <= 0:
            return hl.set(unrelated), hl.set(related)

        selected_sample = Struct(**{field: selected_sample[field] for field in samples_key.dtype})
        unrelated -= {selected_sample}
        related |= {selected_sample}

        # A sample is "affected" if the associated value of eta will change
        # due to the removal of the selected sample from the unrelated set
        assert len(pairs.row_key.dtype) == len(samples_key.dtype)
        are_keys_equal = hl.all(
            list(
                pairs[left_field] == selected_sample[right_field]
                for left_field, right_field in zip(pairs.row_key.dtype, samples_key.dtype)
            )
        )
        affected_samples = pairs.filter_rows(are_keys_equal)
        affected_samples = affected_samples.annotate_cols(
            is_affected=hl.agg.any(affected_samples.phi > relatedness_threshold)
        )
        affected_samples = affected_samples.filter_cols(affected_samples.is_affected)
        affected_samples = affected_samples.aggregate_cols(
            hl.agg.collect_as_set(list(affected_samples[field] for field in affected_samples.col_key.dtype)),
            _localize=False,
        )
        # Subtract 1 from eta for the affected samples
        samples = samples.annotate(
            eta=hl.if_else(
                affected_samples.contains(list(samples[field] for field in samples_key.dtype)),
                samples.eta - 1,
                samples.eta,
            )
        )
        # Set eta to 0 for the selected sample
        are_keys_equal = hl.all(list(samples[field] == selected_sample[field] for field in samples_key.dtype))
        samples = samples.annotate(eta=hl.if_else(are_keys_equal, 0, samples.eta))


@typecheck(genotypes=expr_call, unrelated=expr_any)
def _standardize(genotypes: CallExpression, unrelated: Union[set, SetExpression]):
    """
    Standardize the genotypes.

    If the number of alternate alleles at locus :math:`s` for sample :math:`i` is :math:`g_{is}`
    and the unrelated set is :math:`\\mathcal{U}_s`,
    then the standardized genotype for individual :math:`i` at locus :math:`s` is

    .. math::

       z_{is} = \frac{g_{is} - 2 \\hat{p}^u_s}{\\sqrt{2 \\hat{p}^u_s (1 - \\hat{p}^u_s)}},

    where

    .. math::

       \\hat{p}^u_s = \frac{1}{2 |\\mathcal{U}_s|} \\sum_{i \\in \\mathcal{U}_s} g_{is},

    Notes
    -----
    This method gives missing genotypes a standardized genotype of 0.

    The standardized genotypes have a variant-wise mean of 0 and standard deviation of 1.

    Parameters
    ----------
    genotypes : :class:`.CallExpression`
        A call expression representing the genotype calls.
    unrelated : :class:`.SetExpression`
        The keys of the samples in the unrelated set.
    """
    matrix_table = matrix_table_source('_standardize/genotypes', genotypes)

    # Calculate allele frequency estimates
    matrix_table = matrix_table.select_entries(alt_allele_count=genotypes.n_alt_alleles())
    matrix_table = matrix_table.annotate_rows(
        alt_allele_sum=hl.agg.sum(
            hl.if_else(unrelated.contains(matrix_table.col_key), matrix_table.alt_allele_count, 0)
        )
    )
    matrix_table = matrix_table.annotate_rows(
        sample_count=hl.agg.count_where(
            unrelated.contains(matrix_table.col_key) & hl.is_defined(matrix_table.alt_allele_count)
        )
    )
    matrix_table = matrix_table.annotate_rows(
        alt_allele_frequency_estimate=1 / 2 * matrix_table.alt_allele_sum / matrix_table.sample_count
    )

    # Calculate the standardized genotypes
    matrix_table = matrix_table.annotate_entries(
        standardized_genotype=hl.if_else(
            hl.is_defined(matrix_table.alt_allele_count),
            (matrix_table.alt_allele_count - 2 * matrix_table.alt_allele_frequency_estimate)
            / hl.sqrt(
                2 * matrix_table.alt_allele_frequency_estimate * (1 - matrix_table.alt_allele_frequency_estimate)
            ),
            0,
        )
    )
    return matrix_table.standardized_genotype


@typecheck(genotypes=expr_call, relatedness_threshold=expr_numeric, divergence_threshold=expr_numeric)
def pc_air(
    genotypes: CallExpression,
    *,
    relatedness_threshold: Union[int, float, NumericExpression] = 0.025,
    divergence_threshold: Union[int, float, NumericExpression] = 0.025,
):
    """
    Perform PC-AiR (principal components analysis in related samples) on the genotypes.

    PC-Air is a robust method for inferring population structure from genome-screen data
    described `here <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4836868/>`_.

    Notes
    -----
    This method uses kinship coefficients to determine whether two samples are related.
    Two samples are considered related if their kinship coefficient is greater than the relatedness threshold.
    Otherwise, they are considered unrelated.

    Similarly, this method uses kinship coefficient estimates to determine if two samples are ancestrally divergent.
    Two samples are considered divergent if their ancestral divergence is less than
    the negation of the divergence threshold.

    Parameters
    ----------
    genotypes : :class:`.CallExpression`
        A call expression representing the genotype calls.
    relatedness_threshold : :obj:`int` or :obj:`float` or :class:`.NumericExpression`
        The relatedness threshold. The default is 0.025.
    divergence_threshold : :obj:`int` or :obj:`float` or :class:`.NumericExpression`
        The divergence threshold. The default is 0.025.

    Returns
    -------
    :class:`.BlockMatrix`
        A block matrix with the principal components as the columns.
    """
    unrelated, related = _partition_samples(genotypes, relatedness_threshold, divergence_threshold)
    standardized_genotypes = _standardize(genotypes, unrelated)
    matrix_table = matrix_table_source('pc_air/standardized_genotypes', standardized_genotypes)
    field_name = matrix_table._fields_inverse[standardized_genotypes]
    unrelated_genotypes = matrix_table.filter_cols(unrelated.contains(matrix_table.col_key))[field_name]
    related_genotypes = matrix_table.filter_cols(related.contains(matrix_table.col_key))[field_name]

    # https://en.m.wikipedia.org/wiki/Principal_component_analysis#Singular_value_decomposition
    # https://en.m.wikipedia.org/wiki/Singular_value_decomposition#Relation_to_eigenvalue_decomposition

    # The variable names below correspond to the notation in the PC-AiR paper
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4836868/#S2title

    snp_count = matrix_table.count_rows()
    zu = BlockMatrix.from_entry_expr(unrelated_genotypes).T
    phi = zu @ zu.T / snp_count

    # Here it is necessary to localize because Hail does not offer a method to compute the full
    # singular value decomposition (SVD) of a BlockMatrix.
    # (Hail's BlockMatrix.svd() method computes the reduced SVD, not the full SVD.)
    # TODO: Avoid localizing
    phi = phi.to_numpy() if isinstance(phi, BlockMatrix) else phi
    l, v = _eigh(phi)
    w = zu.T @ v

    zr = BlockMatrix.from_entry_expr(related_genotypes).T
    q = zr @ w @ np.linalg.inv(np.diag(l)) / snp_count
    q = q.to_numpy() if isinstance(q, BlockMatrix) else q

    # Stack v on top of q
    big_gamma = np.row_stack((v, q))

    return BlockMatrix.from_numpy(big_gamma)
