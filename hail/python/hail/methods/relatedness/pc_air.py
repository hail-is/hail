from typing import Union

import numpy as np

import hail as hl
from hail import (
    CallExpression,
    expr_call,
    NumericExpression,
    MatrixTable,
    expr_numeric,
    matrix_table_source,
)
from hail.linalg import BlockMatrix
from hail.methods.pca import hwe_normalize
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
    :class:`.Table`
        A table with a field `is_in_unrelated` that is true if the sample is in the unrelated set and false otherwise.
        The table shares the same key as the columns of the source matrix table of the genotypes.
    """
    # The Greek-letter variable names in this method are based on the notation in the PC-AiR paper.
    # TODO: The paper uses the within-family estimate for ancestral divergence
    # TODO: The paper suggests using the within-family estimate for relatedness as well
    # king returns the KING-robust, between-family kinship estimates for all sample pairs
    pairs: MatrixTable = hl.king(genotypes)
    pairs = pairs.cache()

    assert len(pairs.row_key) == len(pairs.col_key)
    assert isinstance(pairs.row_key.dtype, hl.tstruct) and isinstance(pairs.col_key.dtype, hl.tstruct)
    assert pairs.row_key.dtype.types == pairs.col_key.dtype.types

    keys_are_different = hl.any(
        list(
            pairs[left_field] != pairs[right_field]
            for left_field, right_field in zip(pairs.row_key.dtype, pairs.col_key.dtype)
        )
    )

    divergence_threshold = hl.min(relatedness_threshold, -divergence_threshold)
    agg_expr = hl.agg.filter(
        keys_are_different,
        hl.struct(
            gamma_relatives=hl.agg.filter(
                pairs.phi > relatedness_threshold,
                hl.struct(
                    gamma=hl.agg.sum(pairs.phi),
                    relatives=hl.agg.collect(pairs.row_key),
                ),
            ),
            delta=hl.agg.count_where(pairs.phi < divergence_threshold),
        ),
    )
    pairs = pairs.annotate_cols(
        delta=agg_expr.delta,
        gamma=agg_expr.gamma_relatives.gamma,
        relatives=agg_expr.gamma_relatives.relatives,
    )
    samples = pairs.cols()
    samples = samples.annotate(is_in_unrelated=True)
    index_names = list(samples.key.dtype)
    samples = samples.to_pandas().set_index(index_names, drop=False)
    samples['eta'] = samples['relatives'].apply(len)

    while True:
        max_eta = samples['eta'].max()

        if max_eta == 0:
            return hl.Table.from_pandas(samples[index_names + ['is_in_unrelated']], key=index_names)

        selected_sample = samples[samples['eta'] == max_eta]

        if len(selected_sample) > 1:
            min_delta = selected_sample['delta'].min()
            selected_sample = selected_sample[selected_sample['delta'] == min_delta]

            if len(selected_sample) > 1:
                min_gamma = selected_sample['gamma'].min()
                selected_sample = selected_sample[selected_sample['gamma'] == min_gamma]

        # Select first index in selected samples
        selected_sample = selected_sample.index[0]

        # Remove the selected sample from the unrelated set
        samples.at[selected_sample, 'is_in_unrelated'] = False

        # Subtract 1 from eta for the relatives
        relatives = samples.at[selected_sample, 'relatives']
        relatives = list(map(lambda index_struct: tuple(index_struct.values()), relatives))
        relatives = [index_values[0] if len(index_values) == 1 else index_values for index_values in relatives]
        samples.loc[relatives, 'eta'] -= 1

        # Set eta to 0 for the selected sample
        samples.at[selected_sample, 'eta'] = 0


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
    eigenvalues : :obj:`list` of :obj:`float`
        The eigenvalues of the principal components.
    scores : :class:`.BlockMatrix`
        The principal component scores.
    loadings : :class:`.Table`
        The principal component loadings.
    """
    partition_table = _partition_samples(genotypes, relatedness_threshold, divergence_threshold)
    unrelated_table = partition_table.filter(partition_table.is_in_unrelated)
    matrix_table = matrix_table_source('pc_air/genotypes', genotypes)
    field_name = matrix_table._fields_inverse[genotypes]
    unrelated_genotypes = matrix_table.semi_join_cols(unrelated_table)[field_name]
    eigenvalues, _, loadings = hl._hwe_normalized_blanczos(unrelated_genotypes, compute_loadings=True)
    _loadings = np.array(loadings.loadings.collect())
    standardized_genotypes = hwe_normalize(genotypes)
    standardized_genotypes = BlockMatrix.from_entry_expr(standardized_genotypes).T
    scores = standardized_genotypes @ _loadings
    return eigenvalues, scores, loadings
