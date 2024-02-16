from typing import Union

import pandas as pd

import hail as hl
from hail import CallExpression, expr_call, king, NumericExpression, MatrixTable, expr_numeric, Struct
from hail.typecheck import typecheck


def _partition_samples_pandas(
    genotype: CallExpression, relatedness_threshold: float = 0.025, divergence_threshold: float = 0.025
):
    matrix_table = king(genotype)
    pair_df = matrix_table.entries().to_pandas()
    # Rename column "s_1" to "o"
    pair_df.rename(columns={'s_1': 'o'}, inplace=True)
    # Reorder columns
    pair_df = pair_df[['s', 'o', 'phi']]
    # Drop rows where "s" and "o" are the same
    pair_df = pair_df[pair_df['s'] != pair_df['o']]
    samples_index = pd.Index(pair_df['s'].unique(), name='s')

    eta = pair_df[pair_df['phi'] > relatedness_threshold].groupby('s', sort=False)['phi'].count()
    eta = eta.reindex(samples_index, fill_value=0)

    delta = (
        pair_df[(pair_df['phi'] < relatedness_threshold) & (pair_df['phi'] < -divergence_threshold)]
        .groupby('s', sort=False)['phi']
        .count()
    )
    delta.reindex(samples_index, fill_value=0)

    gamma = pair_df[pair_df['phi'] > relatedness_threshold].groupby('s', sort=False)['phi'].sum()
    gamma = gamma.reindex(samples_index, fill_value=0)

    unrelated = set(pair_df['s'])
    related = set()

    while True:
        max_eta = eta.max()

        if max_eta == 0:
            return unrelated, related

        fraktur_T_star = eta[eta == max_eta].index

        if len(fraktur_T_star) > 1:
            min_delta = delta[delta.index.isin(fraktur_T_star)].min()
            fraktur_T_star = delta[(delta.index.isin(fraktur_T_star)) & (delta == min_delta)].index

            if len(fraktur_T_star) > 1:
                min_gamma = gamma[gamma.index.isin(fraktur_T_star)].min()
                fraktur_T_star = gamma[(gamma.index.isin(fraktur_T_star)) & (gamma == min_gamma)].index[0:1]

        assert len(fraktur_T_star) == 1
        unrelated -= set(fraktur_T_star)
        related |= set(fraktur_T_star)

        selected_sample = fraktur_T_star[0]
        affected_samples = pd.Index(
            pair_df[(pair_df['o'] == selected_sample) & (pair_df['phi'] > relatedness_threshold)]['s']
        )
        eta[affected_samples] -= 1
        eta[selected_sample] = 0
        pass


@typecheck(genotypes=expr_call, relatedness_threshold=expr_numeric, divergence_threshold=expr_numeric)
def _partition_samples(
    genotypes: CallExpression,
    relatedness_threshold: Union[int, float, NumericExpression] = 0.025,
    divergence_threshold: Union[int, float, NumericExpression] = 0.025,
):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4836868/#APP2title
    # Returns the KING-robust, between-family kinship estimates for all sample pairs
    # TODO: The paper uses the within-family estimate for ancestral divergence
    # TODO: The paper suggests using the within-family estimate for relatedness as well
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
            return unrelated, related

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


@typecheck(genotypes=expr_call)
def pc_air(genotypes: CallExpression):
    _unrelated, _related = _partition_samples(genotypes)
    raise NotImplementedError


if __name__ == '__main__':
    mt = hl.read_matrix_table('../../docs/data/example.mt')
    # mt = hl.read_matrix_table('../../../../../data/1kg.mt')
    unrelated, related = _partition_samples(mt.GT)
    pass
