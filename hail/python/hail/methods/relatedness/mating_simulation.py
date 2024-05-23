import hail as hl
from hail.typecheck import typecheck, numeric


@typecheck(mt=hl.MatrixTable, n_rounds=int, generation_size_multiplier=numeric, keep_founders=bool)
def simulate_random_mating(mt, n_rounds=1, generation_size_multiplier=1.0, keep_founders=True):
    """Simulate random diploid mating to produce new individuals.

    Parameters
    ----------
    mt
    n_rounds : :obj:`int`
        Number of rounds of mating.
    generation_size_multiplier  : :obj:`float`
        Ratio of number of offspring to current population for each round of mating.
    keep_founders :obj:`bool`
        If true, keep all founders and intermediate generations in the final sample list. If
        false, keep only offspring in the last generation.

    Returns
    -------
    :class:`.MatrixTable`
    """
    if generation_size_multiplier <= 0:
        raise ValueError(
            f"simulate_random_mating: 'generation_size_multiplier' must be greater than zero: got {generation_size_multiplier}"
        )
    if n_rounds < 1:
        raise ValueError(f"simulate_random_mating: 'n_rounds' must be positive: got {n_rounds}")

    ck = list(mt.col_key)[0]

    mt = mt.select_entries('GT')

    ht = mt.localize_entries('__entries', '__cols')

    ht = ht.annotate_globals(
        generation_0=hl.range(hl.len(ht.__cols)).map(
            lambda i: hl.struct(
                s=hl.str('generation_0_idx_') + hl.str(i),
                original=hl.str(ht.__cols[i][ck]),
                mother=hl.missing('int32'),
                father=hl.missing('int32'),
            )
        )
    )

    def make_new_generation(prev_generation_tup, idx):
        prev_size = prev_generation_tup[1]
        n_new = hl.int32(hl.floor(prev_size * generation_size_multiplier))
        new_generation = hl.range(n_new).map(
            lambda i: hl.struct(
                s=hl.str('generation_') + hl.str(idx + 1) + hl.str('_idx_') + hl.str(i),
                original=hl.missing('str'),
                mother=hl.rand_int32(0, prev_size),
                father=hl.rand_int32(0, prev_size),
            )
        )
        return (new_generation, (prev_size + n_new) if keep_founders else n_new)

    ht = ht.annotate_globals(
        generations=hl.range(n_rounds).scan(
            lambda prev, idx: make_new_generation(prev, idx), (ht.generation_0, hl.len(ht.generation_0))
        )
    )

    def simulate_mating_calls(prev_generation_calls, new_generation):
        new_samples = new_generation.map(
            lambda samp: hl.call(
                prev_generation_calls[samp.mother][hl.rand_int32(0, 2)],
                prev_generation_calls[samp.father][hl.rand_int32(0, 2)],
            )
        )
        if keep_founders:
            return prev_generation_calls.extend(new_samples)
        else:
            return new_samples

    ht = ht.annotate(
        __new_entries=hl.fold(
            lambda prev_calls, generation_metadata: simulate_mating_calls(prev_calls, generation_metadata[0]),
            ht.__entries.GT,
            ht.generations[1:],
        ).map(lambda gt: hl.struct(GT=gt))
    )
    ht = ht.annotate_globals(
        __new_cols=ht.generations.flatmap(lambda x: x[0]) if keep_founders else ht.generations[-1][0]
    )
    ht = ht.drop('__entries', '__cols', 'generation_0', 'generations')
    return ht._unlocalize_entries('__new_entries', '__new_cols', list('s'))
