import hail as hl
from hail.typecheck import typecheck, numeric, nullable
import random


@typecheck(mt=hl.MatrixTable,
           n_rounds=int,
           pairs_per_generation_multiplier=numeric,
           children_per_pair=int,
           seed=nullable(int))
def simulate_random_mating(mt,
                           n_rounds=1,
                           pairs_per_generation_multiplier=0.5,
                           children_per_pair=2,
                           seed=None):
    """Simulate random diploid mating to produce new individuals.

    .. include:: _templates/experimental.rst

    Exmaples
    --------

    >>> dataset_sim = hl.simulate_random_mating(dataset, n_rounds=2, pairs_per_generation_multiplier=0.5)

    Parameters
    ----------
    mt
    n_rounds : :obj:`int`
        Number of rounds of mating.
    pairs_per_generation_multiplier  : :obj:`float`
        Ratio of number of mating pairs to current population size for each round of mating.
    children_per_pair  : :obj:`int`
        Number of children per mating pair.
    Returns
    -------
    :class:`.MatrixTable`
    """
    if pairs_per_generation_multiplier <= 0:
        raise ValueError(
            f"simulate_random_mating: 'generation_size_multiplier' must be greater than zero: got {pairs_per_generation_multiplier}")
    if n_rounds < 1:
        raise ValueError(f"simulate_random_mating: 'n_rounds' must be positive: got {n_rounds}")

    mt = mt.select_entries('GT')
    ht = mt.localize_entries('__entries', '__cols')

    ns = mt.count_cols()

    # dict of true nonzero relatedness. indeed by tuples of (id1, id2) where a pair is stored with the larger (later) id first.
    from collections import defaultdict
    relatedness = defaultdict(dict)

    def get_rel(s1, s2):
        smaller, larger = (s1, s2) if s1 < s2 else s2, s1
        return relatedness[larger].get(smaller, 0.0)

    def set_rel(from_s, to_s, value, fwd, generation_start):
        relatedness[from_s][to_s] = value
        if to_s >= generation_start:
            fwd[to_s][from_s] = value

    samples = [(i, f'founder_{i}', None, None) for i in range(ns)]
    print(f'simulate_random_mating: {len(samples)} founders, {n_rounds} rounds of mating to do')
    last_generation_start_idx = 0
    indices = []

    last_gen_fwd = defaultdict(dict)

    if seed is not None:
        random.seed(seed)
    for generation in range(n_rounds):
        last_generation_end = len(samples)
        mating_generation_size = last_generation_end - last_generation_start_idx

        new_pairs = int(mating_generation_size * pairs_per_generation_multiplier)

        curr_gen_fwd = defaultdict(dict)

        curr_sample_idx = len(samples)
        parent_to_child = defaultdict(set)
        for pair in range(new_pairs):
            mother = int(random.uniform(last_generation_start_idx, last_generation_end))
            father = int(last_generation_start_idx + (
                    mother + random.uniform(1, mating_generation_size)) % mating_generation_size)

            mother_rel = {**relatedness[mother], **last_gen_fwd[mother]}
            father_rel = {**relatedness[father], **last_gen_fwd[father]}

            merged_parent_rel = {}

            for k, v in mother_rel.items():
                merged_parent_rel[k] = .5 * v
            for k, v in father_rel.items():
                if k in mother_rel:
                    merged_parent_rel[k] += .5 * v
                else:
                    merged_parent_rel[k] = .5 * v

            if mother in merged_parent_rel:
                assert father in merged_parent_rel
                merged_parent_rel[mother] += 0.25
                merged_parent_rel[father] += 0.25
            else:
                merged_parent_rel[mother] = 0.25
                merged_parent_rel[father] = 0.25

            # here mother/father be the same, since the edge from father to mother
            # and the edge from mother to father must be identical.
            sibling_rel = merged_parent_rel[mother]

            for child in range(children_per_pair):
                samples.append(
                    (curr_sample_idx, f'generation_{generation + 1}_pair_{pair}_child_{child}', mother, father))
                for k, v in merged_parent_rel.items():
                    set_rel(curr_sample_idx, k, v, curr_gen_fwd, last_generation_start_idx)

                mother_other_kids = parent_to_child[mother]
                father_other_kids = parent_to_child[father]
                for sib in mother_other_kids:
                    if sib in father_other_kids:
                        rel = sibling_rel
                    else:
                        _, _, sib_mom, sib_dad = samples[sib]
                        other_parent = sib_mom if sib_mom != mother else sib_dad
                        rel = .125 + get_rel(mother, other_parent) / 2
                    set_rel(curr_sample_idx, sib, rel, curr_gen_fwd, last_generation_start_idx)
                for sib in father_other_kids:
                    if sib not in mother_other_kids:
                        _, _, sib_mom, sib_dad = samples[sib]
                        other_parent = sib_mom if sib_mom != father else sib_dad
                        set_rel(curr_sample_idx, sib, 0.125 + get_rel(father, other_parent) / 2, curr_gen_fwd,
                                last_generation_start_idx)

                mother_other_kids.add(curr_sample_idx)
                father_other_kids.add(curr_sample_idx)

                curr_sample_idx += 1
        print(
            f'simulate_random_mating: generation {generation + 1}: '
            f'{curr_sample_idx - last_generation_end} new samples, '
            f'for a total of {len(samples)}')

        indices.append((last_generation_end, curr_sample_idx))
        last_generation_start_idx = last_generation_end
        last_gen_fwd = curr_gen_fwd

    ht = ht.annotate_globals(__samples=hl.literal(samples, dtype='tarray<ttuple(int32, str, int32, int32)>')
                             .map(lambda t: hl.struct(sample_idx=t[0], s=t[1], mother=t[2], father=t[3])),
                             __indices=indices,
                             relatedness=relatedness)

    def simulate_mating_calls(prev_generation_calls, samples, indices):
        new_samples = hl.range(indices[0], indices[1]) \
            .map(lambda i: samples[i]) \
            .map(lambda samp: hl.call(prev_generation_calls[samp.mother][hl.rand_int32(0, 2)],
                                      prev_generation_calls[samp.father][hl.rand_int32(0, 2)]))
        return prev_generation_calls.extend(new_samples)

    samples = ht.__samples
    ht = ht.annotate(__new_entries=hl.fold(
        lambda prev_calls, indices: simulate_mating_calls(prev_calls, samples, indices),
        ht.__entries.GT,
        ht.__indices).map(lambda gt: hl.struct(GT=gt)))
    ht = ht.drop('__entries', '__cols', '__indices')
    return ht._unlocalize_entries('__new_entries', '__samples', list('s'))
