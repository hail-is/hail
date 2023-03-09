import hail as hl

from ..helpers import *


def test_mating_simulation():
    mt = get_dataset()

    n_samples = mt.count_cols()

    assert hl.simulate_random_mating(mt, n_rounds=1, pairs_per_generation_multiplier=0.5,
                                     children_per_pair=2).count_cols() == n_samples * 2
    assert hl.simulate_random_mating(mt, n_rounds=4, pairs_per_generation_multiplier=0.5,
                                     children_per_pair=2).count_cols() == n_samples * 5
    assert hl.simulate_random_mating(mt, n_rounds=3, pairs_per_generation_multiplier=1,
                                     children_per_pair=2).count_cols() == n_samples + n_samples * 2 + n_samples * 4 + n_samples * 8
    assert hl.simulate_random_mating(mt, n_rounds=2, pairs_per_generation_multiplier=0.5,
                                     children_per_pair=1).count_cols() == n_samples * 1.75

    hl.simulate_random_mating(mt, n_rounds=2, pairs_per_generation_multiplier=0.5,
                              children_per_pair=2)._force_count_rows()


def test_relatedness():
    mt = hl.balding_nichols_model(n_populations=2, n_samples=2, n_variants=2)
    # this configuration forces 2 unrelated mating to produce 2 samples, who mate to produce 2 more
    rel = hl.eval(hl.simulate_random_mating(mt,
                                            n_rounds=2,
                                            pairs_per_generation_multiplier=0.5,
                                            children_per_pair=2).index_globals().relatedness)

    assert rel == {
        0: {},
        1: {},
        2: {
            0: 0.25,
            1: 0.25
        },
        3: {
            0: 0.25,
            1: 0.25,
            2: 0.25,
        },
        4: {
            0: 0.25,
            1: 0.25,
            2: 0.375,
            3: 0.375,
        },
        5: {
            0: 0.25,
            1: 0.25,
            2: 0.375,
            3: 0.375,
            4: 0.375,
        },
    }
