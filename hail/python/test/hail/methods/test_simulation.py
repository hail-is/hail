import hail as hl

from ..helpers import *


def test_mating_simulation():
    mt = get_dataset()

    n_samples = mt.count_cols()

    assert (
        hl.simulate_random_mating(mt, n_rounds=1, generation_size_multiplier=2, keep_founders=False).count_cols()
        == n_samples * 2
    )
    assert (
        hl.simulate_random_mating(mt, n_rounds=4, generation_size_multiplier=2, keep_founders=False).count_cols()
        == n_samples * 16
    )
    assert (
        hl.simulate_random_mating(mt, n_rounds=2, generation_size_multiplier=1, keep_founders=False).count_cols()
        == n_samples
    )
    assert (
        hl.simulate_random_mating(mt, n_rounds=2, generation_size_multiplier=2, keep_founders=True).count_cols()
        == n_samples * 9
    )

    hl.simulate_random_mating(mt, n_rounds=2, generation_size_multiplier=0.5, keep_founders=True)._force_count_rows()
