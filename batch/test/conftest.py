import random
import logging
import os

import pytest

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def log_before_after():
    log.info('starting test')
    yield
    log.info('ending test')


def pytest_collection_modifyitems(config, items):
    n_splits = int(os.environ.get('HAIL_RUN_IMAGE_SPLITS', '1'))
    split_index = int(os.environ.get('HAIL_RUN_IMAGE_SPLIT_INDEX', '-1'))
    if n_splits <= 1:
        return
    if not (0 <= split_index < n_splits):
        raise RuntimeError(f"invalid split_index: index={split_index}, n_splits={n_splits}\n  env={os.environ}")
    skip_this = pytest.mark.skip(reason="skipped in this round")

    weighted_items = [
        (item, dict(item.user_properties).get('duration_relative_to_average', 1))
        for item in items
    ]

    random.seed(0)
    random.shuffle(weighted_items)

    total_weight = sum(w for _, w in weighted_items)
    cumsum = 0

    this_split_lower_bound = split_index * total_weight / n_splits
    this_split_upper_bound = (split_index + 1) * total_weight / n_splits

    for item, weight in weighted_items:
        if not (this_split_lower_bound <= cumsum < this_split_upper_bound):
            item.add_marker(skip_this)
        cumsum += weight
