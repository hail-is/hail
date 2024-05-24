import re

import pytest

import hail as hl
from hail.utils.java import FatalError

from .helpers import qobtest


@qobtest
def test_exceptions_from_workers_have_stack_traces():
    ht = hl.utils.range_table(10, n_partitions=10)
    ht = ht.annotate(x=hl.int(1) // hl.int(hl.rand_norm(0, 0.1)))
    pattern = (
        '.*'
        + re.escape('java.lang.Math.floorDiv(Math.java:1052)')
        + '.*'
        + re.escape('(BackendUtils.scala:')
        + '[0-9]+'
        + re.escape(')\n')
        + '.*'
    )
    with pytest.raises(FatalError, match=re.compile(pattern, re.DOTALL)):
        ht.collect()
