import hail as hl
import pytest
import re

from hail.utils.java import FatalError


def test_exceptions_from_workers_have_stack_traces():
    ht = hl.utils.range_table(10, n_partitions=10)
    ht = ht.annotate(x=hl.int(1)//hl.int(hl.rand_norm(0, 0.1)))
    with pytest.raises(
        FatalError,
        match=re.compile(
            '.*java.lang.Math.floorDiv(Math.java:1052).*BackendUtils.scala:[0-9]+\n.*',
            re.DOTALL)):
        ht.collect()
