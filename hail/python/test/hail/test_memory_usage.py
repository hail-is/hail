import hail as hl
from hail.utils import new_temp_file
from .helpers import startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


def test_memory_issue_from_9009():
    mt = hl.utils.range_matrix_table(1000, 1, n_partitions=1)
    mt = mt.annotate_entries(x=hl.float(mt.row_idx * mt.col_idx))
    mt = mt.annotate_rows(big=hl.zeros(100_000_000))
    try:
        hl.linalg.BlockMatrix.write_from_entry_expr(mt.x, new_temp_file(), overwrite=True)
    except Exception:
        assert False
