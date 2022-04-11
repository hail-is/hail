import hail as hl

from ..helpers import skip_unless_service_backend

@skip_unless_service_backend()
def test_tiny_driver_has_tiny_memory():
    try:
        hl.utils.range_table(100_000_000, 50).to_pandas()
    except Exception as exc:
        assert 'java.lang.OutOfMemoryError: Java heap space' in exc.args[0]
    else:
        assert Fail

@skip_unless_service_backend()
def test_big_driver_has_big_memory():
    old_driver_cores = hl.current_backend().driver_cores
    old_driver_memory = hl.current_backend().driver_memory
    try:
        hl.current_backend().driver_cores = 8
        hl.current_backend().driver_memory = 'highmem'
        t = hl.utils.range_table(100_000_000, 50)
        # The pytest (client-side) worker dies if we try to realize all 100M rows in memory.
        # Instead, we realize the 100M rows in memory on the driver and then take just the first 10M
        # rows back to the client.
        hl.eval(t.aggregate(hl.agg.collect(t.idx), _localize=False)[:10_000_000])
    finally:
        hl.current_backend().driver_cores = old_driver_cores
        hl.current_backend().driver_memory = old_driver_memory
