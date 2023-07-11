import hail as hl

from hail.utils.misc import new_temp_file
from .helpers import with_flags

def test_execution_cache_creation():
    """Asserts creation of execution cache folder"""
    folder = new_temp_file('hail-execution-cache')
    fs = hl.current_backend().fs
    assert not fs.exists(folder)

    @with_flags(use_fast_restarts='1', cachedir=folder)
    def test():
        (hl.utils.range_table(10)
         .annotate(another_field=5)
         ._force_count())
        assert fs.exists(folder)
        assert len(fs.ls(folder)) == 2

    test()
