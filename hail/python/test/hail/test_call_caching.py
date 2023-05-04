
import hail as hl
import os.path as path

from hail.utils.misc import new_local_temp_dir
from .helpers import with_flags

def test_call_cache_creation():
    tmpdir = new_local_temp_dir()

    @with_flags(use_fast_restarts='1', cachedir=tmpdir)
    def test():
        (hl.utils.range_table(10)
         .annotate(another_field=5)
         ._force_count())
        assert path.exists(f'{tmpdir}/{hl.__pip_version__}')

    return test()
