from glob import glob
from os import path

import hail as hl

from hail.utils.misc import new_local_temp_dir
from .helpers import with_flags

def test_call_cache_creation():
    """Asserts creation of execution cache folder"""
    tmpdir = f"{new_local_temp_dir()}/{hl.__pip_version__}"
    assert not path.exists(tmpdir)

    @with_flags(use_fast_restarts='1', cachedir=tmpdir)
    def test():
        (hl.utils.range_table(10)
         .annotate(another_field=5)
         ._force_count())
        assert path.exists(tmpdir)
        assert len(glob(tmpdir)) == 1

    return test()
