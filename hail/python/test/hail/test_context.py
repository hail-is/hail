import unittest

import hail as hl
from .helpers import startTestHailContext, stopTestHailContext, skip_unless_spark_backend, fails_local_backend, fails_service_backend

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    @skip_unless_spark_backend()
    def test_init_hail_context_twice(self):
        hl.init(idempotent=True)  # Should be no error
        hl.stop()

        hl.init(idempotent=True)
        hl.experimental.define_function(lambda x: x + 2, hl.tint32)
        # ensure functions are cleaned up without error
        hl.stop()

        hl.init(idempotent=True)  # Should be no error
        hl.init(hl.spark_context(), idempotent=True)  # Should be no error

    def test_top_level_functions_are_do_not_error(self):
        hl.current_backend()
        hl.debug_info()

    def test_tmpdir_runs(self):
        isinstance(hl.tmp_dir(), str)
