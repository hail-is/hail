import unittest

import hail as hl
from .helpers import startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def test_init_hail_context_twice(self):
        hl.init(idempotent=True)  # Should be no error
        hl.stop()
        hl.init(idempotent=True)  # Should be no error
        hl.init(hl.spark_context(), idempotent=True)  # Should be no error

    def test_top_level_functions_are_do_not_error(self):
        hl.current_backend()
        hl.debug_info()
