import unittest
import hail as hl
from .helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def test_init_hail_context_twice(self):
        hl.init(allow_existing=True) # Should be no error
        self.assertIsNotNone(hl.Env._hc)
