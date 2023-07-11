import unittest
import hail as hl
from .helpers import *


class Tests(unittest.TestCase):
    @run_in('all')
    def test_get_reference_before_init(self):
        hl.get_reference('GRCh37') # Should be no error
