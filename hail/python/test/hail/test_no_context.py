import unittest

import hail as hl


class Tests(unittest.TestCase):
    def test_get_reference_before_init(self):
        hl.get_reference('GRCh37')  # Should be no error
