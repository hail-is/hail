import unittest

import hail as hl
from hail.genetics import *
from ..helpers import *

class Tests(unittest.TestCase):

    def test_constructor(self):
        l = Locus.parse('1:100')

        self.assertEqual(l, Locus('1', 100))
        self.assertEqual(l, Locus(1, 100))
        self.assertEqual(l.reference_genome, hl.default_reference())
