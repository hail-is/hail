
"""
Unit tests for PyHail.
"""

import unittest

from pyspark import SparkContext
from pyhail import HailContext

class ContextTests(unittest.TestCase):

    def setUp(self):
        self.sc = SparkContext()
        self.hc = HailContext(self.sc)

    def test(self):
        vds = self.hc.import_annotations_table(
            'src/test/resources/variantAnnotations.tsv',
            'Variant(Chromosome, Position.toInt, Ref, Alt)')
        self.assertEqual(vds.count()['nVariants'], 346)
        
    def tearDown(self):
        self.sc.stop()
