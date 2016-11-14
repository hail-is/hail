
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
        test_resources = 'src/test/resources'

        self.hc.grep('Mom1', test_resources + '/mendel.fam')

        annot = self.hc.import_annotations_table(
            test_resources + '/variantAnnotations.tsv',
            'Variant(Chromosome, Position.toInt, Ref, Alt)')
        self.assertEqual(annot.count()['nVariants'], 346)

        # index
        self.hc.index_bgen(test_resources + '/example.v11.bgen')

        bgen = self.hc.import_bgen(test_resources + '/example.v11.bgen',
                                   sample_file = test_resources + '/example.sample')
        self.assertEqual(bgen.count()['nVariants'], 199)

        gen = self.hc.import_gen(test_resources + '/example.gen',
                                 sample_file = test_resources + '/example.sample')
        self.assertEqual(gen.count()['nVariants'], 199)

        vcf = self.hc.import_vcf(test_resources + '/sample2.vcf').split_multi()

        vcf.export_plink('/tmp/sample_plink')

        bfile = '/tmp/sample_plink'
        plink = self.hc.import_plink(
            bfile + '.bed', bfile + '.bim', bfile + '.fam')
        self.assertEqual(vcf.count(genotypes = True), plink.count(genotypes = True))
        
        vcf.write('/tmp/sample.vds', overwrite = True)
        vds = self.hc.read('/tmp/sample.vds')

        self.assertTrue(vcf.same(vds))

        bn = self.hc.balding_nichols_model(3, 10, 100, 8)
        bn_count = bn.count()
        self.assertEqual(bn_count['nSamples'], 10)
        self.assertEqual(bn_count['nVariants'], 100)

    def tearDown(self):
        self.sc.stop()
