
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
        
        self.test_resources = 'src/test/resources'
        
    def test_context(self):
        self.hc.grep('Mom1', self.test_resources + '/mendel.fam')

        annot = self.hc.import_annotations_table(
            self.test_resources + '/variantAnnotations.tsv',
            'Variant(Chromosome, Position.toInt, Ref, Alt)')
        self.assertEqual(annot.count()['nVariants'], 346)

        # index
        self.hc.index_bgen(self.test_resources + '/example.v11.bgen')

        bgen = self.hc.import_bgen(self.test_resources + '/example.v11.bgen',
                                   sample_file = self.test_resources + '/example.sample')
        self.assertEqual(bgen.count()['nVariants'], 199)

        gen = self.hc.import_gen(self.test_resources + '/example.gen',
                                 sample_file = self.test_resources + '/example.sample')
        self.assertEqual(gen.count()['nVariants'], 199)

        vcf = self.hc.import_vcf(self.test_resources + '/sample2.vcf').split_multi()

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

    def test_dataset(self):
        sample = (self.hc.import_vcf(self.test_resources + '/sample.vcf')
                  .cache())
        sample_split = sample.split_multi()
        
        sample2 = (self.hc.import_vcf(self.test_resources + '/sample2.vcf')
                   .persist())
        sample2_split = sample2.split_multi()
        
        sample2.aggregate_intervals(self.test_resources + '/annotinterall.interval_list',
                                    'N = variants.count()',
                                    '/tmp/annotinter.tsv')
        
        (sample2.annotate_global_expr('global.nVariants = variants.count()')
         .show_globals())
        
        (sample2.annotate_global_list(self.test_resources + '/global_list.txt', 'global.genes',
                                      as_set = True)
         .show_globals())
        
        (sample2.annotate_global_table(self.test_resources + '/global_table.tsv',
                                       'global.genes')
         .show_globals())
        
        (sample2.annotate_samples_expr('sa.nCalled = gs.filter(g => g.isCalled).count()')
         .export_samples('/tmp/sa.tsv', 's = s, nCalled = sa.nCalled'))
        
        sample2.annotate_samples_list(self.test_resources + '/sample2.sample_list',
                                      'sa.listed')
        
        sample2_annot = sample2.annotate_samples_table(
            self.test_resources + '/sampleAnnotations.tsv',
            'Sample',
            code = 'sa.isCase = table.Status == "CASE", sa.qPhen = table.qPhen')
        
        sample2.annotate_samples_vds(sample2_annot,
                                     code = 'sa.isCase = vds.isCase')

        (sample.annotate_variants_bed(self.test_resources + '/example1.bed',
                                      root = 'va.bed')
         .filter_variants_expr('va.bed')
         .count())
        
        (sample2.annotate_variants_expr('va.nCalled = gs.filter(g => g.isCalled).count()')
         .count())
        
        (sample2.annotate_variants_intervals(self.test_resources + '/annotinterall.interval_list',
                                             'va.included',
                                             all = True)
         .count())
        
        (sample2.annotate_variants_loci(self.test_resources + '/sample2_loci.tsv',
                                        'Locus(chr, pos.toInt)',
                                        'va.locus_annot')
         .count())
        
        (sample.annotate_variants_table(self.test_resources + '/variantAnnotations.tsv',
                                        'Variant(Chromosome, Position.toInt, Ref, Alt)',
                                        root = 'va.table')
         .count())

        (sample2.annotate_variants_vds(sample2, code = 'va.good = va.info.AF == vds.info.AF')
         .count())
        
        (sample2_split.concordance(sample2_split))

        downsampled = sample2.downsample_variants(20)
        downsampled.export_variants('/tmp/sample2_loci.tsv', 'chr = v.contig, pos = v.start')
        downsampled.export_variants('/tmp/sample2_variants.tsv', 'v')
        
        (sample2.filter_samples_list(self.test_resources + '/sample2.sample_list')
         .count()['nSamples'] == 56)
        
        sample2_split.export_gen('/tmp/sample2.gen')

        (sample2.filter_genotypes('g.isHet && g.gq > 20')
         .export_genotypes('/tmp/sample2_genotypes.tsv', 'v, s, g.nNonRefAlleles'))
        
        sample2_split.export_plink('/tmp/sample2')
        
        sample2.export_vcf('/tmp/sample2.vcf.bgz')

        sample2.filter_multi().count()
        
        self.assertEqual(sample2.filter_samples_all().count()['nSamples'], 0)

        self.assertEqual(sample2.filter_variants_all().count()['nVariants'], 0)
        
        sample2_dedup = (self.hc.import_vcf([self.test_resources + '/sample2.vcf',
                                             self.test_resources + '/sample2.vcf'])
                         .deduplicate())
        self.assertEqual(sample2_dedup.count()['nVariants'], 735)
        
        (sample2.filter_samples_expr('pcoin(0.5)')
         .export_samples('/tmp/sample2.sample_list', 's'))

        (sample2.filter_variants_intervals(self.test_resources + '/annotinterall.interval_list')
         .count())
        
        self.assertEqual(sample2.filter_variants_list(
            self.test_resources + '/sample2_variants.tsv')
                         .count()['nVariants'], 21)
        
        sample2.grm('gcta-grm-bin', '/tmp/sample2.grm')

        sample2.hardcalls().count()

        sample2_split.ibd('/tmp/sample2.ibd')
        
        sample2.impute_sex().print_schema()

        self.assertEqual(sample2.join(sample2.rename_samples(self.test_resources + '/sample2_rename.tsv'))
                         .count()['nSamples'], 200)

        linreg = (self.hc.import_vcf(self.test_resources + '/regressionLinear.vcf')
                  .split_multi()
                  .annotate_samples_table(self.test_resources + '/regressionLinear.cov',
                                          'Sample',
                                          root = 'sa.cov',
                                          types = 'Cov1: Double, Cov2: Double')
                  .annotate_samples_table(self.test_resources + '/regressionLinear.pheno',
                                          'Sample',
                                          code = 'sa.pheno.Pheno = table.Pheno',
                                          types = 'Pheno: Double',
                                          missing = '0')
                  .annotate_samples_table(self.test_resources + '/regressionLogisticBoolean.pheno',
                                          'Sample',
                                          code = 'sa.pheno.isCase = table.isCase',
                                          types = 'isCase: Boolean',
                                          missing = '0'))
        
        (linreg.linreg('sa.pheno.Pheno', covariates = 'sa.cov.Cov1, sa.cov.Cov2 + 1 - 1')
         .count())
        
        (linreg.logreg('wald', 'sa.pheno.isCase', covariates = 'sa.cov.Cov1, sa.cov.Cov2 + 1 - 1')
         .count())
        
        sample_split.mendel_errors('/tmp/sample.mendel', self.test_resources + '/sample.fam')
        
        sample_split.pca('sa.scores')
        
        self.assertTrue(
            (sample2.repartition(16, shuffle = False)
             .same(sample2)))

        sample2.sparkinfo()
        
        sample_split.tdt(self.test_resources + '/sample.fam')
        
        sample2.typecheck()

        sample2_split.variant_qc().print_schema()
        
        sample2.variants_to_pandas()
        
    def tearDown(self):
        self.sc.stop()
