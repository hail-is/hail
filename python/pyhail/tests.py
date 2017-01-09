
"""
Unit tests for PyHail.
"""

import unittest

from pyspark import SparkContext
from pyhail import HailContext, TextTableConfig

hc = None

def setUpModule():
    global hc
    hc = HailContext() # master = 'local[2]')

def tearDownModule():
    global hc
    hc.stop()
    hc = None

class ContextTests(unittest.TestCase):

    def test_context(self):
        test_resources = 'src/test/resources'
        
        hc.grep('Mom1', test_resources + '/mendel.fam')

        annot = hc.import_annotations_table(
            test_resources + '/variantAnnotations.tsv',
            'Variant(Chromosome, Position.toInt, Ref, Alt)')
        self.assertEqual(annot.count()['nVariants'], 346)

        # index
        hc.index_bgen(test_resources + '/example.v11.bgen')

        bgen = hc.import_bgen(test_resources + '/example.v11.bgen',
                                   sample_file = test_resources + '/example.sample')
        self.assertEqual(bgen.count()['nVariants'], 199)

        gen = hc.import_gen(test_resources + '/example.gen',
                                 sample_file = test_resources + '/example.sample')
        self.assertEqual(gen.count()['nVariants'], 199)

        vcf = hc.import_vcf(test_resources + '/sample2.vcf').split_multi()

        vcf.export_plink('/tmp/sample_plink')

        bfile = '/tmp/sample_plink'
        plink = hc.import_plink(
            bfile + '.bed', bfile + '.bim', bfile + '.fam')
        self.assertEqual(vcf.count(genotypes = True), plink.count(genotypes = True))
        
        vcf.write('/tmp/sample.vds', overwrite = True)
        vds = hc.read('/tmp/sample.vds')

        self.assertTrue(vcf.same(vds))

        bn = hc.balding_nichols_model(3, 10, 100, 8)
        bn_count = bn.count()
        self.assertEqual(bn_count['nSamples'], 10)
        self.assertEqual(bn_count['nVariants'], 100)

    def test_dataset(self):
        test_resources = 'src/test/resources'
        
        sample = (hc.import_vcf(test_resources + '/sample.vcf')
                  .cache())
        sample_split = sample.split_multi()
        
        sample2 = (hc.import_vcf(test_resources + '/sample2.vcf')
                   .persist())
        sample2_split = sample2.split_multi()
        
        sample2.aggregate_intervals(test_resources + '/annotinterall.interval_list',
                                    'N = variants.count()',
                                    '/tmp/annotinter.tsv')
        
        (sample2.annotate_global_expr_by_variant('global.nVariants = variants.count()')
         .show_globals())
        
        (sample2.annotate_global_expr_by_sample('global.nSamples = samples.count()')
         .show_globals())
        
        (sample2.annotate_global_list(test_resources + '/global_list.txt', 'global.genes',
                                      as_set = True)
         .show_globals())
        
        (sample2.annotate_global_table(test_resources + '/global_table.tsv',
                                       'global.genes')
         .show_globals())
        
        (sample2.annotate_samples_expr('sa.nCalled = gs.filter(g => g.isCalled).count()')
         .export_samples('/tmp/sa.tsv', 's = s, nCalled = sa.nCalled'))
        
        sample2.annotate_samples_list(test_resources + '/sample2.sample_list',
                                      'sa.listed')
        
        sample2_annot = sample2.annotate_samples_table(
            test_resources + '/sampleAnnotations.tsv',
            'Sample',
            code = 'sa.isCase = table.Status == "CASE", sa.qPhen = table.qPhen')
        
        sample2.annotate_samples_vds(sample2_annot,
                                     code = 'sa.isCase = vds.isCase')

        (sample.annotate_variants_bed(test_resources + '/example1.bed',
                                      root = 'va.bed')
         .filter_variants_expr('va.bed')
         .count())
        
        (sample2.annotate_variants_expr('va.nCalled = gs.filter(g => g.isCalled).count()')
         .count())
        
        (sample2.annotate_variants_intervals(test_resources + '/annotinterall.interval_list',
                                             'va.included',
                                             all = True)
         .count())
        
        (sample2.annotate_variants_loci(test_resources + '/sample2_loci.tsv',
                                        'Locus(chr, pos.toInt)',
                                        'va.locus_annot')
         .count())
        
        (sample.annotate_variants_table(test_resources + '/variantAnnotations.tsv',
                                        'Variant(Chromosome, Position.toInt, Ref, Alt)',
                                        root = 'va.table')
         .count())

        (sample2.annotate_variants_vds(sample2, code = 'va.good = va.info.AF == vds.info.AF')
         .count())
        
        (concordance1, concordance2) = (sample2_split.concordance(sample2_split))
        concordance1.write('/tmp/foo.vds', overwrite = True)
        concordance2.write('/tmp/foo.vds', overwrite = True)

        downsampled = sample2.downsample_variants(20)
        downsampled.export_variants('/tmp/sample2_loci.tsv', 'chr = v.contig, pos = v.start')
        downsampled.export_variants('/tmp/sample2_variants.tsv', 'v')
        
        (sample2.filter_samples_list(test_resources + '/sample2.sample_list')
         .count()['nSamples'] == 56)
        
        sample2_split.export_gen('/tmp/sample2.gen')

        (sample2.filter_genotypes('g.isHet && g.gq > 20')
         .export_genotypes('/tmp/sample2_genotypes.tsv', 'v, s, g.nNonRefAlleles'))
        
        sample2_split.export_plink('/tmp/sample2')
        
        sample2.export_vcf('/tmp/sample2.vcf.bgz')

        sample2.filter_multi().count()
        
        self.assertEqual(sample2.filter_samples_all().count()['nSamples'], 0)

        self.assertEqual(sample2.filter_variants_all().count()['nVariants'], 0)
        
        sample2_dedup = (hc.import_vcf([test_resources + '/sample2.vcf',
                                             test_resources + '/sample2.vcf'])
                         .deduplicate())
        self.assertEqual(sample2_dedup.count()['nVariants'], 735)
        
        (sample2.filter_samples_expr('pcoin(0.5)')
         .export_samples('/tmp/sample2.sample_list', 's'))

        (sample2.filter_variants_intervals(test_resources + '/annotinterall.interval_list')
         .count())
        
        self.assertEqual(sample2.filter_variants_list(
            test_resources + '/sample2_variants.tsv')
                         .count()['nVariants'], 21)
        
        sample2.grm('gcta-grm-bin', '/tmp/sample2.grm')

        sample2.hardcalls().count()

        sample2_split.ibd('/tmp/sample2.ibd')
        
        sample2.impute_sex().print_schema()

        self.assertEqual(sample2.join(sample2.rename_samples(test_resources + '/sample2_rename.tsv'))
                         .count()['nSamples'], 200)

        linreg = (hc.import_vcf(test_resources + '/regressionLinear.vcf')
                  .split_multi()
                  .annotate_samples_table(test_resources + '/regressionLinear.cov',
                                          'Sample',
                                          root = 'sa.cov',
                                          config=TextTableConfig(types='Cov1: Double, Cov2: Double'))
                  .annotate_samples_table(test_resources + '/regressionLinear.pheno',
                                          'Sample',
                                          code = 'sa.pheno.Pheno = table.Pheno',
                                          config=TextTableConfig(types='Pheno: Double', missing='0'))
                  .annotate_samples_table(test_resources + '/regressionLogisticBoolean.pheno',
                                          'Sample',
                                          code = 'sa.pheno.isCase = table.isCase',
                                          config=TextTableConfig(types='isCase: Boolean', missing='0')))

        (linreg.linreg('sa.pheno.Pheno', covariates = 'sa.cov.Cov1, sa.cov.Cov2 + 1 - 1')
         .count())

        (linreg.logreg('wald', 'sa.pheno.isCase', covariates = 'sa.cov.Cov1, sa.cov.Cov2 + 1 - 1')
         .count())

        sample_split.mendel_errors('/tmp/sample.mendel', test_resources + '/sample.fam')

        sample_split.pca('sa.scores')

        self.assertTrue(
            (sample2.repartition(16, shuffle = False)
             .same(sample2)))

        sample2.sparkinfo()

        sample_split.tdt(test_resources + '/sample.fam')

        sample2.typecheck()

        sample2_split.variant_qc().print_schema()

        sample2.export_variants('/tmp/variants.tsv', 'v = v, va = va')
        self.assertTrue((sample2.variants_keytable()
                         .annotate('va = json(va)'))
                        .same(hc.import_keytable('/tmp/variants.tsv', ['v'], config = TextTableConfig(impute = True))))

        sample2.export_samples('/tmp/samples.tsv', 's = s, sa = sa')
        self.assertTrue((sample2.samples_keytable()
                         .annotate('s = s.id, sa = json(sa)'))
                        .same(hc.import_keytable('/tmp/samples.tsv', ['s'], config = TextTableConfig(impute = True))))

        cols = ['v = v, info = va.info']
        for s in sample2.sample_ids():
            cols.append('{s}.gt = va.G["{s}"].gt, {s}.gq = va.G["{s}"].gq'.format(s = s))

        (sample2
         .annotate_variants_expr('va.G = index(gs.map(g => { s: s.id, gt: g.gt, gq: g.gq }).collect(), s)')
         .export_variants('/tmp/sample_kt.tsv', ','.join(cols)))

        ((sample2
          .make_keytable('v = v, info = va.info', 'gt = g.gt, gq = g.gq', ['v']))
         .same(hc.import_keytable('/tmp/sample_kt.tsv', ['v'])))

        sample_split.annotate_variants_expr("va.nHet = gs.filter(g => g.isHet).count()")
        
        sample_split.aggregate_by_key("Variant = v", "nHet = g.map(g => g.isHet.toInt).sum().toLong")
        
        sample2.make_keytable('v = v, info = va.info', 'gt = g.gt', ['v'])

        sample.num_partitions()
        sample.file_version()
        sample.sample_ids()[:5]

        self.assertFalse(sample.was_split())
        self.assertTrue(sample_split.was_split())

        self.assertFalse(sample.is_dosage())

        self.assertEqual(sample.num_samples(), 100)
        self.assertEqual(sample.num_variants(), 346)

        sample_split.ld_prune()

    def test_keytable(self):
        test_resources = 'src/test/resources'
        
        # Import
        # columns: Sample Status qPhen
        kt = hc.import_keytable(test_resources + '/sampleAnnotations.tsv', 'Sample', config = TextTableConfig(impute = True))
        kt2 = hc.import_keytable(test_resources + '/sampleAnnotations2.tsv', 'Sample', config = TextTableConfig(impute = True))

        # Variables
        self.assertEqual(kt.nfields(), 3)
        self.assertEqual(kt.key_names()[0], "Sample")
        self.assertEqual(kt.field_names()[2], "qPhen")
        self.assertEqual(kt.nrows(), 100)
        kt.schema()

        # Export
        kt.export('/tmp/testExportKT.tsv')

        # Filter, Same
        ktcase = kt.filter('Status == "CASE"', True)
        ktcase2 = kt.filter('Status == "CTRL"', False)
        self.assertTrue(ktcase.same(ktcase2))

        # Annotate
        (kt.annotate('X = Status')
         .nrows())

        # Join
        kt.join(kt2, 'left').nrows()

        # AggregateByKey
        (kt.aggregate_by_key("Status = Status", "Sum = qPhen.sum()")
         .nrows())

        # Forall, Exists
        self.assertFalse(kt.forall('Status == "CASE"'))
        self.assertTrue(kt.exists('Status == "CASE"'))


        kt.rename({"Sample": "ID"})
        kt.rename(["Field1", "Field2", "Field3"])
        kt.rename([name + "_a" for name in kt.field_names()])

        kt.select(["Sample"])
        kt.select(["Sample", "Status"])

        kt.key_by(['Sample', 'Status'])
        kt.key_by([])

        kt.flatten()
        kt.expand_types()

        kt.to_dataframe()
