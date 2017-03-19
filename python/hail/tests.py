"""
Unit tests for Hail.
"""
from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import HailContext, TextTableConfig
from hail.representation import *
from hail.type import *
from hail.java import *

hc = None


def setUpModule():
    global hc
    hc = HailContext()  # master = 'local[2]')


def tearDownModule():
    global hc
    hc.stop()
    hc = None


class ContextTests(unittest.TestCase):
    def test_context(self):
        test_resources = 'src/test/resources'

        hc.grep('Mom1', test_resources + '/mendel.fam')

        # index
        hc.index_bgen(test_resources + '/example.v11.bgen')

        bgen = hc.import_bgen(test_resources + '/example.v11.bgen',
                              sample_file=test_resources + '/example.sample')
        self.assertEqual(bgen.count()['nVariants'], 199)

        gen = hc.import_gen(test_resources + '/example.gen',
                            sample_file=test_resources + '/example.sample')
        self.assertEqual(gen.count()['nVariants'], 199)

        vcf = hc.import_vcf(test_resources + '/sample2.vcf').split_multi()

        vcf.export_plink('/tmp/sample_plink')

        bfile = '/tmp/sample_plink'
        plink = hc.import_plink(
            bfile + '.bed', bfile + '.bim', bfile + '.fam')
        self.assertEqual(vcf.count(genotypes=True), plink.count(genotypes=True))

        vcf.write('/tmp/sample.vds', overwrite=True)
        vds = hc.read('/tmp/sample.vds')
        self.assertTrue(vcf.same(vds))

        vcf.write('/tmp/sample.pq.vds', parquet_genotypes=True, overwrite=True)
        self.assertTrue(vcf.same(hc.read('/tmp/sample.pq.vds')))

        bn = hc.balding_nichols_model(3, 10, 100, 8)
        bn_count = bn.count()
        self.assertEqual(bn_count['nSamples'], 10)
        self.assertEqual(bn_count['nVariants'], 100)

        self.assertEqual(hc.eval_expr_typed('[1, 2, 3].map(x => x * 2)'), ([2, 4, 6], TArray(TInt())))
        self.assertEqual(hc.eval_expr('[1, 2, 3].map(x => x * 2)'), [2, 4, 6])

    def test_dataset(self):
        test_resources = 'src/test/resources'

        sample = hc.import_vcf(test_resources + '/sample.vcf')
        sample.cache()

        sample_split = sample.split_multi()

        sample2 = hc.import_vcf(test_resources + '/sample2.vcf')
        sample2.persist()

        sample2_split = sample2.split_multi()

        sample2.aggregate_intervals(test_resources + '/annotinterall.interval_list',
                                    'N = variants.count()',
                                    '/tmp/annotinter.tsv')

        sample2.query_variants(['variants.count()'])

        sample2.query_samples(['samples.count()'])

        (sample2.annotate_global_list(test_resources + '/global_list.txt', 'global.genes',
                                      as_set=True)
         .globals)

        (sample2.annotate_global_table(test_resources + '/global_table.tsv',
                                       'global.genes')
         .globals)

        (sample2.annotate_samples_expr('sa.nCalled = gs.filter(g => g.isCalled()).count()')
         .export_samples('/tmp/sa.tsv', 's = s, nCalled = sa.nCalled'))

        sample2.annotate_samples_list(test_resources + '/sample2.sample_list',
                                      'sa.listed')

        sample2.annotate_global_expr('global.foo = 5')
        sample2.annotate_global_expr(['global.foo = 5', 'global.bar = 6'])

        sample2_annot = sample2.annotate_samples_table(
            test_resources + '/sampleAnnotations.tsv',
            'Sample',
            code='sa.isCase = table.Status == "CASE", sa.qPhen = table.qPhen')

        sample2.annotate_samples_vds(sample2_annot,
                                     code='sa.isCase = vds.isCase')

        (sample.annotate_variants_bed(test_resources + '/example1.bed',
                                      root='va.bed')
         .filter_variants_expr('va.bed')
         .count())

        sample.annotate_alleles_expr('va.gs = gs.callStats(g => v)').count()
        sample.annotate_alleles_expr(['va.gs = gs.callStats(g => v)', 'va.foo = 5']).count()

        (sample2.annotate_variants_expr('va.nCalled = gs.filter(g => g.isCalled()).count()')
         .count())

        (sample2.annotate_variants_intervals(test_resources + '/annotinterall.interval_list',
                                             'va.included',
                                             all=True)
         .count())

        (sample2.annotate_variants_loci(test_resources + '/sample2_loci.tsv',
                                        'Locus(chr, pos.toInt())',
                                        'va.locus_annot')
         .count())

        (sample.annotate_variants_table(test_resources + '/variantAnnotations.tsv',
                                        'Variant(Chromosome, Position.toInt(), Ref, Alt)',
                                        root='va.table')
         .count())

        (sample2.annotate_variants_vds(sample2, code='va.good = va.info.AF == vds.info.AF')
         .count())

        glob, concordance1, concordance2 = (sample2_split.concordance(sample2_split))
        print(glob[1][4])
        print(glob[4][0])
        print(glob[:][3])
        concordance1.write('/tmp/foo.vds', overwrite=True)
        concordance2.write('/tmp/foo.vds', overwrite=True)

        downsampled = sample2.downsample_variants(20)
        downsampled.export_variants('/tmp/sample2_loci.tsv', 'chr = v.contig, pos = v.start')
        downsampled.export_variants('/tmp/sample2_variants.tsv', 'v')

        (sample2.filter_samples_list(test_resources + '/sample2.sample_list')
         .count()['nSamples'] == 56)

        sample2_split.export_gen('/tmp/sample2.gen')

        (sample2.filter_genotypes('g.isHet() && g.gq > 20')
         .export_genotypes('/tmp/sample2_genotypes.tsv', 'v, s, g.nNonRefAlleles()'))

        sample2_split.export_plink('/tmp/sample2')

        sample2.export_vcf('/tmp/sample2.vcf.bgz')

        sample2.filter_multi().count()

        self.assertEqual(sample2.drop_samples().count()['nSamples'], 0)

        self.assertEqual(sample2.drop_variants().count()['nVariants'], 0)

        sample2_dedup = (hc.import_vcf([test_resources + '/sample2.vcf',
                                        test_resources + '/sample2.vcf'])
                         .deduplicate())
        self.assertEqual(sample2_dedup.count()['nVariants'], 735)

        (sample2.filter_samples_expr('pcoin(0.5)')
         .export_samples('/tmp/sample2.sample_list', 's'))

        (sample2.filter_variants_expr('pcoin(0.5)')
         .export_variants('/tmp/sample2.variant_list', 'v'))

        (sample2.filter_variants_intervals(IntervalTree.read(test_resources + '/annotinterall.interval_list'))
         .count())

        sample2.filter_variants_intervals(Interval.parse('1:100-end')).count()
        sample2.filter_variants_intervals(IntervalTree.parse_all(['1:100-end', '3-22'])).count()
        sample2.filter_variants_intervals(IntervalTree([Interval.parse('1:100-end')])).count()

        (sample2.filter_variants_intervals(IntervalTree.read(test_resources + '/annotinterall.interval_list'))
         .count())

        self.assertEqual(sample2.filter_variants_list(
            test_resources + '/sample2_variants.tsv')
                         .count()['nVariants'], 21)

        sample2.split_multi().grm('/tmp/sample2.grm', 'gcta-grm-bin')

        sample2.hardcalls().count()

        sample2_split.ibd(min=0.2, max=0.6)

        sample2.split_multi().impute_sex().variant_schema

        self.assertTrue(isinstance(sample2.genotype_schema, TGenotype))

        m2 = {r._0: r._1 for r in hc.import_keytable(test_resources + '/sample2_rename.tsv',
                                                     config=TextTableConfig(noheader=True))
              .collect()}
        self.assertEqual(sample2.join(sample2.rename_samples(m2))
                         .count()['nSamples'], 200)

        regression = (hc.import_vcf(test_resources + '/regressionLinear.vcf')
                  .split_multi()
                  .annotate_samples_table(test_resources + '/regressionLinear.cov',
                                          'Sample',
                                          root='sa.cov',
                                          config=TextTableConfig(types='Cov1: Double, Cov2: Double'))
                  .annotate_samples_table(test_resources + '/regressionLinear.pheno',
                                          'Sample',
                                          code='sa.pheno.Pheno = table.Pheno',
                                          config=TextTableConfig(types='Pheno: Double', missing='0'))
                  .annotate_samples_table(test_resources + '/regressionLogisticBoolean.pheno',
                                          'Sample',
                                          code='sa.pheno.isCase = table.isCase',
                                          config=TextTableConfig(types='isCase: Boolean', missing='0')))

        (regression.linreg('sa.pheno.Pheno', covariates=['sa.cov.Cov1', 'sa.cov.Cov2 + 1 - 1'])
         .count())

        (regression.logreg('wald', 'sa.pheno.isCase', covariates=['sa.cov.Cov1', 'sa.cov.Cov2 + 1 - 1'])
         .count())

        vds_assoc = (regression
                     .annotate_samples_expr(
            'sa.culprit = gs.filter(g => v == Variant("1", 1, "C", "T")).map(g => g.gt).collect()[0]')
                     .annotate_samples_expr('sa.pheno.PhenoLMM = (1 + 0.1 * sa.cov.Cov1 * sa.cov.Cov2) * sa.culprit'))

        vds_kinship = vds_assoc.filter_variants_expr('v.start < 4')

        vds_assoc = vds_assoc.lmmreg(vds_kinship, 'sa.pheno.PhenoLMM', ['sa.cov.Cov1', 'sa.cov.Cov2'])

        vds_assoc.export_variants('/tmp/lmmreg.tsv', 'Variant = v, va.lmmreg.*')

        sample_split.mendel_errors('/tmp/sample.mendel', test_resources + '/sample.fam')

        sample_split.pca('sa.scores')

        self.assertTrue(
            (sample2.repartition(16, shuffle=False)
             .same(sample2)))

        print(sample2.storage_level())

        sample_split.tdt(test_resources + '/sample.fam')

        sample2._typecheck()

        sample2_split.variant_qc().variant_schema

        sample2.export_variants('/tmp/variants.tsv', 'v = v, va = va')
        self.assertTrue((sample2.variants_keytable()
                         .annotate('va = json(va)'))
                        .same(hc.import_keytable('/tmp/variants.tsv', config=TextTableConfig(impute=True)).key_by('v')))

        sample2.export_samples('/tmp/samples.tsv', 's = s, sa = sa')
        self.assertTrue((sample2.samples_keytable()
                         .annotate('s = s.id, sa = json(sa)'))
                        .same(hc.import_keytable('/tmp/samples.tsv', config=TextTableConfig(impute=True)).key_by('s')))

        cols = ['v = v, info = va.info']
        for s in sample2.sample_ids:
            cols.append('{s}.gt = va.G["{s}"].gt, {s}.gq = va.G["{s}"].gq'.format(s=s))

        (sample2
         .annotate_variants_expr('va.G = index(gs.map(g => { s: s.id, gt: g.gt, gq: g.gq }).collect(), s)')
         .export_variants('/tmp/sample_kt.tsv', ','.join(cols)))

        ((sample2
          .make_keytable('v = v, info = va.info', 'gt = g.gt, gq = g.gq', ['v']))
         .same(hc.import_keytable('/tmp/sample_kt.tsv').key_by('v')))

        sample_split.annotate_variants_expr("va.nHet = gs.filter(g => g.isHet()).count()")

        sample_split.aggregate_by_key("Variant = v", "nHet = g.map(g => g.isHet().toInt()).sum().toLong()")
        sample_split.aggregate_by_key(["Variant = v"], ["nHet = g.map(g => g.isHet().toInt()).sum().toLong()"])

        sample2.make_keytable('v = v, info = va.info', 'gt = g.gt', ['v'])

        sample.num_partitions()
        sample.file_version()
        sample.sample_ids[:5]

        self.assertFalse(sample.was_split())
        self.assertTrue(sample_split.was_split())

        self.assertFalse(sample.is_dosage())

        self.assertEqual(sample.num_samples, 100)
        self.assertEqual(sample.count_variants(), 346)

        sample2.filter_alleles('pcoin(0.5)')

        sample2.annotate_variants_keytable(sample2.variants_keytable(), "va.foo = table.va")

        kt = (sample2.variants_keytable()
              .annotate("v2 = v")
              .key_by(["v", "v2"]))
        sample2.annotate_variants_keytable(kt, "va.foo = table.va", ["v", "v"])
        
        variants_py = (sample
                       .annotate_variants_expr('va.hets = gs.filter(g => g.isHet).collect()')
                       .variants_keytable()
                       .collect())

    def test_keytable(self):
        test_resources = 'src/test/resources'

        # Import
        # columns: Sample Status qPhen
        kt = hc.import_keytable(test_resources + '/sampleAnnotations.tsv',
                                config=TextTableConfig(impute=True)).key_by('Sample')
        kt2 = hc.import_keytable(test_resources + '/sampleAnnotations2.tsv',
                                 config=TextTableConfig(impute=True)).key_by('Sample')

        # Variables
        self.assertEqual(kt.num_columns, 3)
        self.assertEqual(kt.key_names[0], "Sample")
        self.assertEqual(kt.column_names[2], "qPhen")
        self.assertEqual(kt.count_rows(), 100)
        kt.schema

        # Export
        kt.export('/tmp/testExportKT.tsv')

        # Filter, Same
        ktcase = kt.filter('Status == "CASE"', True)
        ktcase2 = kt.filter('Status == "CTRL"', False)
        self.assertTrue(ktcase.same(ktcase2))

        # Annotate
        (kt.annotate('X = Status')
         .count_rows())

        # Join
        kt.join(kt2, 'left').count_rows()

        # AggregateByKey
        (kt.aggregate_by_key("Status = Status", "Sum = qPhen.sum()")
         .count_rows())

        # Forall, Exists
        self.assertFalse(kt.forall('Status == "CASE"'))
        self.assertTrue(kt.exists('Status == "CASE"'))

        kt.rename({"Sample": "ID"})
        kt.rename(["Field1", "Field2", "Field3"])
        kt.rename([name + "_a" for name in kt.column_names])

        kt.select(["Sample"])
        kt.select(["Sample", "Status"])

        kt.key_by(['Sample', 'Status'])
        kt.key_by([])

        kt.flatten()
        kt.expand_types()

        kt.to_dataframe()

        kt.annotate("newField = [0, 1, 2]").explode(["newField"])

        sample = hc.import_vcf(test_resources + '/sample.vcf')
        sample_variants = (sample.variants_keytable()
                           .annotate('v = str(v), va.filters = va.filters.toArray()')
                           .flatten())

        sample_variants2 = hc.dataframe_to_keytable(
            sample_variants.to_dataframe(), ['v'])
        self.assertTrue(sample_variants.same(sample_variants2))

        # cseed: calculated by hand using sort -n -k 3,3 and inspection
        self.assertTrue(kt.filter('qPhen < 10000').count_rows() == 23)

        kt.write('/tmp/sampleAnnotations.kt', overwrite=True)
        kt3 = hc.read_keytable('/tmp/sampleAnnotations.kt')
        self.assertTrue(kt.same(kt3))

    def test_representation(self):
        v = Variant.parse('1:100:A:T')

        self.assertEqual(v, Variant('1', 100, 'A', 'T'))
        self.assertEqual(v, Variant(1, 100, 'A', ['T']))

        v2 = Variant.parse('1:100:A:T,C')

        self.assertEqual(v2, Variant('1', 100, 'A', ['T', 'C']))

        l = Locus.parse('1:100')

        self.assertEqual(l, Locus('1', 100))
        self.assertEqual(l, Locus(1, 100))

        self.assertEqual(l, v.locus())

        self.assertEqual(v2.num_alt_alleles(), 2)
        self.assertFalse(v2.is_biallelic())
        self.assertTrue(v.is_biallelic())
        self.assertEqual(v.alt_allele(), AltAllele('A', 'T'))
        self.assertEqual(v.allele(0), 'A')
        self.assertEqual(v.allele(1), 'T')
        self.assertEqual(v2.num_alleles(), 3)
        self.assertEqual(v.alt(), 'T')
        self.assertEqual(v2.alt_alleles[0], AltAllele('A', 'T'))
        self.assertEqual(v2.alt_alleles[1], AltAllele('A', 'C'))

        self.assertTrue(v2.is_autosomal_or_pseudoautosomal())
        self.assertTrue(v2.is_autosomal())
        self.assertFalse(v2.is_mitochondrial())
        self.assertFalse(v2.in_X_PAR())
        self.assertFalse(v2.in_Y_PAR())
        self.assertFalse(v2.in_X_non_PAR())
        self.assertFalse(v2.in_Y_non_PAR())

        aa1 = AltAllele('A', 'T')
        aa2 = AltAllele('A', 'AAA')
        aa3 = AltAllele('TTTT', 'T')
        aa4 = AltAllele('AT', 'TC')
        aa5 = AltAllele('AAAT', 'AAAA')

        self.assertEqual(aa1.num_mismatch(), 1)
        self.assertEqual(aa5.num_mismatch(), 1)
        self.assertEqual(aa4.num_mismatch(), 2)

        c1, c2 = aa5.stripped_snp()

        self.assertEqual(c1, 'T')
        self.assertEqual(c2, 'A')
        self.assertTrue(aa1.is_SNP())
        self.assertTrue(aa5.is_SNP())
        self.assertTrue(aa4.is_MNP())
        self.assertTrue(aa2.is_insertion())
        self.assertTrue(aa3.is_deletion())
        self.assertTrue(aa3.is_indel())
        self.assertTrue(aa1.is_transversion())

        interval = Interval.parse('1:100-110')

        self.assertEqual(interval, Interval.parse('1:100-1:110'))
        self.assertEqual(interval, Interval(Locus("1", 100), Locus("1", 110)))
        self.assertTrue(interval.contains(Locus("1", 100)))
        self.assertTrue(interval.contains(Locus("1", 109)))
        self.assertFalse(interval.contains(Locus("1", 110)))

        interval2 = Interval.parse("1:109-200")
        interval3 = Interval.parse("1:110-200")
        interval4 = Interval.parse("1:90-101")
        interval5 = Interval.parse("1:90-100")

        self.assertTrue(interval.overlaps(interval2))
        self.assertTrue(interval.overlaps(interval4))
        self.assertFalse(interval.overlaps(interval3))
        self.assertFalse(interval.overlaps(interval5))

        g = Genotype(1, [12, 10], 25, 40, [40, 0, 99])
        g2 = Genotype(4)

        self.assertEqual(g.gt, 1)
        self.assertEqual(g.ad, [12, 10])
        self.assertEqual(g.dp, 25)
        self.assertEqual(g.gq, 40)
        self.assertEqual(g.pl, [40, 0, 99])
        self.assertEqual(g2.gt, 4)
        self.assertEqual(g2.ad, None)
        self.assertEqual(g2.dp, None)
        self.assertEqual(g2.gq, None)
        self.assertEqual(g2.pl, None)

        self.assertEqual(g.od(), 3)
        self.assertFalse(g.is_hom_ref())
        self.assertFalse(g2.is_hom_ref())
        self.assertTrue(g.is_het())
        self.assertTrue(g2.is_het())
        self.assertTrue(g.is_het_ref())
        self.assertTrue(g2.is_het_non_ref())
        self.assertTrue(g.is_called_non_ref())
        self.assertTrue(g2.is_called_non_ref())
        self.assertFalse(g.is_hom_var())
        self.assertFalse(g2.is_hom_var())
        self.assertFalse(g.is_not_called())
        self.assertFalse(g2.is_not_called())
        self.assertTrue(g.is_called())
        self.assertTrue(g2.is_called())
        self.assertEqual(g.num_alt_alleles(), 1)
        self.assertEqual(g2.num_alt_alleles(), 2)

        hom_ref = Genotype(0)
        het = Genotype(1)
        hom_var = Genotype(2)

        num_alleles = 2
        hom_ref.one_hot_alleles(num_alleles) == [2, 0]
        het.one_hot_alleles(num_alleles) == [1, 1]
        hom_var.one_hot_alleles(num_alleles) == [0, 2]

        num_genotypes = 3
        hom_ref.one_hot_genotype(num_genotypes) == [1, 0, 0]
        het.one_hot_genotype(num_genotypes) == [0, 1, 0]
        hom_var.one_hot_genotype(num_genotypes) == [0, 0, 1]

        self.assertTrue(0 < g.p_ab() < 1)
        self.assertEqual(g2.p_ab(), None)

        missing_gt = Genotype(gt=None)
        self.assertEqual(missing_gt.gt, None)
        self.assertEqual(missing_gt.one_hot_alleles(2), None)
        self.assertEqual(missing_gt.one_hot_genotype(4), None)

        self.assertEqual(g.fraction_reads_ref(), 12.0 / (10 + 12))

    def test_types(self):
        self.assertEqual(TInt(), TInt())
        self.assertEqual(TDouble(), TDouble())
        self.assertEqual(TArray(TDouble()), TArray(TDouble()))
        self.assertFalse(TArray(TDouble()) == TArray(TFloat()))
        self.assertFalse(TSet(TDouble()) == TArray(TFloat()))
        self.assertEqual(TSet(TDouble()), TSet(TDouble()))
        self.assertEqual(TDict(TString(), TArray(TInt())), TDict(TString(), TArray(TInt())))

        some_random_types = [
            TInt(),
            TString(),
            TDouble(),
            TFloat(),
            TBoolean(),
            TArray(TString()),
            TSet(TArray(TSet(TBoolean()))),
            TDict(TString(), TInt()),
            TVariant(),
            TLocus(),
            TGenotype(),
            TAltAllele(),
            TInterval(),
            TSet(TInterval()),
            TStruct(['a', 'b', 'c'], [TInt(), TInt(), TArray(TString())]),
            TStruct(['a', 'bb', 'c'], [TDouble(), TInt(), TBoolean()]),
            TStruct(['a', 'b'], [TInt(), TInt()])]

        #  copy and reinitialize to check that two initializations produce equality (not reference equality)
        some_random_types_cp = [
            TInt(),
            TString(),
            TDouble(),
            TFloat(),
            TBoolean(),
            TArray(TString()),
            TSet(TArray(TSet(TBoolean()))),
            TDict(TString(), TInt()),
            TVariant(),
            TLocus(),
            TGenotype(),
            TAltAllele(),
            TInterval(),
            TSet(TInterval()),
            TStruct(['a', 'b', 'c'], [TInt(), TInt(), TArray(TString())]),
            TStruct(['a', 'bb', 'c'], [TDouble(), TInt(), TBoolean()]),
            TStruct(['a', 'b'], [TInt(), TInt()])]

        for i in range(len(some_random_types)):
            for j in range(len(some_random_types)):
                if (i == j):
                    self.assertEqual(some_random_types[i], some_random_types_cp[j])
                else:
                    self.assertNotEqual(some_random_types[i], some_random_types_cp[j])

    def test_query(self):
        vds = hc.import_vcf('src/test/resources/sample.vcf').split_multi().sample_qc()

        self.assertEqual(vds.sample_ids, vds.query_samples(['samples.collect()'])[0])

    def test_annotate_global(self):
        vds = hc.import_vcf('src/test/resources/sample.vcf')

        path = 'global.annotation'

        i1 = 5
        i2 = None
        itype = TInt()
        self.assertEqual(vds.annotate_global_py(path, i1, itype).globals.annotation, i1)
        self.assertEqual(vds.annotate_global_py(path, i2, itype).globals.annotation, i2)

        l1 = 5L
        l2 = None
        ltype = TLong()
        self.assertEqual(vds.annotate_global_py(path, l1, ltype).globals.annotation, l1)
        self.assertEqual(vds.annotate_global_py(path, l2, ltype).globals.annotation, l2)

        # FIXME add these back in when we update py4j, or rip out TFloat altogether.
        # f1 = float(5)
        # f2 = None
        # ftype = TFloat()
        # self.assertEqual(vds.annotate_global_py(path, f1, ftype).globals.annotation, f1)
        # self.assertEqual(vds.annotate_global_py(path, f2, ftype).globals.annotation, f2)

        d1 = float(5)
        d2 = None
        dtype = TDouble()
        self.assertEqual(vds.annotate_global_py(path, d1, dtype).globals.annotation, d1)
        self.assertEqual(vds.annotate_global_py(path, d2, dtype).globals.annotation, d2)

        b1 = True
        b2 = None
        btype = TBoolean()
        self.assertEqual(vds.annotate_global_py(path, b1, btype).globals.annotation, b1)
        self.assertEqual(vds.annotate_global_py(path, b2, btype).globals.annotation, b2)

        arr1 = [1, 2, 3, 4]
        arr2 = [1, 2, None, 4]
        arr3 = None
        arr4 = []
        arrtype = TArray(TInt())
        self.assertEqual(vds.annotate_global_py(path, arr1, arrtype).globals.annotation, arr1)
        self.assertEqual(vds.annotate_global_py(path, arr2, arrtype).globals.annotation, arr2)
        self.assertEqual(vds.annotate_global_py(path, arr3, arrtype).globals.annotation, arr3)
        self.assertEqual(vds.annotate_global_py(path, arr4, arrtype).globals.annotation, arr4)

        set1 = {1, 2, 3, 4}
        set2 = {1, 2, None, 4}
        set3 = None
        set4 = set()
        settype = TSet(TInt())
        self.assertEqual(vds.annotate_global_py(path, set1, settype).globals.annotation, set1)
        self.assertEqual(vds.annotate_global_py(path, set2, settype).globals.annotation, set2)
        self.assertEqual(vds.annotate_global_py(path, set3, settype).globals.annotation, set3)
        self.assertEqual(vds.annotate_global_py(path, set4, settype).globals.annotation, set4)

        dict1 = {'a': 'foo', 'b': 'bar'}
        dict2 = {'a': None, 'b': 'bar'}
        dict3 = None
        dict4 = dict()
        dicttype = TDict(TString(), TString())
        self.assertEqual(vds.annotate_global_py(path, dict1, dicttype).globals.annotation, dict1)
        self.assertEqual(vds.annotate_global_py(path, dict2, dicttype).globals.annotation, dict2)
        self.assertEqual(vds.annotate_global_py(path, dict3, dicttype).globals.annotation, dict3)
        self.assertEqual(vds.annotate_global_py(path, dict4, dicttype).globals.annotation, dict4)

        map5 = {Locus("1", 100): 5, Locus("5", 205): 100}
        map6 = None
        map7 = dict()
        map8 = {Locus("1", 100): None, Locus("5", 205): 100}
        maptype2 = TDict(TLocus(), TInt())
        self.assertEqual(vds.annotate_global_py(path, map5, maptype2).globals.annotation, map5)
        self.assertEqual(vds.annotate_global_py(path, map6, maptype2).globals.annotation, map6)
        self.assertEqual(vds.annotate_global_py(path, map7, maptype2).globals.annotation, map7)
        self.assertEqual(vds.annotate_global_py(path, map8, maptype2).globals.annotation, map8)

        struct1 = Struct({'field1': 5, 'field2': 10, 'field3': [1, 2]})
        struct2 = Struct({'field1': 5, 'field2': None, 'field3': None})
        struct3 = None
        structtype = TStruct(['field1', 'field2', 'field3'], [TInt(), TInt(), TArray(TInt())])
        self.assertEqual(vds.annotate_global_py(path, struct1, structtype).globals.annotation, struct1)
        self.assertEqual(vds.annotate_global_py(path, struct2, structtype).globals.annotation, struct2)
        self.assertEqual(vds.annotate_global_py(path, struct3, structtype).globals.annotation, struct3)

        variant1 = Variant.parse('1:1:A:T,C')
        variant2 = None
        varianttype = TVariant()
        self.assertEqual(vds.annotate_global_py(path, variant1, varianttype).globals.annotation, variant1)
        self.assertEqual(vds.annotate_global_py(path, variant2, varianttype).globals.annotation, variant2)

        altallele1 = AltAllele('T', 'C')
        altallele2 = None
        altalleletype = TAltAllele()
        self.assertEqual(vds.annotate_global_py(path, altallele1, altalleletype).globals.annotation, altallele1)
        self.assertEqual(vds.annotate_global_py(path, altallele2, altalleletype).globals.annotation, altallele2)

        locus1 = Locus.parse('1:100')
        locus2 = None
        locustype = TLocus()
        self.assertEqual(vds.annotate_global_py(path, locus1, locustype).globals.annotation, locus1)
        self.assertEqual(vds.annotate_global_py(path, locus2, locustype).globals.annotation, locus2)

        interval1 = Interval.parse('1:1-100')
        interval2 = None
        intervaltype = TInterval()
        self.assertEqual(vds.annotate_global_py(path, interval1, intervaltype).globals.annotation, interval1)
        self.assertEqual(vds.annotate_global_py(path, interval2, intervaltype).globals.annotation, interval2)

        genotype1 = Genotype(1)
        genotype2 = None
        genotypetype = TGenotype()
        self.assertEqual(vds.annotate_global_py(path, genotype1, genotypetype).globals.annotation, genotype1)
        self.assertEqual(vds.annotate_global_py(path, genotype2, genotypetype).globals.annotation, genotype2)

    def test_concordance(self):
        bn1 = hc.balding_nichols_model(3, 1, 50, 1, seed=10)
        bn2 = hc.balding_nichols_model(3, 1, 50, 1, seed=50)

        glob, samples, variants = bn1.concordance(bn2)
        self.assertEqual(samples.sample_annotations[samples.sample_ids[0]].concordance, glob)
