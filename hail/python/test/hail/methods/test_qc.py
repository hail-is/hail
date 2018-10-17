import unittest

import hail as hl
import hail.expr.aggregators as agg
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def test_sample_qc(self):
        data = [
            {'v': '1:1:A:T', 's': '1', 'GT': hl.Call([0, 0]), 'GQ': 10, 'DP': 0},
            {'v': '1:2:A:T,C', 's': '1', 'GT': hl.Call([1]), 'GQ': 15, 'DP': 5},
            {'v': '1:3:A:G,C', 's': '1', 'GT': hl.Call([2, 2]), 'GQ': 10, 'DP': 4},
            {'v': '1:4:G:A', 's': '1', 'GT': hl.Call([0, 1]), 'GQ': None, 'DP': 5},
            {'v': '1:5:C:CG', 's': '1', 'GT': hl.Call([1, 1]), 'GQ': 20, 'DP': 3},
            {'v': '1:6:C:A', 's': '1', 'GT': None, 'GQ': 0, 'DP': None},
        ]

        ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, GT: call, GQ: int, DP: int}'))
        ht = ht.transmute(**hl.parse_variant(ht.v))
        mt = ht.to_matrix_table(['locus', 'alleles'], ['s'])
        mt = hl.sample_qc(mt, 'sqc')
        r = mt.cols().select('sqc').collect()

        self.assertAlmostEqual(r[0].sqc.gq_stats.mean, 11)
        self.assertAlmostEqual(r[0].sqc.gq_stats.stdev, 6.6332495807)
        self.assertAlmostEqual(r[0].sqc.gq_stats.min, 0)
        self.assertAlmostEqual(r[0].sqc.gq_stats.max, 20)
        self.assertAlmostEqual(r[0].sqc.dp_stats.mean, 3.399999999)
        self.assertAlmostEqual(r[0].sqc.dp_stats.stdev, 1.8547236990)
        self.assertAlmostEqual(r[0].sqc.dp_stats.min, 0)
        self.assertAlmostEqual(r[0].sqc.dp_stats.max, 5)
        self.assertAlmostEqual(r[0].sqc.call_rate, 0.8333333333)
        self.assertEqual(r[0].sqc.n_called, 5)
        self.assertEqual(r[0].sqc.n_not_called, 1)
        self.assertEqual(r[0].sqc.n_hom_ref, 1)
        self.assertEqual(r[0].sqc.n_het, 1)
        self.assertEqual(r[0].sqc.n_hom_var, 3)
        self.assertEqual(r[0].sqc.n_insertion, 2)
        self.assertEqual(r[0].sqc.n_deletion, 0)
        self.assertEqual(r[0].sqc.n_singleton, 3)
        self.assertEqual(r[0].sqc.n_transition, 1)
        self.assertEqual(r[0].sqc.n_transversion, 3)
        self.assertEqual(r[0].sqc.n_star, 0)
        self.assertEqual(r[0].sqc.n_non_ref, 4)
        self.assertAlmostEqual(r[0].sqc.r_ti_tv, 0.333333333)
        self.assertAlmostEqual(r[0].sqc.r_het_hom_var, 0.3333333333)
        self.assertAlmostEqual(r[0].sqc.r_insertion_deletion, None)

    def test_variant_qc(self):
        data = [
            {'v': '1:1:A:T', 's': '1', 'GT': hl.Call([0, 0]), 'GQ': 10, 'DP': 0},
            {'v': '1:1:A:T', 's': '2', 'GT': hl.Call([1, 1]), 'GQ': 10, 'DP': 5},
            {'v': '1:1:A:T', 's': '3', 'GT': hl.Call([0, 1]), 'GQ': 11, 'DP': 100},
            {'v': '1:1:A:T', 's': '4', 'GT': None, 'GQ': None, 'DP': 100},
            {'v': '1:2:A:T,C', 's': '1', 'GT': hl.Call([1, 2]), 'GQ': 10, 'DP': 5},
            {'v': '1:2:A:T,C', 's': '2', 'GT': hl.Call([2, 2]), 'GQ': 10, 'DP': 5},
            {'v': '1:2:A:T,C', 's': '3', 'GT': hl.Call([0, 1]), 'GQ': 10, 'DP': 5},
            {'v': '1:2:A:T,C', 's': '4', 'GT': hl.Call([1, 1]), 'GQ': 10, 'DP': 5},
        ]

        ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, GT: call, GQ: int, DP: int}'))
        ht = ht.transmute(**hl.parse_variant(ht.v))
        mt = ht.to_matrix_table(['locus', 'alleles'], ['s'])
        mt = hl.variant_qc(mt, 'vqc')
        r = mt.rows().collect()

        self.assertEqual(r[0].vqc.AF, [0.5, 0.5])
        self.assertEqual(r[0].vqc.AC, [3, 3])
        self.assertEqual(r[0].vqc.AN, 6)
        self.assertEqual(r[0].vqc.homozygote_count, [1, 1])
        self.assertEqual(r[0].vqc.n_called, 3)
        self.assertEqual(r[0].vqc.n_not_called, 1)
        self.assertEqual(r[0].vqc.call_rate, 0.75)
        self.assertEqual(r[0].vqc.n_het, 1)
        self.assertEqual(r[0].vqc.n_non_ref, 2)
        self.assertEqual(r[0].vqc.het_freq_hwe, 0.6)
        self.assertEqual(r[0].vqc.p_value_hwe, 0.7)
        self.assertEqual(r[0].vqc.dp_stats.min, 0)
        self.assertEqual(r[0].vqc.dp_stats.max, 100)
        self.assertEqual(r[0].vqc.dp_stats.mean, 51.25)
        self.assertAlmostEqual(r[0].vqc.dp_stats.stdev, 48.782040752719645)
        self.assertEqual(r[0].vqc.gq_stats.min, 10)
        self.assertEqual(r[0].vqc.gq_stats.max, 11)
        self.assertAlmostEqual(r[0].vqc.gq_stats.mean, 10.333333333333334)
        self.assertAlmostEqual(r[0].vqc.gq_stats.stdev, 0.47140452079103168)

        self.assertEqual(r[1].vqc.AF, [0.125, 0.5, 0.375])
        self.assertEqual(r[1].vqc.AC, [1, 4, 3])
        self.assertEqual(r[1].vqc.AN, 8)
        self.assertEqual(r[1].vqc.homozygote_count, [0, 1, 1])
        self.assertEqual(r[1].vqc.n_called, 4)
        self.assertEqual(r[1].vqc.n_not_called, 0)
        self.assertEqual(r[1].vqc.call_rate, 1.0)
        self.assertEqual(r[1].vqc.n_het, 2)
        self.assertEqual(r[1].vqc.n_non_ref, 4)
        self.assertEqual(r[1].vqc.p_value_hwe, None)
        self.assertEqual(r[1].vqc.het_freq_hwe, None)
        self.assertEqual(r[1].vqc.dp_stats.min, 5)
        self.assertEqual(r[1].vqc.dp_stats.max, 5)
        self.assertEqual(r[1].vqc.dp_stats.mean, 5)
        self.assertEqual(r[1].vqc.dp_stats.stdev, 0.0)
        self.assertEqual(r[1].vqc.gq_stats.min, 10)
        self.assertEqual(r[1].vqc.gq_stats.max, 10)
        self.assertEqual(r[1].vqc.gq_stats.mean, 10)
        self.assertEqual(r[1].vqc.gq_stats.stdev, 0)

    def test_concordance(self):
        dataset = get_dataset()
        glob_conc, cols_conc, rows_conc = hl.concordance(dataset, dataset)

        self.assertEqual(sum([sum(glob_conc[i]) for i in range(5)]), dataset.count_rows() * dataset.count_cols())

        counts = dataset.aggregate_entries(hl.Struct(n_het=agg.filter(dataset.GT.is_het(), agg.count()),
                                                     n_hom_ref=agg.filter(dataset.GT.is_hom_ref(),
                                                                          agg.count()),
                                                     n_hom_var=agg.filter(dataset.GT.is_hom_var(),
                                                                          agg.count()),
                                                     nNoCall=agg.filter(hl.is_missing(dataset.GT),
                                                                        agg.count())))

        self.assertEqual(glob_conc[0][0], 0)
        self.assertEqual(glob_conc[1][1], counts.nNoCall)
        self.assertEqual(glob_conc[2][2], counts.n_hom_ref)
        self.assertEqual(glob_conc[3][3], counts.n_het)
        self.assertEqual(glob_conc[4][4], counts.n_hom_var)
        [self.assertEqual(glob_conc[i][j], 0) for i in range(5) for j in range(5) if i != j]

        self.assertTrue(cols_conc.all(hl.sum(hl.flatten(cols_conc.concordance)) == dataset.count_rows()))
        self.assertTrue(rows_conc.all(hl.sum(hl.flatten(rows_conc.concordance)) == dataset.count_cols()))

        cols_conc.write('/tmp/foo.kt', overwrite=True)
        rows_conc.write('/tmp/foo.kt', overwrite=True)

    def test_filter_alleles(self):
        # poor man's Gen
        paths = [resource('sample.vcf'),
                 resource('multipleChromosomes.vcf'),
                 resource('sample2.vcf')]
        for path in paths:
            ds = hl.import_vcf(path)
            self.assertEqual(
                hl.filter_alleles(ds, lambda a, i: False).count_rows(), 0)
            self.assertEqual(hl.filter_alleles(ds, lambda a, i: True).count_rows(), ds.count_rows())

    def test_filter_alleles_hts(self):
        # 1 variant: A:T,G
        ds = hl.import_vcf(resource('filter_alleles/input.vcf'))

        self.assertTrue(
            hl.filter_alleles_hts(ds, lambda a, i: a == 'T', subset=True)
                .drop('old_alleles', 'old_locus', 'new_to_old', 'old_to_new')
                ._same(hl.import_vcf(resource('filter_alleles/keep_allele1_subset.vcf'))))

        self.assertTrue(
            hl.filter_alleles_hts(ds, lambda a, i: a == 'G', subset=True)
                .drop('old_alleles', 'old_locus', 'new_to_old', 'old_to_new')
                ._same(hl.import_vcf(resource('filter_alleles/keep_allele2_subset.vcf')))
        )

        self.assertTrue(
            hl.filter_alleles_hts(ds, lambda a, i: a != 'G', subset=False)
                .drop('old_alleles', 'old_locus', 'new_to_old', 'old_to_new')
                ._same(hl.import_vcf(resource('filter_alleles/keep_allele1_downcode.vcf')))
        )

        (hl.filter_alleles_hts(ds, lambda a, i: a == 'G', subset=False)).old_to_new.show()
        self.assertTrue(
            hl.filter_alleles_hts(ds, lambda a, i: a == 'G', subset=False)
                .drop('old_alleles', 'old_locus', 'new_to_old', 'old_to_new')
                ._same(hl.import_vcf(resource('filter_alleles/keep_allele2_downcode.vcf')))
        )
