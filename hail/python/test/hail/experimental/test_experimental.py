import hail as hl
import unittest
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):

    def test_ld_score(self):

        ht = hl.import_table(doctest_resource('ldsc.annot'),
                             types={'BP': hl.tint,
                                    'CM': hl.tfloat,
                                    'binary': hl.tint,
                                    'continuous': hl.tfloat})
        ht = ht.annotate(locus=hl.locus(ht.CHR, ht.BP))
        ht = ht.key_by('locus')

        mt = hl.import_plink(bed=doctest_resource('ldsc.bed'),
                             bim=doctest_resource('ldsc.bim'),
                             fam=doctest_resource('ldsc.fam'))
        mt = mt.annotate_rows(binary=ht[mt.locus].binary,
                              continuous=ht[mt.locus].continuous)

        ht_univariate = hl.experimental.ld_score(
            entry_expr=mt.GT.n_alt_alleles(),
            locus_expr=mt.locus,
            radius=1.0,
            coord_expr=mt.cm_position)

        ht_annotated = hl.experimental.ld_score(
            entry_expr=mt.GT.n_alt_alleles(),
            locus_expr=mt.locus,
            radius=1.0,
            coord_expr=mt.cm_position,
            annotation_exprs=[mt.binary,
                              mt.continuous])

        chr20_univariate = ht_univariate.aggregate(
          hl.struct(univariate=hl.agg.filter(
              (ht_univariate.locus.contig == '20') &
              (ht_univariate.locus.position == 82079),
              hl.agg.collect(ht_univariate.univariate))[0]))

        chr20_annotated = ht_annotated.aggregate(
            hl.struct(binary=hl.agg.filter(
                (ht_annotated.locus.contig == '20') &
                (ht_annotated.locus.position == 82079),
                hl.agg.collect(ht_annotated.binary))[0],
                      continuous=hl.agg.filter(
                          (ht_annotated.locus.contig == '20') &
                          (ht_annotated.locus.position == 82079),
                          hl.agg.collect(ht_annotated.continuous))[0]))

        self.assertAlmostEqual(chr20_univariate.univariate, 1.601, places=3)
        self.assertAlmostEqual(chr20_annotated.binary, 1.152, places=3)
        self.assertAlmostEqual(chr20_annotated.continuous, 73.014, places=3)

        chr22_univariate = ht_univariate.aggregate(
          hl.struct(univariate=hl.agg.filter(
              (ht_univariate.locus.contig == '22') &
              (ht_univariate.locus.position == 16894090),
              hl.agg.collect(ht_univariate.univariate))[0]))

        chr22_annotated = ht_annotated.aggregate(
            hl.struct(
                binary=hl.agg.filter(
                    (ht_annotated.locus.contig == '22') &
                    (ht_annotated.locus.position == 16894090),
                    hl.agg.collect(ht_annotated.binary))[0],
                continuous=hl.agg.filter(
                    (ht_annotated.locus.contig == '22') &
                    (ht_annotated.locus.position == 16894090),
                    hl.agg.collect(ht_annotated.continuous))[0]))

        self.assertAlmostEqual(chr22_univariate.univariate, 1.140, places=3)
        self.assertAlmostEqual(chr22_annotated.binary, 1.107, places=3)
        self.assertAlmostEqual(chr22_annotated.continuous, 102.174, places=3)

        mean_univariate = ht_univariate.aggregate(
            hl.struct(univariate=hl.agg.mean(ht_univariate.univariate)))

        mean_annotated = ht_annotated.aggregate(
            hl.struct(binary=hl.agg.mean(ht_annotated.binary),
                      continuous=hl.agg.mean(ht_annotated.continuous)))

        self.assertAlmostEqual(mean_univariate.univariate, 3.507, places=3)
        self.assertAlmostEqual(mean_annotated.binary, 0.965, places=3)
        self.assertAlmostEqual(mean_annotated.continuous, 176.528, places=3)

    def test_plot_roc_curve(self):
        x = hl.utils.range_table(100).annotate(score1=hl.rand_norm(), score2=hl.rand_norm())
        x = x.annotate(tp=hl.cond(x.score1 > 0, hl.rand_bool(0.7), False), score3=x.score1 + hl.rand_norm())
        ht = x.annotate(fp=hl.cond(~x.tp, hl.rand_bool(0.2), False))
        _, aucs = hl.experimental.plot_roc_curve(ht, ['score1', 'score2', 'score3'])

    def test_ld_score_regression(self):
        mt = hl.read_matrix_table(doctest_resource('univariate_ld_score_regression.chr22_sample.mt'))
        ht_results = hl.experimental.ld_score_regression(weight_expr=mt['ld_score'],
                                                         ld_score_expr=mt['ld_score'],
                                                         chi_squared_exprs=mt['chi_squared'],
                                                         n_samples_exprs=mt['n_complete_samples'],
                                                         n_blocks=20,
                                                         two_step_threshold=10,
                                                         n_reference_panel_variants=1173569)
        results = {x['phenotype']: {'mean_chi_squared': x['mean_chi_squared'],
                                    'intercept_estimate': x['intercept']['estimate'],
                                    'intercept_standard_error': x['intercept']['standard_error'],
                                    'snp_heritability_estimate': x['snp_heritability']['estimate'],
                                    'snp_heritability_standard_error': x['snp_heritability']['standard_error']}
                   for x in ht_results.collect()}

        self.assertAlmostEqual(results['50_irnt']['mean_chi_squared'], 3.531, places=3)
        self.assertAlmostEqual(results['50_irnt']['intercept_estimate'], 0.559, places=3)
        self.assertAlmostEqual(results['50_irnt']['intercept_standard_error'], 0.328, places=1)
        self.assertAlmostEqual(results['50_irnt']['snp_heritability_estimate'], 0.497, places=3)
        self.assertAlmostEqual(results['50_irnt']['snp_heritability_standard_error'], 0.148, places=1)

        self.assertAlmostEqual(results['2443']['mean_chi_squared'], 1.265, places=3)
        self.assertAlmostEqual(results['2443']['intercept_estimate'], 0.372, places=3)
        self.assertAlmostEqual(results['2443']['intercept_standard_error'], 0.105, places=1)
        self.assertAlmostEqual(results['2443']['snp_heritability_estimate'], 0.223, places=3)
        self.assertAlmostEqual(results['2443']['snp_heritability_standard_error'], 0.052, places=1)

        self.assertAlmostEqual(results['20160']['mean_chi_squared'], 1.521, places=3)
        self.assertAlmostEqual(results['20160']['intercept_estimate'], 1.271, places=3)
        self.assertAlmostEqual(results['20160']['intercept_standard_error'], 0.210, places=1)
        self.assertAlmostEqual(results['20160']['snp_heritability_estimate'], 0.037, places=3)
        self.assertAlmostEqual(results['20160']['snp_heritability_standard_error'], 0.052, places=1)

        self.assertAlmostEqual(results['30100_irnt']['mean_chi_squared'], 5.241, places=3)
        self.assertAlmostEqual(results['30100_irnt']['intercept_estimate'], 0.858, places=3)
        self.assertAlmostEqual(results['30100_irnt']['intercept_standard_error'], 0.324, places=1)
        self.assertAlmostEqual(results['30100_irnt']['snp_heritability_estimate'], 0.769, places=3)
        self.assertAlmostEqual(results['30100_irnt']['snp_heritability_standard_error'], 0.308, places=1)

        ht = hl.read_table(doctest_resource('univariate_ld_score_regression.chr22_sample.ht'))
        ht_results = hl.experimental.ld_score_regression(weight_expr=ht['ld_score'],
                                                         ld_score_expr=ht['ld_score'],
                                                         chi_squared_exprs=[ht['50_irnt_chi_squared'],
                                                                            ht['2443_chi_squared'],
                                                                            ht['20160_chi_squared'],
                                                                            ht['30100_irnt_chi_squared']],
                                                         n_samples_exprs=[ht['50_irnt_n_complete_samples'],
                                                                          ht['2443_n_complete_samples'],
                                                                          ht['20160_n_complete_samples'],
                                                                          ht['30100_irnt_n_complete_samples']],
                                                         n_blocks=20,
                                                         two_step_threshold=10,
                                                         n_reference_panel_variants=1173569)
        results = {x['phenotype']: {'mean_chi_squared': x['mean_chi_squared'],
                                    'intercept_estimate': x['intercept']['estimate'],
                                    'intercept_standard_error': x['intercept']['standard_error'],
                                    'snp_heritability_estimate': x['snp_heritability']['estimate'],
                                    'snp_heritability_standard_error': x['snp_heritability']['standard_error']}
                   for x in ht_results.collect()}

        self.assertAlmostEqual(results['y0']['mean_chi_squared'], 3.531, places=3)
        self.assertAlmostEqual(results['y0']['intercept_estimate'], 0.559, places=3)
        self.assertAlmostEqual(results['y0']['intercept_standard_error'], 0.328, places=1)
        self.assertAlmostEqual(results['y0']['snp_heritability_estimate'], 0.497, places=3)
        self.assertAlmostEqual(results['y0']['snp_heritability_standard_error'], 0.148, places=1)

        self.assertAlmostEqual(results['y1']['mean_chi_squared'], 1.265, places=3)
        self.assertAlmostEqual(results['y1']['intercept_estimate'], 0.372, places=3)
        self.assertAlmostEqual(results['y1']['intercept_standard_error'], 0.105, places=1)
        self.assertAlmostEqual(results['y1']['snp_heritability_estimate'], 0.223, places=3)
        self.assertAlmostEqual(results['y1']['snp_heritability_standard_error'], 0.052, places=1)

        self.assertAlmostEqual(results['y2']['mean_chi_squared'], 1.521, places=3)
        self.assertAlmostEqual(results['y2']['intercept_estimate'], 1.271, places=3)
        self.assertAlmostEqual(results['y2']['intercept_standard_error'], 0.210, places=1)
        self.assertAlmostEqual(results['y2']['snp_heritability_estimate'], 0.037, places=3)
        self.assertAlmostEqual(results['y2']['snp_heritability_standard_error'], 0.052, places=1)

        self.assertAlmostEqual(results['y3']['mean_chi_squared'], 5.241, places=3)
        self.assertAlmostEqual(results['y3']['intercept_estimate'], 0.858, places=3)
        self.assertAlmostEqual(results['y3']['intercept_standard_error'], 0.324, places=1)
        self.assertAlmostEqual(results['y3']['snp_heritability_estimate'], 0.769, places=3)
        self.assertAlmostEqual(results['y3']['snp_heritability_standard_error'], 0.308, places=1)
