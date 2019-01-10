import hail as hl
import unittest
from ..helpers import *
from hail.utils import new_temp_file

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

        ht_scores = hl.import_table(
            doctest_resource('ld_score_regression.univariate_ld_scores.tsv'),
            key='SNP', types={'L2': hl.tfloat, 'BP': hl.tint})

        ht_50_irnt = hl.import_table(
            doctest_resource('ld_score_regression.50_irnt.sumstats.tsv'),
            key='SNP', types={'N': hl.tint, 'Z': hl.tfloat})

        ht_50_irnt = ht_50_irnt.annotate(
            chi_squared=ht_50_irnt['Z']**2,
            n=ht_50_irnt['N'],
            ld_score=ht_scores[ht_50_irnt['SNP']]['L2'],
            locus=hl.locus(ht_scores[ht_50_irnt['SNP']]['CHR'],
                           ht_scores[ht_50_irnt['SNP']]['BP']),
            alleles=hl.array([ht_50_irnt['A2'], ht_50_irnt['A1']]),
            phenotype='50_irnt')

        ht_50_irnt = ht_50_irnt.key_by(ht_50_irnt['locus'],
                                       ht_50_irnt['alleles'])

        ht_50_irnt = ht_50_irnt.select(ht_50_irnt['chi_squared'],
                                       ht_50_irnt['n'],
                                       ht_50_irnt['ld_score'],
                                       ht_50_irnt['phenotype'])

        ht_20160 = hl.import_table(
            doctest_resource('ld_score_regression.20160.sumstats.tsv'),
            key='SNP', types={'N': hl.tint, 'Z': hl.tfloat})

        ht_20160 = ht_20160.annotate(
            chi_squared=ht_20160['Z']**2,
            n=ht_20160['N'],
            ld_score=ht_scores[ht_20160['SNP']]['L2'],
            locus=hl.locus(ht_scores[ht_20160['SNP']]['CHR'],
                           ht_scores[ht_20160['SNP']]['BP']),
            alleles=hl.array([ht_20160['A2'], ht_20160['A1']]),
            phenotype='20160')

        ht_20160 = ht_20160.key_by(ht_20160['locus'],
                                   ht_20160['alleles'])

        ht_20160 = ht_20160.select(ht_20160['chi_squared'],
                                   ht_20160['n'],
                                   ht_20160['ld_score'],
                                   ht_20160['phenotype'])

        ht = ht_50_irnt.union(ht_20160)
        mt = ht.to_matrix_table(row_key=['locus', 'alleles'],
                                col_key=['phenotype'],
                                row_fields=['ld_score'],
                                col_fields=[])

        mt_tmp = new_temp_file()
        mt.write(mt_tmp, overwrite=True)
        mt = hl.read_matrix_table(mt_tmp)

        ht_results = hl.experimental.ld_score_regression(
            weight_expr=mt['ld_score'],
            ld_score_expr=mt['ld_score'],
            chi_sq_exprs=mt['chi_squared'],
            n_samples_exprs=mt['n'],
            n_blocks=20,
            two_step_threshold=5,
            n_reference_panel_variants=1173569)

        results = {
            x['phenotype']: {
                'mean_chi_sq': x['mean_chi_sq'],
                'intercept_estimate': x['intercept']['estimate'],
                'intercept_standard_error': x['intercept']['standard_error'],
                'snp_heritability_estimate': x['snp_heritability']['estimate'],
                'snp_heritability_standard_error':
                    x['snp_heritability']['standard_error']}
            for x in ht_results.collect()}

        self.assertAlmostEqual(
            results['50_irnt']['mean_chi_sq'],
            3.4386, places=4)
        self.assertAlmostEqual(
            results['50_irnt']['intercept_estimate'],
            0.7727, places=4)
        self.assertAlmostEqual(
            results['50_irnt']['intercept_standard_error'],
            0.2461, places=4)
        self.assertAlmostEqual(
            results['50_irnt']['snp_heritability_estimate'],
            0.3845, places=4)
        self.assertAlmostEqual(
            results['50_irnt']['snp_heritability_standard_error'],
            0.1067, places=4)

        self.assertAlmostEqual(
            results['20160']['mean_chi_sq'],
            1.5209, places=4)
        self.assertAlmostEqual(
            results['20160']['intercept_estimate'],
            1.2109, places=4)
        self.assertAlmostEqual(
            results['20160']['intercept_standard_error'],
            0.2238, places=4)
        self.assertAlmostEqual(
            results['20160']['snp_heritability_estimate'],
            0.0486, places=4)
        self.assertAlmostEqual(
            results['20160']['snp_heritability_standard_error'],
            0.0416, places=4)

        ht = ht_50_irnt.annotate(
            chi_squared_50_irnt=ht_50_irnt['chi_squared'],
            n_50_irnt=ht_50_irnt['n'],
            chi_squared_20160=ht_20160[ht_50_irnt.key]['chi_squared'],
            n_20160=ht_20160[ht_50_irnt.key]['n'])

        ht_results = hl.experimental.ld_score_regression(
            weight_expr=ht['ld_score'],
            ld_score_expr=ht['ld_score'],
            chi_sq_exprs=[ht['chi_squared_50_irnt'],
                               ht['chi_squared_20160']],
            n_samples_exprs=[ht['n_50_irnt'],
                             ht['n_20160']],
            n_blocks=20,
            two_step_threshold=5,
            n_reference_panel_variants=1173569)

        results = {
            x['phenotype']: {
                'mean_chi_sq': x['mean_chi_sq'],
                'intercept_estimate': x['intercept']['estimate'],
                'intercept_standard_error': x['intercept']['standard_error'],
                'snp_heritability_estimate': x['snp_heritability']['estimate'],
                'snp_heritability_standard_error':
                    x['snp_heritability']['standard_error']}
            for x in ht_results.collect()}

        self.assertAlmostEqual(
            results[0]['mean_chi_sq'],
            3.4386, places=4)
        self.assertAlmostEqual(
            results[0]['intercept_estimate'],
            0.7727, places=4)
        self.assertAlmostEqual(
            results[0]['intercept_standard_error'],
            0.2461, places=4)
        self.assertAlmostEqual(
            results[0]['snp_heritability_estimate'],
            0.3845, places=4)
        self.assertAlmostEqual(
            results[0]['snp_heritability_standard_error'],
            0.1067, places=4)

        self.assertAlmostEqual(
            results[1]['mean_chi_sq'],
            1.5209, places=4)
        self.assertAlmostEqual(
            results[1]['intercept_estimate'],
            1.2109, places=4)
        self.assertAlmostEqual(
            results[1]['intercept_standard_error'],
            0.2238, places=4)
        self.assertAlmostEqual(
            results[1]['snp_heritability_estimate'],
            0.0486, places=4)
        self.assertAlmostEqual(
            results[1]['snp_heritability_standard_error'],
            0.0416, places=4)
