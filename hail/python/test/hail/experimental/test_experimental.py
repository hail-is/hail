import numpy as np
import hail as hl
import unittest
from ..helpers import *
from hail.utils import new_temp_file, new_local_temp_dir

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    """
    @skip_unless_spark_backend()
    def test_calculate_ld_scores(self):

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

        ht_univariate = hl.experimental.ld_score.calculate_ld_scores(
            entry_expr=mt.GT.n_alt_alleles(),
            locus_expr=mt.locus,
            radius=1.0,
            coord_expr=mt.cm_position)

        ht_annotated = hl.experimental.ld_score.calculate_ld_scores(
            entry_expr=mt.GT.n_alt_alleles(),
            locus_expr=mt.locus,
            radius=1.0,
            coord_expr=mt.cm_position,
            annotation_exprs=[mt.binary,
                              mt.continuous])

        univariate = ht_univariate.aggregate(hl.struct(
            chr20=hl.agg.filter(
                (ht_univariate.locus.contig == '20') &
                (ht_univariate.locus.position == 82079),
                hl.agg.collect(ht_univariate.univariate))[0],
            chr22 =hl.agg.filter(
                (ht_univariate.locus.contig == '22') &
                (ht_univariate.locus.position == 16894090),
                hl.agg.collect(ht_univariate.univariate))[0],
            mean=hl.agg.mean(ht_univariate.univariate)))

        self.assertAlmostEqual(univariate.chr20, 1.601, places=3)
        self.assertAlmostEqual(univariate.chr22, 1.140, places=3)
        self.assertAlmostEqual(univariate.mean, 3.507, places=3)

        annotated = ht_annotated.aggregate(
            hl.struct(
                chr20=hl.struct(binary=hl.agg.filter(
                    (ht_annotated.locus.contig == '20') &
                    (ht_annotated.locus.position == 82079),
                    hl.agg.collect(ht_annotated.binary))[0],
                                continuous=hl.agg.filter(
                                    (ht_annotated.locus.contig == '20') &
                                    (ht_annotated.locus.position == 82079),
                                    hl.agg.collect(ht_annotated.continuous))[0]),
                chr22=hl.struct(
                    binary=hl.agg.filter(
                        (ht_annotated.locus.contig == '22') &
                        (ht_annotated.locus.position == 16894090),
                        hl.agg.collect(ht_annotated.binary))[0],
                    continuous=hl.agg.filter(
                        (ht_annotated.locus.contig == '22') &
                        (ht_annotated.locus.position == 16894090),
                        hl.agg.collect(ht_annotated.continuous))[0]),
                mean_stats=hl.struct(binary=hl.agg.mean(ht_annotated.binary),
                                     continuous=hl.agg.mean(ht_annotated.continuous))))

        self.assertAlmostEqual(annotated.chr20.binary, 1.152, places=3)
        self.assertAlmostEqual(annotated.chr20.continuous, 73.014, places=3)
        self.assertAlmostEqual(annotated.chr22.binary, 1.107, places=3)
        self.assertAlmostEqual(annotated.chr22.continuous, 102.174, places=3)
        self.assertAlmostEqual(annotated.mean_stats.binary, 0.965, places=3)
        self.assertAlmostEqual(annotated.mean_stats.continuous, 176.528, places=3)
    """

    @skip_unless_spark_backend()
    def test_plot_roc_curve(self):
        x = hl.utils.range_table(100).annotate(score1=hl.rand_norm(), score2=hl.rand_norm())
        x = x.annotate(tp=hl.cond(x.score1 > 0, hl.rand_bool(0.7), False), score3=x.score1 + hl.rand_norm())
        ht = x.annotate(fp=hl.cond(~x.tp, hl.rand_bool(0.2), False))
        _, aucs = hl.experimental.plot_roc_curve(ht, ['score1', 'score2', 'score3'])

    
    def test_ld_score_regression(self):
        mt = hl.read_matrix_table(doctest_resource('ld_score_regression.sample.mt'))

        ht = hl.experimental.ld_score.ld_score_regression(
            z_expr=mt['Z'],
            n_samples_expr=mt['N'],
            ld_score_exprs=[mt['ld_scores'][x] for x in list(mt['ld_scores'])][0],
            weight_expr=mt['weight'],
            n_reference_panel_variants_exprs=[mt['M_5_50'][x] for x in list(mt['M_5_50'])][0],
            n_blocks=4,
            two_step_threshold=30,
            n_iterations=3,
            max_chi_sq=None)
        results = {x['trait']: x for x in ht.collect()}
        
        self.assertAlmostEqual(
            results['50_irnt']['mean_chi_sq'],
            1.3852, places=4)
        self.assertAlmostEqual(
            results['50_irnt']['intercept']['estimate'],
            3.8540, places=4)
        self.assertAlmostEqual(
            results['50_irnt']['intercept']['standard_error'],
            2.6233, places=4)
        self.assertAlmostEqual(
            results['50_irnt']['snp_heritability']['estimate'],
            -0.2262, places=4)
        self.assertAlmostEqual(
            results['50_irnt']['snp_heritability']['standard_error'],
            0.2957, places=4)
    
    def test_sparse(self):
        expected_split_mt = hl.import_vcf(resource('sparse_split_test_b.vcf'))
        unsplit_mt = hl.import_vcf(resource('sparse_split_test.vcf'), call_fields=['LGT', 'LPGT'])
        mt = (hl.experimental.sparse_split_multi(unsplit_mt)
              .drop('a_index', 'was_split').select_entries(*expected_split_mt.entry.keys()))
        assert mt._same(expected_split_mt)
    
    def test_define_function(self):
        f = hl.experimental.define_function(
            lambda a, b: (a + 7) * b, hl.tint32, hl.tint32)
        self.assertEqual(hl.eval(f(1, 3)), 24)

    def test_mt_full_outer_join(self):
        mt1 = hl.utils.range_matrix_table(10, 10)
        mt1 = mt1.annotate_cols(c1=hl.rand_unif(0, 1))
        mt1 = mt1.annotate_rows(r1=hl.rand_unif(0, 1))
        mt1 = mt1.annotate_entries(e1=hl.rand_unif(0, 1))

        mt2 = hl.utils.range_matrix_table(10, 10)
        mt2 = mt2.annotate_cols(c1=hl.rand_unif(0, 1))
        mt2 = mt2.annotate_rows(r1=hl.rand_unif(0, 1))
        mt2 = mt2.annotate_entries(e1=hl.rand_unif(0, 1))

        mtj = hl.experimental.full_outer_join_mt(mt1, mt2)
        assert(mtj.aggregate_entries(hl.agg.all(mtj.left_entry == mt1.index_entries(mtj.row_key, mtj.col_key))))
        assert(mtj.aggregate_entries(hl.agg.all(mtj.right_entry == mt2.index_entries(mtj.row_key, mtj.col_key))))

        mt2 = mt2.key_cols_by(new_col_key = 5 - (mt2.col_idx // 2)) # duplicate col keys
        mt1 = mt1.key_rows_by(new_row_key = 5 - (mt1.row_idx // 2)) # duplicate row keys
        mtj = hl.experimental.full_outer_join_mt(mt1, mt2)

        assert(mtj.count() == (15, 15))

    def test_block_matrices_tofiles(self):
        data = [
            np.random.rand(11*12),
            np.random.rand(5*17)
        ]
        arrs = [
            data[0].reshape((11, 12)),
            data[1].reshape((5, 17))
        ]
        bms = [
            hl.linalg.BlockMatrix._create(11, 12, data[0].tolist(), block_size=4),
            hl.linalg.BlockMatrix._create(5, 17, data[1].tolist(), block_size=8)
        ]
        prefix = new_local_temp_dir()
        hl.experimental.block_matrices_tofiles(bms, f'{prefix}/files')
        for i in range(len(bms)):
            a = data[i]
            a2 = np.fromfile(f'{prefix}/files/{i}')
            self.assertTrue(np.array_equal(a, a2))

    def test_export_block_matrices(self):
        data = [
            np.random.rand(11*12),
            np.random.rand(5*17)
        ]
        arrs = [
            data[0].reshape((11, 12)),
            data[1].reshape((5, 17))
        ]
        bms = [
            hl.linalg.BlockMatrix._create(11, 12, data[0].tolist(), block_size=4),
            hl.linalg.BlockMatrix._create(5, 17, data[1].tolist(), block_size=8)
        ]
        prefix = new_local_temp_dir()
        hl.experimental.export_block_matrices(bms, f'{prefix}/files')
        for i in range(len(bms)):
            a = arrs[i]
            a2 = np.loadtxt(f'{prefix}/files/{i}.tsv')
            self.assertTrue(np.array_equal(a, a2))
