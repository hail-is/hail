import numpy as np
import hail as hl
import unittest
import pytest
from ..helpers import *
from hail.utils import new_temp_file


class Tests(unittest.TestCase):
    @fails_service_backend()
    @fails_local_backend
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


    @backend_specific_timeout(batch=4 * 60)
    def test_plot_roc_curve(self):
        x = hl.utils.range_table(100).annotate(score1=hl.rand_norm(), score2=hl.rand_norm())
        x = x.annotate(tp=hl.if_else(x.score1 > 0, hl.rand_bool(0.7), False), score3=x.score1 + hl.rand_norm())
        ht = x.annotate(fp=hl.if_else(~x.tp, hl.rand_bool(0.2), False))
        _, aucs = hl.experimental.plot_roc_curve(ht, ['score1', 'score2', 'score3'])

    def test_import_keyby_count_ldsc_lowered_shuffle(self):
        # integration test pulled out of test_ld_score_regression to isolate issues with lowered shuffles
        # and RDD serialization, 2021-07-06
        # if this comment no longer reflects the backend system, that's a really good thing
        ht_scores = hl.import_table(
            doctest_resource('ld_score_regression.univariate_ld_scores.tsv'),
            key='SNP', types={'L2': hl.tfloat, 'BP': hl.tint})

        ht_20160 = hl.import_table(
            doctest_resource('ld_score_regression.20160.sumstats.tsv'),
            key='SNP', types={'N': hl.tint, 'Z': hl.tfloat})

        j1 = ht_scores[ht_20160['SNP']]
        ht_20160 = ht_20160.annotate(
            ld_score=j1['L2'],
            locus=hl.locus(j1['CHR'],
                           j1['BP']),
            alleles=hl.array([ht_20160['A2'], ht_20160['A1']]))

        ht_20160 = ht_20160.key_by(ht_20160['locus'],
                                   ht_20160['alleles'])
        assert ht_20160._force_count() == 151


    @pytest.mark.unchecked_allocator
    @backend_specific_timeout(6 * 60, 10 * 60)
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

    def test_sparse(self):
        expected_split_mt = hl.import_vcf(resource('sparse_split_test_b.vcf'))
        unsplit_mt = hl.import_vcf(resource('sparse_split_test.vcf'), call_fields=['LGT', 'LPGT'])
        mt = (hl.experimental.sparse_split_multi(unsplit_mt)
              .drop('a_index', 'was_split').select_entries(*expected_split_mt.entry.keys()))
        assert mt._same(expected_split_mt)

    def test_define_function(self):
        f1 = hl.experimental.define_function(
            lambda a, b: (a + 7) * b, hl.tint32, hl.tint32)
        self.assertEqual(hl.eval(f1(1, 3)), 24)
        f2 = hl.experimental.define_function(
            lambda a, b: (a + 7) * b, hl.tint32, hl.tint32)
        self.assertEqual(hl.eval(f1(1, 3)), 24) # idempotent
        self.assertEqual(hl.eval(f2(1, 3)), 24) # idempotent

    @fails_local_backend()
    @backend_specific_timeout(batch=4 * 60)
    def test_pc_project(self):
        mt = hl.balding_nichols_model(3, 100, 50)
        _, _, loadings_ht = hl.hwe_normalized_pca(mt.GT, k=10, compute_loadings=True)
        mt = mt.annotate_rows(af=hl.agg.mean(mt.GT.n_alt_alleles()) / 2)
        loadings_ht = loadings_ht.annotate(af=mt.rows()[loadings_ht.key].af)
        mt_to_project = hl.balding_nichols_model(3, 100, 50)
        ht = hl.experimental.pc_project(mt_to_project.GT, loadings_ht.loadings, loadings_ht.af)
        assert ht._force_count() == 100

    @backend_specific_timeout(batch=4 * 60)
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

    def test_mt_full_outer_join_self(self):
        mt = hl.import_vcf(resource('sample.vcf'))
        jmt = hl.experimental.full_outer_join_mt(mt, mt)
        assert jmt.filter_cols(hl.is_defined(jmt.left_col) & hl.is_defined(jmt.right_col)).count_cols() == mt.count_cols()
        assert jmt.filter_rows(hl.is_defined(jmt.left_row) & hl.is_defined(jmt.right_row)).count_rows() == mt.count_rows()
        assert jmt.filter_entries(hl.is_defined(jmt.left_entry) & hl.is_defined(jmt.right_entry)).entries().count() == mt.entries().count()

    @fails_service_backend()
    @fails_local_backend()
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
        with hl.TemporaryDirectory() as prefix:
            hl.experimental.block_matrices_tofiles(bms, f'{prefix}/files')
            for i in range(len(bms)):
                a = data[i]
                a2 = np.frombuffer(
                    hl.current_backend().fs.open(f'{prefix}/files/{i}', mode='rb').read())
                self.assertTrue(np.array_equal(a, a2))

    @fails_service_backend()
    @fails_local_backend()
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
        with hl.TemporaryDirectory() as prefix:
            hl.experimental.export_block_matrices(bms, f'{prefix}/files')
            for i in range(len(bms)):
                a = arrs[i]
                a2 = np.loadtxt(
                    hl.current_backend().fs.open(f'{prefix}/files/{i}.tsv'))
                self.assertTrue(np.array_equal(a, a2))

        with hl.TemporaryDirectory() as prefix2:
            custom_names = ["nameA", "inner/nameB.tsv"]
            hl.experimental.export_block_matrices(bms, f'{prefix2}/files', custom_filenames=custom_names)
            for i in range(len(bms)):
                a = arrs[i]
                a2 = np.loadtxt(
                    hl.current_backend().fs.open(f'{prefix2}/files/{custom_names[i]}'))
                self.assertTrue(np.array_equal(a, a2))

    def test_trivial_loop(self):
        assert hl.eval(hl.experimental.loop(lambda recur: 0, hl.tint32)) == 0

    def test_loop(self):
        def triangle_with_ints(n):
            return hl.experimental.loop(
                lambda f, x, c: hl.if_else(x > 0, f(x - 1, c + x), c),
                hl.tint32, n, 0)

        def triangle_with_tuple(n):
            return hl.experimental.loop(
                lambda f, xc: hl.if_else(xc[0] > 0, f((xc[0] - 1, xc[1] + xc[0])), xc[1]),
                hl.tint32, (n, 0))

        for triangle in [triangle_with_ints, triangle_with_tuple]:
            assert_evals_to(triangle(20), sum(range(21)))
            assert_evals_to(triangle(0), 0)
            assert_evals_to(triangle(-1), 0)

        def fails_typecheck(regex, f):
            with self.assertRaisesRegex(TypeError, regex):
                hl.eval(hl.experimental.loop(f, hl.tint32, 1))

        fails_typecheck("outside of tail position",
                        lambda f, x: x + f(x))
        fails_typecheck("wrong number of arguments",
                        lambda f, x: f(x, x + 1))
        fails_typecheck("bound value",
                        lambda f, x: hl.bind(lambda x: x, f(x)))
        fails_typecheck("branch condition",
                        lambda f, x: hl.if_else(f(x) == 0, x, 1))
        fails_typecheck("Type error",
                        lambda f, x: hl.if_else(x == 0, f("foo"), 1))

    def test_nested_loops(self):
        def triangle_loop(n, add_f):
            recur = lambda f, x, c: hl.if_else(x <= n, f(x + 1, add_f(x, c)), c)
            return hl.experimental.loop(recur, hl.tint32, 0, 0)

        assert_evals_to(triangle_loop(5, lambda x, c: c + x), 15)
        assert_evals_to(triangle_loop(5, lambda x, c: c + triangle_loop(x, lambda x2, c2: c2 + x2)), 15 + 10 + 6 + 3 + 1)

        n1 = 5
        calls_recur_from_nested_loop = hl.experimental.loop(
            lambda f, x1, c1:
            hl.if_else(x1 <= n1,
                    hl.experimental.loop(
                        lambda f2, x2, c2:
                        hl.if_else(x2 <= x1,
                                f2(x2 + 1, c2 + x2),
                                f(x1 + 1, c1 + c2)),
                        'int32', 0, 0),
                    c1),
            'int32', 0, 0)

        assert_evals_to(calls_recur_from_nested_loop, 15 + 10 + 6 + 3 + 1)

    def test_loop_errors(self):
        with pytest.raises(TypeError, match="requested type ndarray<int32, 2> does not match inferred type ndarray<float64, 2>"):
            result = hl.experimental.loop(
                lambda f, my_nd:
                hl.if_else(my_nd[0, 0] == 1000, my_nd, f(my_nd + 1)),
                hl.tndarray(hl.tint32, 2), hl.nd.zeros((20, 10), hl.tfloat64))

    def test_loop_with_struct_of_strings(self):
        def loop_func(recur_f, my_struct):
            return hl.if_else(hl.len(my_struct.s1) > hl.len(my_struct.s2),
                              my_struct,
                              recur_f(hl.struct(s1=my_struct.s1 + my_struct.s2[-1], s2=my_struct.s2[:-1])))

        initial_struct = hl.struct(s1="a", s2="gfedcb")
        assert hl.eval(hl.experimental.loop(loop_func, hl.tstruct(s1=hl.tstr, s2=hl.tstr), initial_struct)) == hl.Struct(s1="abcd", s2="gfe")

    def test_loop_memory(self):
        def foo(recur, arr, idx): return hl.if_else(idx > 10, arr, recur(arr.append(hl.str(idx)), idx+1))

        assert hl.eval(hl.experimental.loop(foo, hl.tarray(hl.tstr), hl.literal(['foo']), 1)) == ['foo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
