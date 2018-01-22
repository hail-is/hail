from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail2 import *
from subprocess import call as syscall
import numpy as np
from struct import unpack
import hail.utils as utils
from hail.expr.expression import ExpressionException
from hail.utils.misc import test_file
from hail.linalg import BlockMatrix
from math import sqrt

hc = None

def setUpModule():
    init(master='local[2]', min_block_size=0)


def tearDownModule():
    stop()


class Tests(unittest.TestCase):
    _dataset = None

    def get_dataset(self):
        if Tests._dataset is None:
            Tests._dataset = methods.split_multi_hts(methods.import_vcf(test_file('sample.vcf')))
        return Tests._dataset

    def test_ibd(self):
        dataset = self.get_dataset()

        def plinkify(ds, min=None, max=None):
            vcf = utils.new_temp_file(prefix="plink", suffix="vcf")
            plinkpath = utils.new_temp_file(prefix="plink")
            methods.export_vcf(ds, vcf)
            threshold_string = "{} {}".format("--min {}".format(min) if min else "",
                                              "--max {}".format(max) if max else "")

            plink_command = "plink --double-id --allow-extra-chr --vcf {} --genome full --out {} {}"\
                .format(utils.get_URI(vcf),
                        utils.get_URI(plinkpath),
                        threshold_string)
            result_file = utils.get_URI(plinkpath + ".genome")

            syscall(plink_command, shell=True)

            ### format of .genome file is:
            # _, fid1, iid1, fid2, iid2, rt, ez, z0, z1, z2, pihat, phe,
            # dst, ppc, ratio, ibs0, ibs1, ibs2, homhom, hethet (+ separated)

            ### format of ibd is:
            # i (iid1), j (iid2), ibd: {Z0, Z1, Z2, PI_HAT}, ibs0, ibs1, ibs2
            results = {}
            with open(result_file) as f:
                f.readline()
                for line in f:
                    row = line.strip().split()
                    results[(row[1], row[3])] = (map(float, row[6:10]),
                                                 map(int, row[14:17]))
            return results

        def compare(ds, min=None, max=None):
            plink_results = plinkify(ds, min, max)
            hail_results = methods.ibd(ds, min=min, max=max).collect()

            for row in hail_results:
                key = (row.i, row.j)
                self.assertAlmostEqual(plink_results[key][0][0], row.ibd.Z0, places=4)
                self.assertAlmostEqual(plink_results[key][0][1], row.ibd.Z1, places=4)
                self.assertAlmostEqual(plink_results[key][0][2], row.ibd.Z2, places=4)
                self.assertAlmostEqual(plink_results[key][0][3], row.ibd.PI_HAT, places=4)
                self.assertEqual(plink_results[key][1][0], row.ibs0)
                self.assertEqual(plink_results[key][1][1], row.ibs1)
                self.assertEqual(plink_results[key][1][2], row.ibs2)

        compare(dataset)
        compare(dataset, min=0.0, max=1.0)
        dataset = dataset.annotate_rows(dummy_maf=0.01)
        methods.ibd(dataset, dataset['dummy_maf'], min=0.0, max=1.0)
        methods.ibd(dataset, dataset['dummy_maf'].to_float32(), min=0.0, max=1.0)

    def test_ld_matrix(self):
        dataset = self.get_dataset()

        ldm = methods.ld_matrix(dataset, force_local=True)

    def test_linreg(self):
        dataset = methods.import_vcf(test_file('regressionLinear.vcf'))

        phenos = methods.import_table(test_file('regressionLinear.pheno'),
                                 types={'Pheno': TFloat64()},
                                 key='Sample')
        covs = methods.import_table(test_file('regressionLinear.cov'),
                               types={'Cov1': TFloat64(), 'Cov2': TFloat64()},
                               key='Sample')

        dataset = dataset.annotate_cols(pheno=phenos[dataset.s], cov = covs[dataset.s])
        dataset = methods.linreg(dataset,
                         ys=dataset.pheno,
                         x=dataset.GT.num_alt_alleles(),
                         covariates=[dataset.cov.Cov1, dataset.cov.Cov2 + 1 - 1])

        dataset.count_rows()

    def test_trio_matrix(self):
        ped = Pedigree.read(test_file('triomatrix.fam'))
        fam_table = methods.import_fam(test_file('triomatrix.fam'))

        dataset = methods.import_vcf(test_file('triomatrix.vcf'))
        dataset = dataset.annotate_cols(fam = fam_table[dataset.s])

        tm = methods.trio_matrix(dataset, ped, complete_trios=True)

        tm.count_rows()

    def test_sample_qc(self):
        dataset = self.get_dataset()
        dataset = methods.sample_qc(dataset)

    def test_grm(self):
        tolerance = 0.001

        def load_id_file(path):
            ids = []
            with hadoop_read(path) as f:
                for l in f:
                    r = l.strip().split('\t')
                    self.assertEqual(len(r), 2)
                    ids.append(r[1])
            return ids

        def load_rel(ns, path):
            rel = np.zeros((ns, ns))
            with hadoop_read(path) as f:
                for i,l in enumerate(f):
                    for j,n in enumerate(map(float, l.strip().split('\t'))):
                        rel[i,j] = n
                    self.assertEqual(j, i)
                self.assertEqual(i, ns - 1)
            return rel

        def load_grm(ns, nv, path):
            m = np.zeros((ns, ns))
            with utils.hadoop_read(path) as f:
                i = 0
                for l in f:
                    row = l.strip().split('\t')
                    self.assertEqual(int(row[2]), nv)
                    m[int(row[0])-1, int(row[1])-1] = float(row[3])
                    i += 1

                self.assertEqual(i, ns * (ns + 1) / 2)
            return m

        def load_bin(ns, path):
            m = np.zeros((ns, ns))
            with utils.hadoop_read_binary(path) as f:
                for i in range(ns):
                    for j in range(i + 1):
                        b = f.read(4)
                        self.assertEqual(len(b), 4)
                        m[i, j] = unpack('<f', bytearray(b))[0]
                left = f.read()
                print(left)
                self.assertEqual(len(left), 0)
            return m

        b_file = utils.new_temp_file(prefix="plink")
        rel_file = utils.new_temp_file(prefix="test", suffix="rel")
        rel_id_file = utils.new_temp_file(prefix="test", suffix="rel.id")
        grm_file = utils.new_temp_file(prefix="test", suffix="grm")
        grm_bin_file = utils.new_temp_file(prefix="test", suffix="grm.bin")
        grm_nbin_file = utils.new_temp_file(prefix="test", suffix="grm.N.bin")

        dataset = self.get_dataset()
        n_samples = dataset.count_cols()
        dataset = dataset.annotate_rows(AC=agg.sum(dataset.GT.num_alt_alleles()),
                              n_called=agg.count_where(functions.is_defined(dataset.GT)))
        dataset = dataset.filter_rows((dataset.AC > 0) & (dataset.AC < 2 * dataset.n_called))
        dataset = dataset.filter_rows(dataset.n_called == n_samples).persist()

        methods.export_plink(dataset, b_file, id = dataset.s)

        sample_ids = [row.s for row in dataset.cols_table().select('s').collect()]
        n_variants = dataset.count_rows()
        self.assertGreater(n_variants, 0)

        grm = methods.grm(dataset)
        grm.export_id_file(rel_id_file)

        ############
        ### rel

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-rel --out {}'''
                .format(utils.get_URI(b_file), utils.get_URI(p_file)), shell=True)
        self.assertEqual(load_id_file(p_file + ".rel.id"), sample_ids)

        grm.export_rel(rel_file)
        self.assertEqual(load_id_file(rel_id_file), sample_ids)
        self.assertTrue(np.allclose(load_rel(n_samples, p_file + ".rel"),
                           load_rel(n_samples, rel_file),
                           atol=tolerance))

        ############
        ### gcta-grm

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-grm-gz --out {}'''
                .format(utils.get_URI(b_file), utils.get_URI(p_file)), shell=True)
        self.assertEqual(load_id_file(p_file + ".grm.id"), sample_ids)

        grm.export_gcta_grm(grm_file)
        self.assertTrue(np.allclose(load_grm(n_samples, n_variants, p_file + ".grm.gz"),
                           load_grm(n_samples, n_variants, grm_file),
                           atol=tolerance))

        ############
        ### gcta-grm-bin

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-grm-bin --out {}'''
                .format(utils.get_URI(b_file), utils.get_URI(p_file)), shell=True)

        self.assertEqual(load_id_file(p_file + ".grm.id"), sample_ids)

        grm.export_gcta_grm_bin(grm_bin_file, grm_nbin_file)

        self.assertTrue(np.allclose(load_bin(n_samples, p_file + ".grm.bin"),
                           load_bin(n_samples, grm_bin_file),
                           atol=tolerance))
        self.assertTrue(np.allclose(load_bin(n_samples, p_file + ".grm.N.bin"),
                           load_bin(n_samples, grm_nbin_file),
                           atol=tolerance))

    def test_rrm(self):
        seed = 0
        n1 = 100
        m1 = 200
        k = 3
        fst = .9

        dataset = methods.balding_nichols_model(k,
                                                n1,
                                                m1,
                                                fst=(k * [fst]),
                                                seed=seed,
                                                num_partitions=4)

        def direct_calculation(ds):
            ds = BlockMatrix.from_matrix_table(ds['GT'].num_alt_alleles()).to_numpy_matrix()

            # filter out constant rows
            isconst = lambda r: any([all([(gt < c + .01) and (gt > c - .01) for gt in r]) for c in range(3)])
            ds = ds[[not isconst(row) for row in ds]]

            nvariants, nsamples = ds.shape
            sumgt = lambda r: sum([i for i in r if i >= 0])
            sumsq = lambda r: sum([i**2 for i in r if i >= 0])

            mean = [sumgt(row) / nsamples for row in ds]
            stddev = [sqrt(sumsq(row) / nsamples - mean[i] ** 2)
                      for i, row in enumerate(ds)]

            mat = np.array([[(g-mean[i])/stddev[i] for g in row] for i, row in enumerate(ds)])

            rrm = mat.T.dot(mat) / nvariants
            
            return rrm

        def hail_calculation(ds):
            rrm = methods.rrm(ds['GT'])
            fn = utils.new_temp_file(suffix='.tsv')

            rrm.export_tsv(fn)
            data = []
            with open(utils.get_URI(fn)) as f:
                f.readline()
                for line in f:
                    row = line.strip().split()
                    data.append(map(float, row))

            return np.array(data)

        manual = direct_calculation(dataset)
        rrm = hail_calculation(dataset)

        self.assertTrue(np.allclose(manual, rrm))

    def test_pca(self):
        dataset = methods.balding_nichols_model(3, 100, 100)
        eigenvalues, scores, loadings = methods.pca(dataset.GT.num_alt_alleles(), k=2, compute_loadings=True)

        self.assertEqual(len(eigenvalues), 2)
        self.assertTrue(isinstance(scores, Table))
        self.assertEqual(scores.count(), 100)
        self.assertTrue(isinstance(loadings, Table))
        self.assertEqual(loadings.count(), 100)

        _, _, loadings = methods.pca(dataset.GT.num_alt_alleles(), k=2, compute_loadings=False)
        self.assertEqual(loadings, None)

    def test_pcrelate(self):
        dataset = methods.balding_nichols_model(3, 100, 100)
        t = methods.pc_relate(dataset, 2, 0.05, block_size=64, statistics="phi")

        self.assertTrue(isinstance(t, Table))
        t.count()

    def test_rename_duplicates(self):
        dataset = self.get_dataset() # FIXME - want to rename samples with same id
        renamed_ids = methods.rename_duplicates(dataset).cols_table().select('s').collect()
        self.assertTrue(len(set(renamed_ids)), len(renamed_ids))

    def test_split_multi_hts(self):
        ds1 = methods.import_vcf(test_file('split_test.vcf'))
        ds1 = methods.split_multi_hts(ds1)
        ds2 = methods.import_vcf(test_file('split_test_b.vcf'))
        self.assertEqual(ds1.aggregate_entries(foo = agg.product((ds1.wasSplit == (ds1.v.start != 1180)).to_int32())).foo, 1)
        ds1 = ds1.drop('wasSplit','aIndex')
        # required python3
        # self.assertTrue(ds1._same(ds2))

        ds = self.get_dataset()
        ds = ds.annotate_entries(X = ds.GT)
        self.assertRaisesRegexp(utils.FatalError,
                                "split_multi_hts: entry schema must be the HTS genotype schema",
                                methods.split_multi_hts,
                                ds)

    def test_mendel_errors(self):
        dataset = self.get_dataset()
        men, fam, ind, var = methods.mendel_errors(dataset, Pedigree.read(test_file('sample.fam')))
        men.select('fam_id', 's', 'code')
        fam.select('pat_id', 'children')
        self.assertEqual(ind.key, ['s'])
        self.assertEqual(var.key, ['v'])
        dataset.annotate_rows(mendel=var[dataset.v]).count_rows()

    def test_export_vcf(self):
        dataset = methods.import_vcf(test_file('sample.vcf.bgz'))
        vcf_metadata = methods.get_vcf_metadata(test_file('sample.vcf.bgz'))
        methods.export_vcf(dataset, '/tmp/sample.vcf', metadata=vcf_metadata)
        dataset_imported = methods.import_vcf('/tmp/sample.vcf')
        self.assertTrue(dataset._same(dataset_imported))

        metadata_imported = methods.get_vcf_metadata('/tmp/sample.vcf')
        self.assertDictEqual(vcf_metadata, metadata_imported)

    def test_concordance(self):
        dataset = self.get_dataset()
        glob_conc, cols_conc, rows_conc = methods.concordance(dataset, dataset)

        self.assertEqual(sum([sum(glob_conc[i]) for i in range(5)]), dataset.count_rows() * dataset.count_cols())

        counts = dataset.aggregate_entries(nHet=agg.count(agg.filter(dataset.GT.is_het(), dataset.GT)),
                                           nHomRef=agg.count(agg.filter(dataset.GT.is_hom_ref(), dataset.GT)),
                                           nHomVar=agg.count(agg.filter(dataset.GT.is_hom_var(), dataset.GT)),
                                           nNoCall=agg.count(agg.filter(functions.is_missing(dataset.GT), dataset.GT)))

        self.assertEqual(glob_conc[0][0], 0)
        self.assertEqual(glob_conc[1][1], counts.nNoCall)
        self.assertEqual(glob_conc[2][2], counts.nHomRef)
        self.assertEqual(glob_conc[3][3], counts.nHet)
        self.assertEqual(glob_conc[4][4], counts.nHomVar)
        [self.assertEqual(glob_conc[i][j], 0) for i in range(5) for j in range(5) if i != j]

        self.assertTrue(cols_conc.forall(cols_conc.concordance.flatten().sum() == dataset.count_rows()))
        self.assertTrue(rows_conc.forall(rows_conc.concordance.flatten().sum() == dataset.count_cols()))

        cols_conc.write('/tmp/foo.kt', overwrite=True)
        rows_conc.write('/tmp/foo.kt', overwrite=True)

    def test_import_interval_list(self):
        interval_file = test_file('annotinterall.interval_list')
        nint = methods.import_interval_list(interval_file).count()
        i = 0
        with open(interval_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nint, i)

    def test_import_bed(self):
        bed_file = test_file('example1.bed')
        nbed = methods.import_bed(bed_file).count()
        i = 0
        with open(bed_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    try:
                        int(line.split()[0])
                        i += 1
                    except:
                        pass
        self.assertEqual(nbed, i)

    def test_import_fam(self):
        fam_file = test_file('sample.fam')
        nfam = methods.import_fam(fam_file).count()
        i = 0
        with open(fam_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nfam, i)

    def test_export_plink(self):
        ds = self.get_dataset()
        
        methods.export_plink(ds, '/tmp/plink_example', id = ds.s)
        
        methods.export_plink(ds, '/tmp/plink_example2', id = ds.s, fam_id = ds.s, pat_id = "nope",
                             mat_id = "nada", is_female = True, is_case = False)
        
        methods.export_plink(ds, '/tmp/plink_example3', id = ds.s, fam_id = ds.s, pat_id = "nope",
                             mat_id = "nada", is_female = True, quant_pheno = ds.s.length().to_float64())
        
        self.assertRaises(ValueError, lambda: methods.export_plink(ds, '/tmp/plink_example', is_case = True, quant_pheno = 0.0))
        
        self.assertRaises(ValueError, lambda: methods.export_plink(ds, '/tmp/plink_example', foo = 0.0))
        
        self.assertRaises(TypeError, lambda: methods.export_plink(ds, '/tmp/plink_example', is_case = 0.0))
        
        # FIXME still resolving: these should throw an error due to unexpected row / entry indicies, still looking into why a more cryptic error is being thrown
        # self.assertRaises(ExpressionException, lambda: methods.export_plink(ds, '/tmp/plink_example', id = ds.v.contig))
        # self.assertRaises(ExpressionException, lambda: methods.export_plink(ds, '/tmp/plink_example', id = ds.GT))
        
    def test_tdt(self):
        pedigree = Pedigree.read(test_file('tdt.fam'))
        tdt_tab = (methods.tdt(
            methods.split_multi_hts(methods.import_vcf(test_file('tdt.vcf'), min_partitions=4)),
            pedigree))

        truth = methods.import_table(
            test_file('tdt_results.tsv'),
            types={'POSITION': TInt32(), 'T': TInt32(), 'U': TInt32(),
                   'Chi2': TFloat64(), 'Pval': TFloat64()})
        truth = (truth
            .transmute(v = functions.variant(truth.CHROM, truth.POSITION, truth.REF, [truth.ALT]))
            .key_by('v'))
        
        if tdt_tab.count() != truth.count():
           self.fail('Result has {} rows but should have {} rows'.format(tdt_tab.count(), truth.count()))

        bad = (tdt_tab.filter(functions.is_nan(tdt_tab.pval), keep=False)
            .join(truth.filter(functions.is_nan(truth.Pval), keep=False), how='outer'))
        
        bad = bad.filter(~(
            (bad.t == bad.T) &
            (bad.u == bad.U) &
            ((bad.chi2 - bad.Chi2).abs() < 0.001) &
            ((bad.pval - bad.Pval).abs() < 0.001)))
        
        if bad.count() != 0:
            bad.order_by(asc(bad.v)).show()
            self.fail('Found rows in violation of the predicate (see show output)')

    def test_maximal_independent_set(self):
        # prefer to remove nodes with higher index
        t = Table.range(10)
        graph = t.select(i = t.idx, j = t.idx + 10)
        mis = methods.maximal_independent_set(graph.i, graph.j, lambda l, r: l - r)
        self.assertEqual(sorted(mis), range(0, 10))

    def test_filter_alleles(self):
        # poor man's Gen
        paths = ['src/test/resources/sample.vcf',
                 'src/test/resources/multipleChromosomes.vcf',
                 'src/test/resources/sample2.vcf']
        for path in paths:
            ds = methods.import_vcf(path)
            self.assertEqual(
                methods.FilterAlleles(functions.range(0, ds.v.num_alt_alleles()).map(lambda i: False))
                .filter()
                .count_rows(), 0)
            self.assertEqual(
                methods.FilterAlleles(functions.range(0, ds.v.num_alt_alleles()).map(lambda i: True))
                .filter()
                .count_rows(), ds.count_rows())

    def test_ld_prune(self):
        ds = methods.split_multi_hts(
            methods.import_vcf(test_file('sample.vcf')))
        methods.ld_prune(ds, 8).count_rows()

    def test_min_rep(self):
        # FIXME actually test
        ds = self.get_dataset()
        methods.min_rep(ds)

    def test_filter_intervals(self):
        ds = self.get_dataset()
        self.assertEqual(
            methods.filter_intervals(ds, Interval.parse('20:10639222-10644705')).count_rows(), 3)
