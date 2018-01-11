from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail2 import *
from subprocess import call as syscall
import numpy as np
from struct import unpack
import hail.utils as utils

hc = None


def setUpModule():
    global hc
    hc = HailContext()  # master = 'local[2]')


def tearDownModule():
    global hc
    hc.stop()
    hc = None


class Tests(unittest.TestCase):
    _dataset = None

    def get_dataset(self):
        if Tests._dataset is None:
            Tests._dataset = hc.import_vcf('src/test/resources/sample.vcf').to_hail1().split_multi_hts().to_hail2()
        return Tests._dataset

    def test_ld_matrix(self):
        dataset = self.get_dataset()

        ldm = methods.ld_matrix(dataset, force_local=True)

    def test_linreg(self):
        dataset = hc.import_vcf('src/test/resources/regressionLinear.vcf')

        phenos = hc.import_table('src/test/resources/regressionLinear.pheno',
                                 types={'Pheno': TFloat64()},
                                 key='Sample')
        covs = hc.import_table('src/test/resources/regressionLinear.cov',
                               types={'Cov1': TFloat64(), 'Cov2': TFloat64()},
                               key='Sample')

        dataset = dataset.annotate_cols(pheno=phenos[dataset.s], cov = covs[dataset.s])
        dataset = methods.linreg(dataset,
                         ys=dataset.pheno,
                         x=dataset.GT.num_alt_alleles(),
                         covariates=[dataset.cov.Cov1, dataset.cov.Cov2 + 1 - 1])

        dataset.count_rows()

    def test_trio_matrix(self):
        ped = Pedigree.read('src/test/resources/triomatrix.fam')
        from hail import KeyTable
        fam_table = KeyTable.import_fam('src/test/resources/triomatrix.fam').to_hail2()

        dataset = hc.import_vcf('src/test/resources/triomatrix.vcf')
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
                    assert(len(r) == 2)
                    ids.append(r[1])
            return ids

        def load_rel(ns, path):
            rel = np.zeros((ns, ns))
            with hadoop_read(path) as f:
                for i,l in enumerate(f):
                    for j,n in enumerate(map(float, l.strip().split('\t'))):
                        rel[i,j] = n
                    assert(j == i)
                assert(i == ns - 1)
            return rel

        def load_grm(ns, nv, path):
            m = np.zeros((ns, ns))
            with utils.hadoop_read(path) as f:
                i = 0
                for l in f:
                    row = l.strip().split('\t')
                    assert(int(row[2]) == nv)
                    m[int(row[0])-1, int(row[2])-1] = float(row[3])
                    i += 1

                assert(i == ns * (ns + 1) / 2 - 1)
            return m

        def load_bin(ns, path):
            m = np.zeros((ns, ns))
            with utils.hadoop_read_binary(path) as f:
                for i in range(ns):
                    for j in range(i):
                        m[i, j] = unpack('<f', bytearray(f.read(4)))
                assert(len(f.read()) == 0)
            return m

        b_file = utils.new_temp_file(prefix="plink")
        rel_file = utils.new_temp_file(prefix="test", suffix=".rel")
        rel_id_file = utils.new_temp_file(prefix="test", suffix=".rel.id")
        grm_file = utils.new_temp_file(prefix="test", suffix=".grm")
        grm_bin_file = utils.new_temp_file(prefix="test", suffix=".grm.bin")
        grm_nbin_file = utils.new_temp_file(prefix="test", suffix=".grm.N.bin")

        dataset = self.get_dataset()
        dataset.to_hail1().export_plink(b_file)

        sample_ids = [row.s for row in dataset.cols_table().select('s').collect()]
        n_samples = dataset.count_cols()
        n_variants = dataset.count_rows()
        assert(n_variants > 0)

        grm = methods.grm(dataset)
        grm.export_id_file(rel_id_file)

        ############
        ### rel

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-rel --out {}'''
                .format(utils.get_URI(b_file), utils.get_URI(p_file)), shell=True)
        assert(load_id_file(p_file + ".rel.id") == sample_ids)

        grm.export_rel(rel_file)
        assert(load_id_file(rel_id_file) == sample_ids)
        assert(np.allclose(load_rel(n_samples, p_file + ".rel"),
                           load_rel(n_samples, rel_file),
                           atol=tolerance))

        ############
        ### gcta-grm

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-grm-gz --out {}'''
                .format(utils.get_URI(b_file), utils.get_URI(p_file)), shell=True)
        assert(load_id_file(p_file + ".grm.id") == sample_ids)

        grm.export_gcta_grm(grm_file)
        assert(np.allclose(load_grm(n_samples, n_variants, p_file + ".grm.gz"),
                           load_grm(n_samples, n_variants, grm_file),
                           atol=tolerance))

        ############
        ### gcta-grm-bin

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-grm-bin --out {}'''
                .format(utils.get_URI(b_file), utils.get_URI(p_file)), shell=True)

        assert(load_id_file(p_file + ".grm.id") == sample_ids)

        grm.export_gcta_grm_bin(grm_bin_file, grm_nbin_file)

        assert(np.allclose(load_bin(n_samples, p_file + ".grm.bin"),
                           load_bin(n_samples, grm_bin_file),
                           atol=tolerance) and
               np.allclose(load_bin(n_samples, p_file + ".grm.N.bin"),
                           load_bin(n_samples, grm_nbin_file),
                           atol=tolerance))

    def test_pca(self):
        dataset = hc._hc1.balding_nichols_model(3, 100, 100).to_hail2()
        eigenvalues, scores, loadings = methods.pca(dataset.GT.num_alt_alleles(), k=2, compute_loadings=True)

        self.assertEqual(len(eigenvalues), 2)
        self.assertTrue(isinstance(scores, Table))
        self.assertEqual(scores.count(), 100)
        self.assertTrue(isinstance(loadings, Table))
        self.assertEqual(loadings.count(), 100)

        _, _, loadings = methods.pca(dataset.GT.num_alt_alleles(), k=2, compute_loadings=False)
        self.assertEqual(loadings, None)

    def test_rename_duplicates(self):
        dataset = self.get_dataset() # FIXME - want to rename samples with same id
        renamed_ids = methods.rename_duplicates(dataset).cols_table().select('s').collect()
        self.assertTrue(len(set(renamed_ids)), len(renamed_ids))

    def test_split_multi_hts(self):
        ds1 = hc.import_vcf('src/test/resources/split_test.vcf')
        ds1 = methods.split_multi_hts(ds1)
        ds2 = hc.import_vcf('src/test/resources/split_test_b.vcf')
        self.assertEqual(ds1.aggregate_entries(foo = agg.product((ds1.wasSplit == (ds1.v.start != 1180)).to_int32())).foo, 1)
        ds1 = ds1.drop('wasSplit','aIndex')
        # required python3
        # self.assertTrue(ds1._same(ds2))

    def test_mendel_errors(self):
        dataset = self.get_dataset()
        men, fam, ind, var = methods.mendel_errors(dataset, Pedigree.read('src/test/resources/sample.fam'))
        men.select('fid', 's', 'code')
        fam.select('father', 'nChildren')
        self.assertEqual(ind.key, ['s'])
        self.assertEqual(var.key, ['v'])
        dataset.annotate_rows(mendel=var[dataset.v]).count_rows()
