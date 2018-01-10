from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail2 import *

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
        dataset = methods.rename_duplicates(dataset)
        renamed_samples = dataset.cols_table()
        renamed_ids = renamed_samples.select(renamed_samples.s).collect()
        self.assertTrue(len(set(renamed_ids)), len(renamed_ids))

