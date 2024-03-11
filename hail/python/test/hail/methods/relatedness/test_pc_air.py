import unittest

import numpy as np

import hail as hl
from hail.methods.relatedness.pc_air import _partition_samples
from test.hail.helpers import resource


class Tests(unittest.TestCase):
    def test_partition_samples(self):
        plink_path = resource('fastlmmTest')
        mt = hl.import_plink(
            bed=f'{plink_path}.bed', bim=f'{plink_path}.bim', fam=f'{plink_path}.fam', reference_genome=None
        )
        table = _partition_samples(mt.GT, 0.05)
        unrelated = set(table.filter(table.is_in_unrelated).s.collect())
        related = set(table.filter(~table.is_in_unrelated).s.collect())
        expected_related = {
            'cid93P0',
            'cid389P1',
            'cid392P1',
            'cid104P0',
            'cid490P1',
            'cid47P0',
            'cid16P0',
            'cid96P0',
            'cid8P0',
            'cid62P0',
            'cid462P1',
            'cid381P1',
            'cid110P0',
            'cid396P1',
            'cid5P0',
            'cid390P1',
            'cid382P1',
            'cid448P1',
            'cid391P1',
            'cid467P1',
            'cid73P0',
            'cid121P0',
            'cid466P1',
            'cid32P0',
        }
        expected_unrelated = set(mt.col_key.s.collect()) - expected_related

        self.assertEqual(expected_related, related)
        self.assertEqual(expected_unrelated, unrelated)

        pairs = hl.king(mt.GT).entries().to_pandas()

        # Confirm that all pairs within the unrelated set are indeed unrelated
        max_unrelated_phi = pairs[
            (pairs['s'] != pairs['s_1']) & pairs['s'].isin(unrelated) & pairs['s_1'].isin(unrelated)
        ]['phi'].max()
        self.assertLess(max_unrelated_phi, 0.05)

    def test_pc_air(self):
        plink_path = resource('fastlmmTest')
        mt = hl.import_plink(
            bed=f'{plink_path}.bed', bim=f'{plink_path}.bim', fam=f'{plink_path}.fam', reference_genome=None
        )
        eigenvalues, scores, loadings = hl.pc_air(mt.GT, relatedness_threshold=0.05, _seed=0)
        expected_eigenvalues = [
            28694.50283292,
            3436.72398636,
            3375.03438979,
            3338.77060759,
            3311.3869508,
            3284.42217958,
            3251.83577347,
            3227.64678472,
            3194.68304824,
            3181.67907063,
        ]
        self.assertTrue(np.allclose(expected_eigenvalues, eigenvalues))
        self.assertEqual(250, scores.count())
        self.assertEqual(10, len(scores.scores.take(1)[0]))
        self.assertEqual(2000, loadings.count())
        self.assertEqual(10, len(loadings.loadings.take(1)[0]))

        partition_table = _partition_samples(mt.GT, 0.05)
        unrelated_table = partition_table.filter(partition_table.is_in_unrelated)
        mt = mt.semi_join_cols(unrelated_table)
        _, _, unrelated_loadings = hl._hwe_normalized_blanczos(mt.GT, compute_loadings=True, _seed=0)

        # Confirm that the loadings are the same
        loadings = np.array(loadings.loadings.collect())
        unrelated_loadings = np.array(unrelated_loadings.loadings.collect())
        self.assertTrue(np.allclose(loadings, unrelated_loadings))

        # Read expected scores from file
        expected_scores = hl.read_table(resource('pc-air-expected-scores.ht'))
        expected_scores = np.array(expected_scores.scores.collect())
        scores = np.array(scores.scores.collect())
        self.assertTrue(np.allclose(expected_scores, scores))
