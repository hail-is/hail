import unittest

import hail as hl
from hail.methods.relatedness.pc_air import _partition_samples

from test.hail.helpers import resource


class Tests(unittest.TestCase):
    def test_partition_samples(self):
        plink_path = resource('fastlmmTest')
        mt = hl.import_plink(
            bed=f'{plink_path}.bed', bim=f'{plink_path}.bim', fam=f'{plink_path}.fam', reference_genome=None
        )
        unrelated, related = _partition_samples(mt.GT, 0.05)
        unrelated = set(map(lambda x: x.s, unrelated))
        related = set(map(lambda x: x.s, related))
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
