import unittest

from hail.genetics import *
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def test_pedigree(self):
        ped = Pedigree.read(resource('sample.fam'))
        ped.write('/tmp/sample_out.fam')
        ped2 = Pedigree.read('/tmp/sample_out.fam')
        self.assertEqual(ped, ped2)
        print(ped.trios[:5])
        print(ped.complete_trios())

        t1 = Trio('kid1', pat_id='dad1', is_female=True)
        t2 = Trio('kid1', pat_id='dad1', is_female=True)

        self.assertEqual(t1, t2)

        self.assertEqual(t1.fam_id, None)
        self.assertEqual(t1.s, 'kid1')
        self.assertEqual(t1.pat_id, 'dad1')
        self.assertEqual(t1.mat_id, None)
        self.assertEqual(t1.is_female, True)
        self.assertEqual(t1.is_complete(), False)
        self.assertEqual(t1.is_female, True)
        self.assertEqual(t1.is_male, False)
