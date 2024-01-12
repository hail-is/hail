import unittest

from hail.genetics import Pedigree, Trio
from hail.utils.java import FatalError

from ..helpers import resource


class Tests(unittest.TestCase):
    def test_trios(self):
        t1 = Trio('kid1', pat_id='dad1', is_female=True)
        t2 = Trio('kid1', pat_id='dad1', is_female=True)
        t3 = Trio('kid1', pat_id='dad1', is_female=False)
        t4 = Trio('kid1', pat_id='dad1', mat_id='mom2', is_female=True)
        t5 = Trio('kid2', mat_id='mom2', is_female=False)
        t6 = Trio('kid2', pat_id='dad2', mat_id="mom2")

        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)
        self.assertNotEqual(t1, t4)

        self.assertEqual(t1.fam_id, None)
        self.assertEqual(t1.s, 'kid1')
        self.assertEqual(t1.pat_id, 'dad1')
        self.assertEqual(t1.mat_id, None)
        self.assertEqual(t1.is_female, True)
        self.assertEqual(t1.is_female, True)
        self.assertEqual(t1.is_male, False)

        self.assertEqual(t1.is_complete(), False)
        self.assertEqual(t4.is_complete(), True)
        self.assertEqual(t5.is_complete(), False)
        self.assertEqual(t6.is_complete(), True)

    def test_pedigree(self):
        ped = Pedigree.read(resource('sample.fam'))
        ped.write('/tmp/sample_out.fam')
        ped2 = Pedigree.read('/tmp/sample_out.fam')
        self.assertEqual(ped, ped2)

        complete_trios = ped.complete_trios()
        self.assertEqual(len(complete_trios), 3)

        with self.assertRaises(FatalError):
            Pedigree.read(resource('duplicate_id.fam'))
