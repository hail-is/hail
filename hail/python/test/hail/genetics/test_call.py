import unittest

from hail.genetics import *
from ..helpers import *


class Tests(unittest.TestCase):
    def test_hom_ref(self):
        c_hom_ref = Call([0, 0])
        self.assertEqual(c_hom_ref.alleles, [0, 0])
        self.assertEqual(c_hom_ref.ploidy, 2)
        self.assertFalse(c_hom_ref.phased)
        self.assertFalse(c_hom_ref.is_haploid())
        self.assertTrue(c_hom_ref.is_diploid())
        self.assertEqual(c_hom_ref.n_alt_alleles(), 0)
        self.assertTrue(c_hom_ref.one_hot_alleles(2) == [2, 0])
        self.assertTrue(c_hom_ref.is_hom_ref())
        self.assertFalse(c_hom_ref.is_het())
        self.assertFalse(c_hom_ref.is_hom_var())
        self.assertFalse(c_hom_ref.is_non_ref())
        self.assertFalse(c_hom_ref.is_het_non_ref())
        self.assertFalse(c_hom_ref.is_het_ref())
        self.assertTrue(c_hom_ref.unphased_diploid_gt_index() == 0)

    def test_het(self):
        c_het_phased = Call([1, 0], phased=True)
        self.assertEqual(c_het_phased.alleles, [1, 0])
        self.assertEqual(c_het_phased.ploidy, 2)
        self.assertTrue(c_het_phased.phased)
        self.assertFalse(c_het_phased.is_haploid())
        self.assertTrue(c_het_phased.is_diploid())
        self.assertEqual(c_het_phased.n_alt_alleles(), 1)
        self.assertTrue(c_het_phased.one_hot_alleles(2) == [1, 1])
        self.assertFalse(c_het_phased.is_hom_ref())
        self.assertTrue(c_het_phased.is_het())
        self.assertFalse(c_het_phased.is_hom_var())
        self.assertTrue(c_het_phased.is_non_ref())
        self.assertFalse(c_het_phased.is_het_non_ref())
        self.assertTrue(c_het_phased.is_het_ref())

    def test_hom_var(self):
        c_hom_var = Call([1, 1])
        self.assertEqual(c_hom_var.alleles, [1, 1])
        self.assertEqual(c_hom_var.ploidy, 2)
        self.assertFalse(c_hom_var.phased)
        self.assertFalse(c_hom_var.is_haploid())
        self.assertTrue(c_hom_var.is_diploid())
        self.assertEqual(c_hom_var.n_alt_alleles(), 2)
        self.assertTrue(c_hom_var.one_hot_alleles(2) == [0, 2])
        self.assertFalse(c_hom_var.is_hom_ref())
        self.assertFalse(c_hom_var.is_het())
        self.assertTrue(c_hom_var.is_hom_var())
        self.assertTrue(c_hom_var.is_non_ref())
        self.assertFalse(c_hom_var.is_het_non_ref())
        self.assertFalse(c_hom_var.is_het_ref())
        self.assertTrue(c_hom_var.unphased_diploid_gt_index() == 2)

    def test_haploid(self):
        c_haploid = Call([2], phased=True)
        self.assertEqual(c_haploid.alleles, [2])
        self.assertEqual(c_haploid.ploidy, 1)
        self.assertTrue(c_haploid.phased)
        self.assertTrue(c_haploid.is_haploid())
        self.assertFalse(c_haploid.is_diploid())
        self.assertEqual(c_haploid.n_alt_alleles(), 1)
        self.assertTrue(c_haploid.one_hot_alleles(3) == [0, 0, 1])
        self.assertFalse(c_haploid.is_hom_ref())
        self.assertFalse(c_haploid.is_het())
        self.assertTrue(c_haploid.is_hom_var())
        self.assertTrue(c_haploid.is_non_ref())
        self.assertFalse(c_haploid.is_het_non_ref())
        self.assertFalse(c_haploid.is_het_ref())

    def test_zeroploid(self):
        c_zeroploid = Call([])
        self.assertEqual(c_zeroploid.alleles, [])
        self.assertEqual(c_zeroploid.ploidy, 0)
        self.assertFalse(c_zeroploid.phased)
        self.assertFalse(c_zeroploid.is_haploid())
        self.assertFalse(c_zeroploid.is_diploid())
        self.assertEqual(c_zeroploid.n_alt_alleles(), 0)
        self.assertTrue(c_zeroploid.one_hot_alleles(3) == [0, 0, 0])
        self.assertFalse(c_zeroploid.is_hom_ref())
        self.assertFalse(c_zeroploid.is_het())
        self.assertFalse(c_zeroploid.is_hom_var())
        self.assertFalse(c_zeroploid.is_non_ref())
        self.assertFalse(c_zeroploid.is_het_non_ref())
        self.assertFalse(c_zeroploid.is_het_ref())

        self.assertRaisesRegex(
            NotImplementedError, "Calls with greater than 2 alleles are not supported.", Call, [1, 1, 1, 1]
        )


def test_call_rich_comparison():
    val = Call([0, 0])
    expr = hl.call(0, 0)

    assert hl.eval(val == expr)
    assert hl.eval(expr == val)
