from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import HailContext
from hail.genetics import *

hc = None

def setUpModule():
    global hc
    hc = HailContext()  # master = 'local[2]')

def tearDownModule():
    global hc
    hc.stop()
    hc = None

class Tests(unittest.TestCase):
    def test_classes(self):
        v = Variant.parse('1:100:A:T')

        self.assertEqual(v, Variant('1', 100, 'A', 'T'))
        self.assertEqual(v, Variant(1, 100, 'A', ['T']))
        self.assertEqual(v.reference_genome, hc.default_reference)

        v2 = Variant.parse('1:100:A:T,C')

        self.assertEqual(v2, Variant('1', 100, 'A', ['T', 'C']))

        l = Locus.parse('1:100')

        self.assertEqual(l, Locus('1', 100))
        self.assertEqual(l, Locus(1, 100))
        self.assertEqual(l.reference_genome, hc.default_reference)

        self.assertEqual(l, v.locus())

        self.assertEqual(v2.num_alt_alleles(), 2)
        self.assertFalse(v2.is_biallelic())
        self.assertTrue(v.is_biallelic())
        self.assertEqual(v.alt_allele(), AltAllele('A', 'T'))
        self.assertEqual(v.allele(0), 'A')
        self.assertEqual(v.allele(1), 'T')
        self.assertEqual(v2.num_alleles(), 3)
        self.assertEqual(v.alt(), 'T')
        self.assertEqual(v2.alt_alleles[0], AltAllele('A', 'T'))
        self.assertEqual(v2.alt_alleles[1], AltAllele('A', 'C'))

        self.assertTrue(v2.is_autosomal_or_pseudoautosomal())
        self.assertTrue(v2.is_autosomal())
        self.assertFalse(v2.is_mitochondrial())
        self.assertFalse(v2.in_X_PAR())
        self.assertFalse(v2.in_Y_PAR())
        self.assertFalse(v2.in_X_non_PAR())
        self.assertFalse(v2.in_Y_non_PAR())

        aa1 = AltAllele('A', 'T')
        aa2 = AltAllele('A', 'AAA')
        aa3 = AltAllele('TTTT', 'T')
        aa4 = AltAllele('AT', 'TC')
        aa5 = AltAllele('AAAT', 'AAAA')

        self.assertEqual(aa1.num_mismatch(), 1)
        self.assertEqual(aa5.num_mismatch(), 1)
        self.assertEqual(aa4.num_mismatch(), 2)

        c1, c2 = aa5.stripped_snp()

        self.assertEqual(c1, 'T')
        self.assertEqual(c2, 'A')
        self.assertTrue(aa1.is_SNP())
        self.assertTrue(aa5.is_SNP())
        self.assertTrue(aa4.is_MNP())
        self.assertTrue(aa2.is_insertion())
        self.assertTrue(aa3.is_deletion())
        self.assertTrue(aa3.is_indel())
        self.assertTrue(aa1.is_transversion())

        interval = Interval.parse('1:100-110')

        self.assertEqual(interval, Interval.parse('1:100-1:110'))
        self.assertEqual(interval, Interval(Locus("1", 100), Locus("1", 110)))
        self.assertTrue(interval.contains(Locus("1", 100)))
        self.assertTrue(interval.contains(Locus("1", 109)))
        self.assertFalse(interval.contains(Locus("1", 110)))
        self.assertEqual(interval.reference_genome, hc.default_reference)

        interval2 = Interval.parse("1:109-200")
        interval3 = Interval.parse("1:110-200")
        interval4 = Interval.parse("1:90-101")
        interval5 = Interval.parse("1:90-100")

        self.assertTrue(interval.overlaps(interval2))
        self.assertTrue(interval.overlaps(interval4))
        self.assertFalse(interval.overlaps(interval3))
        self.assertFalse(interval.overlaps(interval5))

        c_hom_ref = Call(0)

        self.assertEqual(c_hom_ref.gt, 0)
        self.assertEqual(c_hom_ref.num_alt_alleles(), 0)
        self.assertTrue(c_hom_ref.one_hot_alleles(2) == [2, 0])
        self.assertTrue(c_hom_ref.one_hot_genotype(3) == [1, 0, 0])
        self.assertTrue(c_hom_ref.is_hom_ref())
        self.assertFalse(c_hom_ref.is_het())
        self.assertFalse(c_hom_ref.is_hom_var())
        self.assertFalse(c_hom_ref.is_non_ref())
        self.assertFalse(c_hom_ref.is_het_non_ref())
        self.assertFalse(c_hom_ref.is_het_ref())

        gr = GenomeReference.GRCh37()
        self.assertEqual(gr.name, "GRCh37")
        self.assertEqual(gr.contigs[0], "1")
        self.assertListEqual(gr.x_contigs, ["X"])
        self.assertListEqual(gr.y_contigs, ["Y"])
        self.assertListEqual(gr.mt_contigs, ["MT"])
        self.assertEqual(gr.par[0], Interval.parse("X:60001-2699521"))
        self.assertEqual(gr.contig_length("1"), 249250621)

        name = "test"
        contigs = ["1", "X", "Y", "MT"]
        lengths = {"1": 10000, "X": 2000, "Y": 4000, "MT": 1000}
        x_contigs = ["X"]
        y_contigs = ["Y"]
        mt_contigs = ["MT"]
        par = [("X", 5, 1000)]

        gr2 = GenomeReference(name, contigs, lengths, x_contigs, y_contigs, mt_contigs, par)
        self.assertEqual(gr2.name, name)
        self.assertListEqual(gr2.contigs, contigs)
        self.assertListEqual(gr2.x_contigs, x_contigs)
        self.assertListEqual(gr2.y_contigs, y_contigs)
        self.assertListEqual(gr2.mt_contigs, mt_contigs)
        self.assertEqual(gr2.par, [Interval.parse("X:5-1000", gr2)])
        self.assertEqual(gr2.contig_length("1"), 10000)
        self.assertDictEqual(gr2.lengths, lengths)
        gr2.write("/tmp/my_gr.json")

        gr3 = GenomeReference.read("src/test/resources/fake_ref_genome.json")
        self.assertEqual(gr3.name, "my_reference_genome")


    def test_pedigree(self):

        ped = Pedigree.read('src/test/resources/sample.fam')
        ped.write('/tmp/sample_out.fam')
        ped2 = Pedigree.read('/tmp/sample_out.fam')
        self.assertEqual(ped, ped2)
        print(ped.trios[:5])
        print(ped.complete_trios)

        t1 = Trio('kid1', father='dad1', is_female=True)
        t2 = Trio('kid1', father='dad1', is_female=True)

        self.assertEqual(t1, t2)

        self.assertEqual(t1.fam, None)
        self.assertEqual(t1.proband, 'kid1')
        self.assertEqual(t1.father, 'dad1')
        self.assertEqual(t1.mother, None)
        self.assertEqual(t1.is_female, True)
        self.assertEqual(t1.is_complete(), False)
        self.assertEqual(t1.is_female, True)
        self.assertEqual(t1.is_male, False)
