import unittest
import hail as hl
from hail.genetics import *
from .utils import resource, startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

class Tests(unittest.TestCase):
    def test_classes(self):
        l = Locus.parse('1:100')

        self.assertEqual(l, Locus('1', 100))
        self.assertEqual(l, Locus(1, 100))
        self.assertEqual(l.reference_genome, hl.default_reference())

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

        self.assertRaisesRegex(NotImplementedError,
                               "Calls with greater than 2 alleles are not supported.",
                               Call,
                               [1, 1, 1, 1])


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

    def test_reference_genome(self):
        rg = hl.get_reference('GRCh37')
        self.assertEqual(rg.name, "GRCh37")
        self.assertEqual(rg.contigs[0], "1")
        self.assertListEqual(rg.x_contigs, ["X"])
        self.assertListEqual(rg.y_contigs, ["Y"])
        self.assertListEqual(rg.mt_contigs, ["MT"])
        self.assertEqual(rg.par[0], hl.parse_locus_interval("X:60001-2699521").value)
        self.assertEqual(rg.contig_length("1"), 249250621)

        name = "test"
        contigs = ["1", "X", "Y", "MT"]
        lengths = {"1": 10000, "X": 2000, "Y": 4000, "MT": 1000}
        x_contigs = ["X"]
        y_contigs = ["Y"]
        mt_contigs = ["MT"]
        par = [("X", 5, 1000)]

        gr2 = ReferenceGenome(name, contigs, lengths, x_contigs, y_contigs, mt_contigs, par)
        self.assertEqual(gr2.name, name)
        self.assertListEqual(gr2.contigs, contigs)
        self.assertListEqual(gr2.x_contigs, x_contigs)
        self.assertListEqual(gr2.y_contigs, y_contigs)
        self.assertListEqual(gr2.mt_contigs, mt_contigs)
        self.assertEqual(gr2.par, [hl.parse_locus_interval("X:5-1000", gr2).value])
        self.assertEqual(gr2.contig_length("1"), 10000)
        self.assertDictEqual(gr2.lengths, lengths)
        gr2.write("/tmp/my_gr.json")

    def test_reference_genome_sequence(self):
        gr3 = ReferenceGenome.read(resource("fake_ref_genome.json"))
        self.assertEqual(gr3.name, "my_reference_genome")
        self.assertFalse(gr3.has_sequence())

        gr4 = ReferenceGenome.from_fasta_file("test_rg", resource("fake_reference.fasta"), resource("fake_reference.fasta.fai"),
                                              mt_contigs=["b", "c"], x_contigs=["a"])
        self.assertTrue(gr4.has_sequence())
        self.assertTrue(gr4.x_contigs == ["a"])

        t = hl.import_table(resource("fake_reference.tsv"), impute=True)
        self.assertTrue(t.all(hl.get_sequence(t.contig, t.pos, reference_genome=gr4) == t.base))

        l = hl.locus("a", 7, gr4)
        self.assertTrue(l.sequence_context(before=3, after=3).value == "TTTCGAA")

    def test_reference_genome_liftover(self):
        grch37 = hl.get_reference('GRCh37')
        grch38 = hl.get_reference('GRCh38')

        self.assertTrue(not grch37.has_liftover('GRCh38') and not grch38.has_liftover('GRCh37'))
        grch37.add_liftover(resource('grch37_to_grch38_chr20.over.chain.gz'), 'GRCh38')
        grch38.add_liftover(resource('grch38_to_grch37_chr20.over.chain.gz'), 'GRCh37')
        self.assertTrue(grch37.has_liftover('GRCh38') and grch38.has_liftover('GRCh37'))

        ds = hl.import_vcf(resource('sample.vcf'))
        t = ds.annotate_rows(liftover = hl.liftover(hl.liftover(ds.locus, 'GRCh38'), 'GRCh37')).rows()
        self.assertTrue(t.all(t.locus == t.liftover))
        
        null_locus = hl.null(hl.tlocus('GRCh38'))

        rows = [
            {'l37': hl.locus('20', 1, 'GRCh37'), 'l38': null_locus},
            {'l37': hl.locus('20', 60000, 'GRCh37'), 'l38': null_locus},
            {'l37': hl.locus('20', 60001, 'GRCh37'), 'l38': hl.locus('chr20', 79360, 'GRCh38')},
            {'l37': hl.locus('20', 278686, 'GRCh37'), 'l38': hl.locus('chr20', 298045, 'GRCh38')},
            {'l37': hl.locus('20', 278687, 'GRCh37'), 'l38': hl.locus('chr20', 298046, 'GRCh38')},
            {'l37': hl.locus('20', 278688, 'GRCh37'), 'l38': null_locus},
            {'l37': hl.locus('20', 278689, 'GRCh37'), 'l38': null_locus},
            {'l37': hl.locus('20', 278690, 'GRCh37'), 'l38': null_locus},
            {'l37': hl.locus('20', 278691, 'GRCh37'), 'l38': hl.locus('chr20', 298047, 'GRCh38')},
            {'l37': hl.locus('20', 37007586, 'GRCh37'), 'l38': hl.locus('chr12', 32563117, 'GRCh38')},
            {'l37': hl.locus('20', 62965520, 'GRCh37'), 'l38': hl.locus('chr20', 64334167, 'GRCh38')},
            {'l37': hl.locus('20', 62965521, 'GRCh37'), 'l38': null_locus}
        ]
        schema = hl.tstruct(l37=hl.tlocus(grch37), l38=hl.tlocus(grch38))
        t = hl.Table.parallelize(rows, schema)
        self.assertTrue(t.all(hl.cond(hl.is_defined(t.l38),
                                      hl.liftover(t.l37, 'GRCh38') == t.l38,
                                      hl.is_missing(hl.liftover(t.l37, 'GRCh38')))))

        t = t.filter(hl.is_defined(t.l38))
        self.assertTrue(t.count() == 6)

        t = t.key_by('l38')
        t.count()
        self.assertTrue(list(t.key) == ['l38'])

        null_locus_interval = hl.null(hl.tinterval(hl.tlocus('GRCh38')))
        rows = [
            {'i37': hl.locus_interval('20', 1, 60000, True, False, 'GRCh37'), 'i38': null_locus_interval},
            {'i37': hl.locus_interval('20', 60001, 82456, True, True, 'GRCh37'), 'i38': hl.locus_interval('chr20', 79360, 101815, True, True, 'GRCh38')}
        ]
        schema = hl.tstruct(i37=hl.tinterval(hl.tlocus(grch37)), i38=hl.tinterval(hl.tlocus(grch38)))
        t = hl.Table.parallelize(rows, schema)
        self.assertTrue(t.all(hl.liftover(t.i37, 'GRCh38') == t.i38))
