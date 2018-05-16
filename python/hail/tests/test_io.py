import hail as hl
from .utils import resource, startTestHailContext, stopTestHailContext
from hail.utils import new_temp_file, FatalError
import unittest

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

class VCFTests(unittest.TestCase):
    def test_info_char(self):
        self.assertEqual(hl.import_vcf(resource('infochar.vcf')).count_rows(), 1)

    def test_glob(self):
        full = hl.import_vcf(resource('sample.vcf'))
        parts = hl.import_vcf(resource('samplepart*.vcf'))
        self.assertTrue(parts._same(full))

    def test_undeclared_info(self):
        mt = hl.import_vcf(resource('undeclaredinfo.vcf'))

        rows = mt.rows()
        self.assertTrue(rows.all(hl.is_defined(rows.info)))

        info_type = mt.row.dtype['info']
        self.assertTrue('InbreedingCoeff' in info_type)
        self.assertFalse('undeclared' in info_type)
        self.assertFalse('undeclaredFlag' in info_type)

    def test_malformed(self):
        with self.assertRaisesRegex(FatalError, "invalid character"):
            mt = hl.import_vcf(resource('malformed.vcf'))
            mt._force_count_rows()

    def test_not_identical_headers(self):
        t = new_temp_file('vcf')
        mt = hl.import_vcf(resource('sample.vcf'))
        hl.export_vcf(mt.filter_cols((mt.s != "C1048::HG02024") & (mt.s != "HG00255")), t)
        
        with self.assertRaisesRegex(FatalError, 'invalid sample ids'):
            (hl.import_vcf([resource('sample.vcf'), t])
             ._force_count_rows())

    def test_haploid(self):
        expected = hl.Table.parallelize(
            [hl.struct(locus = hl.locus("X", 16050036), s = "C1046::HG02024",
                       GT = hl.call(0, 0), AD = [10, 0], GQ = 44),
             hl.struct(locus = hl.locus("X", 16050036), s = "C1046::HG02025",
                       GT = hl.call(1), AD = [0, 6], GQ = 70),
             hl.struct(locus = hl.locus("X", 16061250), s = "C1046::HG02024",
                       GT = hl.call(2, 2), AD = [0, 0, 11], GQ = 33),
             hl.struct(locus = hl.locus("X", 16061250), s = "C1046::HG02025",
                       GT = hl.call(2), AD = [0, 0, 9], GQ = 24)],
            key=['locus', 's'])

        mt = hl.import_vcf(resource('haploid.vcf'))
        entries = mt.entries()
        entries = entries.key_by('locus', 's')
        entries = entries.select('GT', 'AD', 'GQ')
        self.assertTrue(entries._same(expected))

    def test_call_fields(self):
        expected = hl.Table.parallelize(
            [hl.struct(locus = hl.locus("X", 16050036), s = "C1046::HG02024",
                       GT = hl.call(0, 0), GTA = hl.null(hl.tcall), GTZ = hl.call(0, 1)),
             hl.struct(locus = hl.locus("X", 16050036), s = "C1046::HG02025",
                       GT = hl.call(1), GTA = hl.null(hl.tcall), GTZ = hl.call(0)),
             hl.struct(locus = hl.locus("X", 16061250), s = "C1046::HG02024",
                       GT = hl.call(2, 2), GTA = hl.call(2, 1), GTZ = hl.call(1, 1)),
             hl.struct(locus = hl.locus("X", 16061250), s = "C1046::HG02025",
                       GT = hl.call(2), GTA = hl.null(hl.tcall), GTZ = hl.call(1))],
            key=['locus', 's'])

        mt = hl.import_vcf(resource('generic.vcf'), call_fields=['GT', 'GTA', 'GTZ'])
        entries = mt.entries()
        entries = entries.key_by('locus', 's')
        entries = entries.select('GT', 'GTA', 'GTZ')
        self.assertTrue(entries._same(expected))
