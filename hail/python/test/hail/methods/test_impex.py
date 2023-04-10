import json
import os
import shutil
import unittest

from unittest import mock

from avro.datafile import DataFileReader
from avro.io import DatumReader
from hail.context import TemporaryFilename

import pytest
import hail as hl
from ..helpers import *
from hail import ir
from hail.utils import new_temp_file, FatalError, run_command, uri_path, HailUserError

_FLOAT_INFO_FIELDS = [
    'BaseQRankSum',
    'ClippingRankSum',
    'FS',
    'GQ_MEAN',
    'GQ_STDDEV',
    'HWP',
    'HaplotypeScore',
    'InbreedingCoeff',
    'MQ',
    'MQRankSum',
    'QD',
    'ReadPosRankSum',
    'VQSLOD',
]

_FLOAT_ARRAY_INFO_FIELDS = ['AF', 'MLEAF']


class VCFTests(unittest.TestCase):
    def test_info_char(self):
        self.assertEqual(hl.import_vcf(resource('infochar.vcf')).count_rows(), 1)

    def test_import_export_same(self):
        for i in range(10):
            mt = hl.import_vcf(resource(f'random_vcfs/{i}.vcf.bgz'))
            f1 = new_temp_file(extension='vcf.bgz')
            hl.export_vcf(mt, f1)
            mt2 = hl.import_vcf(f1)
            f2 = new_temp_file(extension='vcf.bgz')
            hl.export_vcf(mt2, f2)
            mt3 = hl.import_vcf(f2)

            assert mt._same(mt2)
            assert mt._same(mt3)

    def test_info_float64(self):
        """Test that floating-point info fields are 64-bit regardless of the entry float type"""
        mt = hl.import_vcf(resource('infochar.vcf'), entry_float_type=hl.tfloat32)
        for f in _FLOAT_INFO_FIELDS:
            self.assertEqual(mt['info'][f].dtype, hl.tfloat64)
        for f in _FLOAT_ARRAY_INFO_FIELDS:
            self.assertEqual(mt['info'][f].dtype, hl.tarray(hl.tfloat64))

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

    def test_can_import_bad_number_flag(self):
        hl.import_vcf(resource('bad_flag_number.vcf')).rows()._force_count()

    def test_malformed(self):
        with self.assertRaisesRegex(FatalError, "invalid character"):
            mt = hl.import_vcf(resource('malformed.vcf'))
            mt._force_count_rows()

    def test_not_identical_headers(self):
        t = new_temp_file(extension='vcf')
        mt = hl.import_vcf(resource('sample.vcf'))
        hl.export_vcf(mt.filter_cols((mt.s != "C1048::HG02024") & (mt.s != "HG00255")), t)

        with self.assertRaisesRegex(FatalError, 'invalid sample IDs'):
            (hl.import_vcf([resource('sample.vcf'), t])
             ._force_count_rows())

    def test_filter(self):
        mt = hl.import_vcf(resource('malformed.vcf'), filter='rs685723')
        mt._force_count_rows()

        mt = hl.import_vcf(resource('sample.vcf'), filter=r'\trs\d+\t')
        assert mt.aggregate_rows(hl.agg.all(hl.is_missing(mt.rsid)))

    def test_find_replace(self):
        mt = hl.import_vcf(resource('sample.vcf'), find_replace=(r'\trs\d+\t', '\t.\t'))
        mt.rows().show()
        assert mt.aggregate_rows(hl.agg.all(hl.is_missing(mt.rsid)))

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
                       GT = hl.call(0, 0), GTA = hl.missing(hl.tcall), GTZ = hl.call(0, 1)),
             hl.struct(locus = hl.locus("X", 16050036), s = "C1046::HG02025",
                       GT = hl.call(1), GTA = hl.missing(hl.tcall), GTZ = hl.call(0)),
             hl.struct(locus = hl.locus("X", 16061250), s = "C1046::HG02024",
                       GT = hl.call(2, 2), GTA = hl.call(2, 1), GTZ = hl.call(1, 1)),
             hl.struct(locus = hl.locus("X", 16061250), s = "C1046::HG02025",
                       GT = hl.call(2), GTA = hl.missing(hl.tcall), GTZ = hl.call(1))],
            key=['locus', 's'])

        mt = hl.import_vcf(resource('generic.vcf'), call_fields=['GT', 'GTA', 'GTZ'])
        entries = mt.entries()
        entries = entries.key_by('locus', 's')
        entries = entries.select('GT', 'GTA', 'GTZ')
        self.assertTrue(entries._same(expected))

    def test_import_vcf(self):
        vcf = hl.split_multi_hts(
            hl.import_vcf(resource('sample2.vcf'),
                          reference_genome=hl.get_reference('GRCh38'),
                          contig_recoding={"22": "chr22"}))

        vcf_table = vcf.rows()
        self.assertTrue(vcf_table.all(vcf_table.locus.contig == "chr22"))
        self.assertTrue(vcf.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_vcf_empty(self):
        mt = hl.import_vcf([resource('0var.vcf.bgz'), resource('3var.vcf.bgz')])
        assert mt._same(hl.import_vcf(resource('3var.vcf.bgz')))

    def test_import_vcf_no_reference_specified(self):
        vcf = hl.import_vcf(resource('sample2.vcf'),
                            reference_genome=None)
        self.assertEqual(vcf.locus.dtype, hl.tstruct(contig=hl.tstr, position=hl.tint32))
        self.assertEqual(vcf.count_rows(), 735)

    def test_import_vcf_bad_reference_allele(self):
        vcf = hl.import_vcf(resource('invalid_base.vcf'))
        self.assertEqual(vcf.count_rows(), 1)

    def test_import_vcf_flags_are_defined(self):
        # issue 3277
        t = hl.import_vcf(resource('sample.vcf')).rows()
        self.assertTrue(t.all(hl.is_defined(t.info.NEGATIVE_TRAIN_SITE) &
                              hl.is_defined(t.info.POSITIVE_TRAIN_SITE) &
                              hl.is_defined(t.info.DB) &
                              hl.is_defined(t.info.DS)))

    def test_import_vcf_can_import_float_array_format(self):
        mt = hl.import_vcf(resource('floating_point_array.vcf'))
        self.assertTrue(mt.aggregate_entries(hl.agg.all(mt.numeric_array == [1.5, 2.5])))
        self.assertEqual(hl.tarray(hl.tfloat64), mt['numeric_array'].dtype)

    def test_import_vcf_can_import_float32_array_format(self):
        mt = hl.import_vcf(resource('floating_point_array.vcf'), entry_float_type=hl.tfloat32)
        self.assertTrue(mt.aggregate_entries(hl.agg.all(mt.numeric_array == [1.5, 2.5])))
        self.assertEqual(hl.tarray(hl.tfloat32), mt['numeric_array'].dtype)

    def test_import_vcf_can_import_negative_numbers(self):
        mt = hl.import_vcf(resource('negative_format_fields.vcf'))
        self.assertTrue(mt.aggregate_entries(hl.agg.all(mt.negative_int == -1) &
                                             hl.agg.all(mt.negative_float == -1.5) &
                                             hl.agg.all(mt.negative_int_array == [-1, -2]) &
                                             hl.agg.all(mt.negative_float_array == [-0.5, -1.5])))

    def test_import_vcf_missing_info_field_elements(self):
        mt = hl.import_vcf(resource('missingInfoArray.vcf'), reference_genome='GRCh37', array_elements_required=False)
        mt = mt.select_rows(FOO=mt.info.FOO, BAR=mt.info.BAR)
        expected = hl.Table.parallelize([{'locus': hl.Locus('X', 16050036), 'alleles': ['A', 'C'],
                                          'FOO': [1, None], 'BAR': [2, None, None]},
                                         {'locus': hl.Locus('X', 16061250), 'alleles': ['T', 'A', 'C'],
                                          'FOO': [None, 2, None], 'BAR': [None, 1.0, None]}],
                                        hl.tstruct(locus=hl.tlocus('GRCh37'), alleles=hl.tarray(hl.tstr),
                                                   FOO=hl.tarray(hl.tint), BAR=hl.tarray(hl.tfloat64)),
                                        key=['locus', 'alleles'])
        self.assertTrue(mt.rows()._same(expected))

    def test_import_vcf_missing_format_field_elements(self):
        mt = hl.import_vcf(resource('missingFormatArray.vcf'), reference_genome='GRCh37', array_elements_required=False)
        mt = mt.select_rows().select_entries('AD', 'PL')

        expected = hl.Table.parallelize([{'locus': hl.Locus('X', 16050036), 'alleles': ['A', 'C'], 's': 'C1046::HG02024',
                                          'AD': [None, None], 'PL': [0, None, 180]},
                                         {'locus': hl.Locus('X', 16050036), 'alleles': ['A', 'C'], 's': 'C1046::HG02025',
                                          'AD': [None, 6], 'PL': [70, None]},
                                         {'locus': hl.Locus('X', 16061250), 'alleles': ['T', 'A', 'C'], 's': 'C1046::HG02024',
                                          'AD': [0, 0, None], 'PL': [396, None, None, 33, None, 0]},
                                         {'locus': hl.Locus('X', 16061250), 'alleles': ['T', 'A', 'C'], 's': 'C1046::HG02025',
                                          'AD': [0, 0, 9], 'PL': [None, None, None]}],
                                        hl.tstruct(locus=hl.tlocus('GRCh37'), alleles=hl.tarray(hl.tstr), s=hl.tstr,
                                                   AD=hl.tarray(hl.tint), PL=hl.tarray(hl.tint)),
                                        key=['locus', 'alleles', 's'])

        self.assertTrue(mt.entries()._same(expected))

    def test_vcf_unsorted_alleles(self):
        mt = hl.import_vcf(resource('sample.pksorted.vcf'), n_partitions=4)
        mt.rows()._force_count()

    def test_import_vcf_skip_invalid_loci(self):
        mt = hl.import_vcf(resource('skip_invalid_loci.vcf'), reference_genome='GRCh37',
                           skip_invalid_loci=True)
        self.assertEqual(mt._force_count_rows(), 3)

        with self.assertRaisesRegex(FatalError, 'Invalid locus'):
            hl.import_vcf(resource('skip_invalid_loci.vcf')).count()

    def test_import_vcf_set_field_missing(self):
        mt = hl.import_vcf(resource('test_set_field_missing.vcf'))
        mt.aggregate_entries(hl.agg.sum(mt.DP))

    def test_import_vcf_dosages_as_doubles_or_floats(self):
        mt = hl.import_vcf(resource('small-ds.vcf'))
        self.assertEqual(hl.expr.expressions.typed_expressions.Float64Expression, type(mt.entry.DS))
        mt32 = hl.import_vcf(resource('small-ds.vcf'),  entry_float_type=hl.tfloat32)
        self.assertEqual(hl.expr.expressions.typed_expressions.Float32Expression, type(mt32.entry.DS))
        mt_result = mt.annotate_entries(DS32=mt32.index_entries(mt.row_key, mt.col_key).DS)
        compare = mt_result.annotate_entries(
            test=(hl.coalesce(hl.approx_equal(mt_result.DS, mt_result.DS32, nan_same=True), True))
        )
        self.assertTrue(all(compare.test.collect()))

    def test_import_vcf_invalid_float_type(self):
        with self.assertRaises(TypeError):
            mt = hl.import_vcf(resource('small-ds.vcf'), entry_float_type=hl.tstr)
        with self.assertRaises(TypeError):
            mt = hl.import_vcf(resource('small-ds.vcf'), entry_float_type=hl.tint)
        with self.assertRaises(TypeError):
            mt = hl.import_vcf(resource('small-ds.vcf'), entry_float_type=hl.tint32)
        with self.assertRaises(TypeError):
            mt = hl.import_vcf(resource('small-ds.vcf'), entry_float_type=hl.tint64)

    def test_export_vcf(self):
        dataset = hl.import_vcf(resource('sample.vcf.bgz'))
        vcf_metadata = hl.get_vcf_metadata(resource('sample.vcf.bgz'))
        with TemporaryFilename(suffix='.vcf') as sample_vcf, \
             TemporaryFilename(suffix='.vcf') as no_sample_vcf:
            hl.export_vcf(dataset, sample_vcf, metadata=vcf_metadata)
            dataset_imported = hl.import_vcf(sample_vcf)
            self.assertTrue(dataset._same(dataset_imported))

            no_sample_dataset = dataset.filter_cols(False).select_entries()
            hl.export_vcf(no_sample_dataset, no_sample_vcf, metadata=vcf_metadata)
            no_sample_dataset_imported = hl.import_vcf(no_sample_vcf)
            self.assertTrue(no_sample_dataset._same(no_sample_dataset_imported))

            metadata_imported = hl.get_vcf_metadata(sample_vcf)
            # are py4 JavaMaps, not dicts, so can't use assertDictEqual
            self.assertEqual(vcf_metadata, metadata_imported)

    def test_export_vcf_quotes_and_backslash_in_description(self):
        ds = hl.import_vcf(resource("sample.vcf"))
        meta = hl.get_vcf_metadata(resource("sample.vcf"))
        meta["info"]["AF"]["Description"] = 'foo "bar" \\'
        with TemporaryFilename(suffix='.vcf') as test_vcf:
             hl.export_vcf(ds, test_vcf, metadata=meta)
             af_lines = [
                 line for line in hl.current_backend().fs.open(test_vcf).read().split('\n')
                 if line.startswith("##INFO=<ID=AF")
             ]
        assert len(af_lines) == 1, af_lines
        line = af_lines[0]
        assert line.startswith("##INFO=<") and line.endswith(">"), line
        line = line[8:-1]
        fields = dict([f.split("=") for f in line.split(",")])
        description = fields["Description"]
        assert description == '"foo \\"bar\\" \\\\"'

    def test_export_vcf_empty_format(self):
        mt = hl.import_vcf(resource('sample.vcf.bgz')).select_entries()
        tmp = new_temp_file(extension="vcf")
        hl.export_vcf(mt, tmp)

        assert hl.import_vcf(tmp)._same(mt)

    def test_export_vcf_no_gt(self):
        mt = hl.import_vcf(resource('sample.vcf.bgz')).drop('GT')
        tmp = new_temp_file(extension="vcf")
        hl.export_vcf(mt, tmp)

        assert hl.import_vcf(tmp)._same(mt)

    def test_export_vcf_no_alt_alleles(self):
        mt = hl.import_vcf(resource('gvcfs/HG0096_excerpt.g.vcf'), reference_genome='GRCh38')
        self.assertEqual(mt.filter_rows(hl.len(mt.alleles) == 1).count_rows(), 5)

        tmp = new_temp_file(extension="vcf")
        hl.export_vcf(mt, tmp)
        mt2 = hl.import_vcf(tmp, reference_genome='GRCh38')
        self.assertTrue(mt._same(mt2))

    def test_export_sites_only_from_table(self):
        mt = hl.import_vcf(resource('sample.vcf.bgz'))\
            .select_entries()\
            .filter_cols(False)

        tmp = new_temp_file(extension="vcf")
        hl.export_vcf(mt.rows(), tmp)
        assert hl.import_vcf(tmp)._same(mt)

    def import_gvcfs_sample_vcf(self, path):
        parts_type = hl.tarray(hl.tinterval(hl.tstruct(locus=hl.tlocus('GRCh37'))))
        parts = [
            hl.Interval(start=hl.Struct(locus=hl.Locus('20', 1)),
                        end=hl.Struct(locus=hl.Locus('20', 13509135)),
                        includes_end=True),
            hl.Interval(start=hl.Struct(locus=hl.Locus('20', 13509136)),
                        end=hl.Struct(locus=hl.Locus('20', 16493533)),
                        includes_end=True),
            hl.Interval(start=hl.Struct(locus=hl.Locus('20', 16493534)),
                        end=hl.Struct(locus=hl.Locus('20', 20000000)),
                        includes_end=True)
        ]
        parts_str = json.dumps(parts_type._convert_to_json(parts))
        vir = ir.MatrixVCFReader(path=path, call_fields=['PGT'], entry_float_type=hl.tfloat64,
                                 header_file=None, block_size=None, min_partitions=None,
                                 reference_genome='default', contig_recoding=None,
                                 array_elements_required=True, skip_invalid_loci=False,
                                 force_bgz=False, force_gz=False, filter=None, find_replace=None,
                                 n_partitions=None, _partitions_json=parts_str,
                                 _partitions_type=parts_type)

        vcf1 = hl.import_vcf(path)
        vcf2 = hl.MatrixTable(ir.MatrixRead(vir))
        self.assertEqual(len(parts), vcf2.n_partitions())
        self.assertTrue(vcf1._same(vcf2))

        interval = [hl.parse_locus_interval('[20:13509136-16493533]')]
        filter1 = hl.filter_intervals(vcf1, interval)
        filter2 = hl.filter_intervals(vcf2, interval)
        self.assertEqual(1, filter2.n_partitions())
        self.assertTrue(filter1._same(filter2))

        # we've selected exactly the middle partition ±1 position on either end
        interval_a = [hl.parse_locus_interval('[20:13509135-16493533]')]
        interval_b = [hl.parse_locus_interval('[20:13509136-16493534]')]
        interval_c = [hl.parse_locus_interval('[20:13509135-16493534]')]
        self.assertEqual(hl.filter_intervals(vcf2, interval_a).n_partitions(), 2)
        self.assertEqual(hl.filter_intervals(vcf2, interval_b).n_partitions(), 2)
        self.assertEqual(hl.filter_intervals(vcf2, interval_c).n_partitions(), 3)

    def test_tabix_export(self):
        mt = hl.import_vcf(resource('sample.vcf.bgz'))
        tmp = new_temp_file(extension="bgz")
        hl.export_vcf(mt, tmp, tabix=True)
        self.import_gvcfs_sample_vcf(tmp)

    def test_tabix_export_file_exists(self):
        mt = hl.import_vcf(resource('sample.vcf.bgz'))
        tmp = new_temp_file(extension="bgz")
        hl.export_vcf(mt, tmp, tabix=True, parallel='header_per_shard')
        files = hl.current_backend().fs.ls(tmp)
        self.assertTrue(any(f.path.endswith('.tbi') for f in files))

    def test_import_gvcfs(self):
        path = resource('sample.vcf.bgz')
        self.import_gvcfs_sample_vcf(path)

    @fails_service_backend()
    @fails_local_backend()
    def test_import_gvcfs_subset(self):
        path = resource('sample.vcf.bgz')
        parts = [
            hl.Interval(start=hl.Struct(locus=hl.Locus('20', 13509136)),
                        end=hl.Struct(locus=hl.Locus('20', 16493533)),
                        includes_end=True)
        ]
        vcf1 = hl.import_vcf(path).key_rows_by('locus')
        vcf2 = hl.import_gvcfs([path], parts)[0]
        interval = [hl.parse_locus_interval('[20:13509136-16493533]')]
        filter1 = hl.filter_intervals(vcf1, interval)
        self.assertTrue(vcf2._same(filter1))
        self.assertEqual(len(parts), vcf2.n_partitions())

    @fails_service_backend()
    @fails_local_backend()
    def test_import_gvcfs_long_line(self):
        import bz2
        path = resource('gvcfs/long_line.g.vcf.gz')
        parts = [
            hl.Interval(start=hl.Struct(locus=hl.Locus('1', 1)),
                        end=hl.Struct(locus=hl.Locus('1', 1_000_000)),
                        includes_end=True)
        ]
        [vcf] = hl.import_gvcfs([path], parts)
        [data] = vcf.info.Custom.collect()
        with bz2.open(resource('gvcfs/long_line.ref.bz2')) as ref:
            ref_str = ref.read().decode('utf-8')
            self.assertEqual(ref_str, data)

    def test_vcf_parser_golden_master__ex_GRCh37(self):
        self._test_vcf_parser_golden_master(resource('ex.vcf'), 'GRCh37')

    def test_vcf_parser_golden_master__sample_GRCh37(self):
        self._test_vcf_parser_golden_master(resource('sample.vcf'), 'GRCh37')

    def test_vcf_parser_golden_master__gvcf_GRCh37(self):
        self._test_vcf_parser_golden_master(resource('gvcfs/HG00096.g.vcf.gz'), 'GRCh38')

    def _test_vcf_parser_golden_master(self, vcf_path, rg):
        vcf = hl.import_vcf(
            vcf_path,
            reference_genome=rg,
            array_elements_required=False,
            force_bgz=True)
        mt = hl.read_matrix_table(vcf_path + '.mt')
        self.assertTrue(mt._same(vcf))

    @fails_service_backend()
    @fails_local_backend()
    def test_import_multiple_vcfs(self):
        _paths = ['gvcfs/HG00096.g.vcf.gz', 'gvcfs/HG00268.g.vcf.gz']
        paths = [resource(p) for p in _paths]
        parts = [
            hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 17821257, reference_genome='GRCh38')),
                        end=hl.Struct(locus=hl.Locus('chr20', 18708366, reference_genome='GRCh38')),
                        includes_end=True),
            hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 18708367, reference_genome='GRCh38')),
                        end=hl.Struct(locus=hl.Locus('chr20', 19776611, reference_genome='GRCh38')),
                        includes_end=True),
            hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 19776612, reference_genome='GRCh38')),
                        end=hl.Struct(locus=hl.Locus('chr20', 21144633, reference_genome='GRCh38')),
                        includes_end=True)
        ]
        int0 = hl.parse_locus_interval('[chr20:17821257-18708366]', reference_genome='GRCh38')
        int1 = hl.parse_locus_interval('[chr20:18708367-19776611]', reference_genome='GRCh38')
        hg00096, hg00268 = hl.import_gvcfs(paths, parts, reference_genome='GRCh38')
        filt096 = hl.filter_intervals(hg00096, [int0])
        filt268 = hl.filter_intervals(hg00268, [int1])
        self.assertEqual(1, filt096.n_partitions())
        self.assertEqual(1, filt268.n_partitions())
        pos096 = set(filt096.locus.position.collect())
        pos268 = set(filt268.locus.position.collect())
        self.assertFalse(pos096 & pos268)

    @fails_service_backend()
    @fails_local_backend()
    def test_combiner_works(self):
        from hail.experimental.vcf_combiner.vcf_combiner import transform_one, combine_gvcfs
        _paths = ['gvcfs/HG00096.g.vcf.gz', 'gvcfs/HG00268.g.vcf.gz']
        paths = [resource(p) for p in _paths]
        parts = [
            hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 17821257, reference_genome='GRCh38')),
                        end=hl.Struct(locus=hl.Locus('chr20', 18708366, reference_genome='GRCh38')),
                        includes_end=True),
            hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 18708367, reference_genome='GRCh38')),
                        end=hl.Struct(locus=hl.Locus('chr20', 19776611, reference_genome='GRCh38')),
                        includes_end=True),
            hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 19776612, reference_genome='GRCh38')),
                        end=hl.Struct(locus=hl.Locus('chr20', 21144633, reference_genome='GRCh38')),
                        includes_end=True)
        ]
        vcfs = [transform_one(mt.annotate_rows(info=mt.info.annotate(
            MQ_DP=hl.missing(hl.tint32),
            VarDP=hl.missing(hl.tint32),
            QUALapprox=hl.missing(hl.tint32))))
                for mt in hl.import_gvcfs(paths, parts, reference_genome='GRCh38',
                                         array_elements_required=False)]
        comb = combine_gvcfs(vcfs)
        self.assertEqual(len(parts), comb.n_partitions())
        comb._force_count_rows()

    def test_haploid_combiner_ok(self):
        from hail.experimental.vcf_combiner.vcf_combiner import transform_gvcf
        # make a combiner table
        mt = hl.utils.range_matrix_table(2, 1)
        mt = mt.annotate_cols(s='S01')
        mt = mt.key_cols_by('s')
        mt = mt.select_cols()
        mt = mt.annotate_rows(locus=hl.locus(contig='chrX', pos=mt.row_idx + 100, reference_genome='GRCh38'))
        mt = mt.key_rows_by('locus')
        mt = mt.annotate_rows(alleles=['A', '<NON_REF>'])
        mt = mt.annotate_entries(GT=hl.if_else((mt.row_idx % 2) == 0, hl.call(0), hl.call(0, 0)))
        mt = mt.annotate_entries(DP=31)
        mt = mt.annotate_entries(GQ=30)
        mt = mt.annotate_entries(PL=hl.if_else((mt.row_idx % 2) == 0, [0, 20], [0, 20, 400]))
        mt = mt.annotate_rows(info=hl.struct(END=mt.locus.position))
        mt = mt.annotate_rows(rsid=hl.missing(hl.tstr))
        mt = mt.drop('row_idx')
        transform_gvcf(mt)._force_count()

    def test_combiner_parse_as_annotations(self):
        from hail.experimental.vcf_combiner.vcf_combiner import parse_as_fields
        infos = hl.array([
            hl.struct(
                AS_QUALapprox="|1171|",
                AS_SB_TABLE="0,0|30,27|0,0",
                AS_VarDP="0|57|0",
                AS_RAW_MQ="0.00|15100.00|0.00",
                AS_RAW_MQRankSum="|0.0,1|NaN",
                AS_RAW_ReadPosRankSum="|0.7,1|NaN"),
            hl.struct(
                AS_QUALapprox="|1171|",
                AS_SB_TABLE="0,0|30,27|0,0",
                AS_VarDP="0|57|0",
                AS_RAW_MQ="0.00|15100.00|0.00",
                AS_RAW_MQRankSum="|NaN|NaN",
                AS_RAW_ReadPosRankSum="|NaN|NaN")])

        output = hl.eval(infos.map(lambda info: parse_as_fields(info, False)))
        expected = [
            hl.Struct(
                AS_QUALapprox=[None, 1171, None],
                AS_SB_TABLE=[[0, 0], [30, 27], [0, 0]],
                AS_VarDP=[0, 57, 0],
                AS_RAW_MQ=[0.00, 15100.00, 0.00],
                AS_RAW_MQRankSum=[None, (0.0, 1), None],
                AS_RAW_ReadPosRankSum=[None, (0.7, 1), None]),
            hl.Struct(
                AS_QUALapprox=[None, 1171, None],
                AS_SB_TABLE=[[0, 0], [30, 27], [0, 0]],
                AS_VarDP=[0, 57, 0],
                AS_RAW_MQ=[0.00, 15100.00, 0.00],
                AS_RAW_MQRankSum=[None, None, None],
                AS_RAW_ReadPosRankSum=[None, None, None])]
        assert output == expected

    def test_flag_at_eol(self):
        vcf_path = resource('flag_at_end.vcf')
        mt = hl.import_vcf(vcf_path)
        assert mt._force_count_rows() == 1

    def test_missing_float_entries(self):
        vcf = hl.import_vcf(resource('noglgp.vcf'), array_elements_required=False,
                            reference_genome='GRCh38')
        gl_gp = vcf.aggregate_entries(hl.agg.collect(hl.struct(GL=vcf.GL, GP=vcf.GP)))
        assert gl_gp == [hl.Struct(GL=[None, None, None], GP=[0.22, 0.5, 0.27]),
                         hl.Struct(GL=[None, None, None], GP=[None, None, None])]

    def test_same_bgzip(self):
        mt = hl.import_vcf(resource('sample.vcf'), min_partitions=4)
        f = new_temp_file(extension='vcf.bgz')
        hl.export_vcf(mt, f)
        assert hl.import_vcf(f)._same(mt)

    def test_vcf_parallel_separate_header_export(self):
        fs = hl.current_backend().fs
        def concat_files(outpath, inpaths):
            with fs.open(outpath, 'wb') as outfile:
                for path in inpaths:
                    with fs.open(path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)

        mt = hl.import_vcf(resource('sample.vcf'), min_partitions=4)
        f = new_temp_file(extension='vcf.bgz')
        hl.export_vcf(mt, f, parallel='separate_header')
        stat = fs.stat(f)
        assert stat
        assert stat.is_dir()
        shard_paths = [info.path for info in fs.ls(f)
                       if os.path.splitext(info.path)[-1] == '.bgz']
        assert shard_paths
        shard_paths.sort()
        nf = new_temp_file(extension='vcf.bgz')
        concat_files(nf, shard_paths)

        assert hl.import_vcf(nf)._same(mt)

    def test_custom_rg_import(self):
        rg = hl.ReferenceGenome.read(resource('deid_ref_genome.json'))
        mt = hl.import_vcf(resource('custom_rg.vcf'), reference_genome=rg)
        assert mt.locus.collect() == [hl.Locus('D', 123, reference_genome=rg)]

    def test_sorted(self):
        mt = hl.utils.range_matrix_table(10, 10, n_partitions=4).filter_cols(False)
        mt = mt.key_cols_by(s='dummy')
        mt = mt.annotate_entries(GT=hl.unphased_diploid_gt_index_call(0))
        mt = mt.key_rows_by(locus=hl.locus('1', 100 - mt.row_idx), alleles=['A', 'T'])
        f = new_temp_file(extension='vcf')
        hl.export_vcf(mt, f)

        last = 0
        with hl.current_backend().fs.open(f, 'r') as i:
            for line in i:
                if line.startswith('#'):
                    continue
                else:
                    pos = int(line.split('\t')[1])
                    assert pos >= last
                    last = pos

    def test_empty_read_write(self):
        mt = hl.import_vcf(resource('sample.vcf'), min_partitions=4).filter_rows(False)

        out1 = new_temp_file(extension='vcf')
        out2 = new_temp_file(extension='vcf.bgz')

        hl.export_vcf(mt, out1)
        hl.export_vcf(mt, out2)

        assert hl.current_backend().fs.stat(out1).size > 0
        assert hl.current_backend().fs.stat(out2).size > 0

        assert hl.import_vcf(out1)._same(mt)
        assert hl.import_vcf(out2)._same(mt)

    def test_empty_import_vcf_group_by_collect(self):
        mt = hl.import_vcf(resource('sample.vcf'), min_partitions=4).filter_rows(False)
        ht = mt._localize_entries('entries', 'columns')
        groups = ht.group_by(the_key=ht.key).aggregate(values=hl.agg.collect(ht.row_value)).collect()
        assert not groups

    def test_format_header(self):
        mt = hl.import_vcf(resource('sample2.vcf'))
        metadata = hl.get_vcf_metadata(resource('sample2.vcf'))
        f = new_temp_file(extension='vcf')
        hl.export_vcf(mt, f, metadata=metadata)

        s = set()
        with hl.current_backend().fs.open(f, 'r') as i:
            for line in i:
                if line.startswith('##FORMAT'):
                    s.add(line.strip())

        assert s == {
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">',
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">',
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">',
            '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">',
        }

    def test_format_genotypes(self):
        mt = hl.import_vcf(resource('sample.vcf'))
        f = new_temp_file(extension='vcf')
        hl.export_vcf(mt, f)
        with hl.current_backend().fs.open(f, 'r') as i:
            for line in i:
                if line.startswith('20\t13029920'):
                    expected = "GT:AD:DP:GQ:PL\t1/1:0,6:6:18:234,18,0\t1/1:0,4:4:12:159,12,0\t" \
                               "1/1:0,4:4:12:163,12,0\t1/1:0,12:12:36:479,36,0\t" \
                               "1/1:0,4:4:12:149,12,0\t1/1:0,6:6:18:232,18,0\t" \
                               "1/1:0,6:6:18:242,18,0\t1/1:0,3:3:9:119,9,0\t1/1:0,9:9:27:374,27,0" \
                               "\t./.:1,0:1\t1/1:0,3:3:9:133,9,0"
                    assert expected in line
                    break
            else:
                assert False, 'expected pattern not found'

    def test_contigs_header(self):
        mt = hl.import_vcf(resource('sample.vcf')).filter_cols(False)
        f = new_temp_file(extension='vcf')
        hl.export_vcf(mt, f)
        with hl.current_backend().fs.open(f, 'r') as i:
            for line in i:
                if line.startswith('##contig=<ID=10'):
                    assert line.strip() == '##contig=<ID=10,length=135534747,assembly=GRCh37>'
                    break
            else:
                assert False, 'expected pattern not found'

    def test_metadata_argument(self):
        mt = hl.import_vcf(resource('multipleChromosomes.vcf'))
        f = new_temp_file(extension='vcf')
        metadata = {
            'filter': {'LowQual': {'Description': 'Low quality'}},
            'format': {'GT': {'Description': 'Genotype call.', 'Number': 'foo'}},
            'fakeField': {}
        }
        hl.export_vcf(mt, f, metadata=metadata)

        saw_gt = False
        saw_lq = False
        with hl.current_backend().fs.open(f, 'r') as f:
            for line in f:
                print(line[:25])
                if line.startswith('##FORMAT=<ID=GT'):
                    assert line.strip() == '##FORMAT=<ID=GT,Number=foo,Type=String,Description="Genotype call.">'
                    assert not saw_gt
                    saw_gt = True
                elif line.startswith('##FILTER=<ID=LowQual'):
                    assert line.strip() == '##FILTER=<ID=LowQual,Description="Low quality">'
                    assert not saw_lq
                    saw_lq = True
        assert saw_gt
        assert saw_lq

    def test_invalid_info_fields(self):
        t = new_temp_file(extension='vcf')
        mt = hl.import_vcf(resource('sample.vcf'))


        with mock.patch("hail.methods.impex.warning", autospec=True) as warning:
            hl.export_vcf(mt, t)
            assert warning.call_count == 0

        for invalid_field in ["foo-1", "123", "bar baz"]:
            with mock.patch("hail.methods.impex.warning", autospec=True) as warning:
                hl.export_vcf(mt.annotate_rows(info=mt.info.annotate(**{invalid_field: True})), t)
                assert warning.call_count == 1

    def test_vcf_different_info_errors(self):
        with self.assertRaisesRegex(FatalError, "Check that all files have same INFO fields"):
            mt = hl.import_vcf([resource('different_info_test1.vcf'), resource('different_info_test2.vcf')])
            mt.rows()._force_count()


class PLINKTests(unittest.TestCase):
    def test_import_fam(self):
        fam_file = resource('sample.fam')
        nfam = hl.import_fam(fam_file).count()
        i = 0
        with hl.current_backend().fs.open(fam_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nfam, i)

    def test_export_import_plink_same(self):
        mt = get_dataset()
        mt = mt.select_rows(rsid=hl.delimit([mt.locus.contig, hl.str(mt.locus.position), mt.alleles[0], mt.alleles[1]], ':'),
                            cm_position=15.0)
        mt = mt.select_cols(fam_id=hl.missing(hl.tstr), pat_id=hl.missing(hl.tstr), mat_id=hl.missing(hl.tstr),
                            is_female=hl.missing(hl.tbool), is_case=hl.missing(hl.tbool))
        mt = mt.select_entries('GT')

        bfile = new_temp_file(prefix='test_import_export_plink')
        hl.export_plink(mt, bfile, ind_id=mt.s, cm_position=mt.cm_position)

        mt_imported = hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam',
                                      a2_reference=True, reference_genome='GRCh37', n_partitions=8)
        self.assertTrue(mt._same(mt_imported))
        self.assertTrue(mt.aggregate_rows(hl.agg.all(mt.cm_position == 15.0)))

    def test_import_plink_empty_fam(self):
        mt = get_dataset().filter_cols(False)
        bfile = new_temp_file(prefix='test_empty_fam')
        hl.export_plink(mt, bfile, ind_id=mt.s)
        with self.assertRaisesRegex(FatalError, "Empty FAM file"):
            hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam')

    def test_import_plink_empty_bim(self):
        mt = get_dataset().filter_rows(False)
        bfile = new_temp_file(prefix='test_empty_bim')
        hl.export_plink(mt, bfile, ind_id=mt.s)
        with self.assertRaisesRegex(FatalError, "BIM file does not contain any variants"):
            hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam')

    def test_import_plink_a1_major(self):
        mt = get_dataset()
        bfile = new_temp_file(prefix='sample_plink')
        hl.export_plink(mt, bfile, ind_id=mt.s)

        def get_data(a2_reference):
            mt_imported = hl.import_plink(bfile + '.bed', bfile + '.bim',
                                          bfile + '.fam', a2_reference=a2_reference)
            return (hl.variant_qc(mt_imported)
                    .rows()
                    .key_by('rsid'))

        a2 = get_data(a2_reference=True)
        a1 = get_data(a2_reference=False)

        j = (a2.annotate(a1_alleles=a1[a2.rsid].alleles, a1_vqc=a1[a2.rsid].variant_qc)
             .rename({'variant_qc': 'a2_vqc', 'alleles': 'a2_alleles'}))

        self.assertTrue(j.all((j.a1_alleles[0] == j.a2_alleles[1]) &
                              (j.a1_alleles[1] == j.a2_alleles[0]) &
                              (j.a1_vqc.n_not_called == j.a2_vqc.n_not_called) &
                              (j.a1_vqc.n_het == j.a2_vqc.n_het) &
                              (j.a1_vqc.homozygote_count[0] == j.a2_vqc.homozygote_count[1]) &
                              (j.a1_vqc.homozygote_count[1] == j.a2_vqc.homozygote_count[0])))

    def test_import_plink_same_locus(self):
        mt = hl.balding_nichols_model(n_populations=2, n_samples=10, n_variants=100)
        mt = mt.key_rows_by(locus=hl.locus('1', 100, reference_genome='GRCh37'), alleles=mt.alleles).select_rows()
        mt = mt.key_cols_by(s=hl.str(mt.sample_idx)).select_cols()
        mt = mt.select_globals()
        out = new_temp_file(prefix='plink')
        hl.export_plink(mt, out)
        mt2 = hl.import_plink(f'{out}.bed', f'{out}.bim', f'{out}.fam').select_cols().select_rows()
        assert mt2._same(mt)

        mt3 = hl.import_plink(f'{out}.bed', f'{out}.bim', f'{out}.fam', min_partitions=10).select_cols().select_rows()
        assert mt3._same(mt)

    def test_import_plink_partitions(self):
        mt = hl.balding_nichols_model(n_populations=2, n_samples=10, n_variants=100)
        mt = mt.select_rows()
        mt = mt.key_cols_by(s=hl.str(mt.sample_idx)).select_cols()
        mt = mt.select_globals()
        out = new_temp_file(prefix='plink')
        hl.export_plink(mt, out)
        mt2 = hl.import_plink(f'{out}.bed', f'{out}.bim', f'{out}.fam', min_partitions=10).select_cols().select_rows()
        assert mt2.n_partitions() == 10
        assert mt2._same(mt)

    def test_import_plink_contig_recoding_w_reference(self):
        vcf = hl.split_multi_hts(
            hl.import_vcf(resource('sample2.vcf'),
                          reference_genome=hl.get_reference('GRCh38'),
                          contig_recoding={"22": "chr22"}))

        bfile = new_temp_file(prefix='sample_plink')
        hl.export_plink(vcf, bfile)

        plink = hl.import_plink(
            bfile + '.bed', bfile + '.bim', bfile + '.fam',
            a2_reference=True,
            contig_recoding={'chr22': '22'},
            reference_genome='GRCh37').rows()
        self.assertTrue(plink.all(plink.locus.contig == "22"))
        self.assertEqual(vcf.count_rows(), plink.count())
        self.assertTrue(plink.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_plink_no_reference_specified(self):
        bfile = resource('fastlmmTest')
        plink = hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam',
                                reference_genome=None)
        self.assertEqual(plink.locus.dtype,
                         hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_plink_and_ignore_rows(self):
        bfile = doctest_resource('ldsc')
        plink = hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam', block_size=16)
        self.assertEqual(plink.aggregate_cols(hl.agg.count()), 489)

    def test_import_plink_skip_invalid_loci(self):
        mt = hl.import_plink(resource('skip_invalid_loci.bed'),
                             resource('skip_invalid_loci.bim'),
                             resource('skip_invalid_loci.fam'),
                             reference_genome='GRCh37',
                             skip_invalid_loci=True,
                             contig_recoding={'chr1': '1'})
        self.assertEqual(mt._force_count_rows(), 3)

        with self.assertRaisesRegex(FatalError, 'Invalid locus'):
            (hl.import_plink(resource('skip_invalid_loci.bed'),
                            resource('skip_invalid_loci.bim'),
                            resource('skip_invalid_loci.fam'))
             ._force_count_rows())

    @unittest.skipIf('HAIL_TEST_SKIP_PLINK' in os.environ, 'Skipping tests requiring plink')
    @fails_service_backend()
    def test_export_plink(self):
        vcf_file = resource('sample.vcf')
        mt = hl.split_multi_hts(hl.import_vcf(vcf_file, min_partitions=10))

        # permute columns so not in alphabetical order!
        import random
        indices = list(range(mt.count_cols()))
        random.shuffle(indices)
        mt = mt.choose_cols(indices)

        split_vcf_file = uri_path(new_temp_file())
        hl_output = uri_path(new_temp_file())
        plink_output = uri_path(new_temp_file())
        merge_output = uri_path(new_temp_file())

        hl.export_vcf(mt, split_vcf_file)
        hl.export_plink(mt, hl_output)

        run_command(["plink", "--vcf", split_vcf_file,
                     "--make-bed", "--out", plink_output,
                     "--const-fid", "--keep-allele-order"])

        data = []
        with open(uri_path(plink_output + ".bim")) as file:
            for line in file:
                row = line.strip().split()
                row[1] = ":".join([row[0], row[3], row[5], row[4]])
                data.append("\t".join(row) + "\n")

        with open(plink_output + ".bim", 'w') as f:
            f.writelines(data)

        run_command(["plink", "--bfile", plink_output,
                     "--bmerge", hl_output, "--merge-mode",
                     "6", "--out", merge_output])

        same = True
        with open(merge_output + ".diff") as f:
            for line in f:
                row = line.strip().split()
                if row != ["SNP", "FID", "IID", "NEW", "OLD"]:
                    same = False
                    break

        self.assertTrue(same)

    def test_export_plink_exprs(self):
        ds = get_dataset()
        fam_mapping = {'f0': 'fam_id', 'f1': 'ind_id', 'f2': 'pat_id', 'f3': 'mat_id',
                       'f4': 'is_female', 'f5': 'pheno'}
        bim_mapping = {'f0': 'contig', 'f1': 'varid', 'f2': 'cm_position',
                       'f3': 'position', 'f4': 'a1', 'f5': 'a2'}

        # Test default arguments
        out1 = new_temp_file()
        hl.export_plink(ds, out1)
        fam1 = (hl.import_table(out1 + '.fam', no_header=True, impute=False, missing="")
                .rename(fam_mapping))
        bim1 = (hl.import_table(out1 + '.bim', no_header=True, impute=False)
                .rename(bim_mapping))

        self.assertTrue(fam1.all((fam1.fam_id == "0") & (fam1.pat_id == "0") &
                                 (fam1.mat_id == "0") & (fam1.is_female == "0") &
                                 (fam1.pheno == "NA")))
        self.assertTrue(bim1.all((bim1.varid == bim1.contig + ":" + bim1.position + ":" + bim1.a2 + ":" + bim1.a1) &
                                 (bim1.cm_position == "0.0")))

        # Test non-default FAM arguments
        out2 = new_temp_file()
        hl.export_plink(ds, out2, ind_id=ds.s, fam_id=ds.s, pat_id="nope",
                        mat_id="nada", is_female=True, pheno=False)
        fam2 = (hl.import_table(out2 + '.fam', no_header=True, impute=False, missing="")
                .rename(fam_mapping))

        self.assertTrue(fam2.all((fam2.fam_id == fam2.ind_id) & (fam2.pat_id == "nope") &
                                 (fam2.mat_id == "nada") & (fam2.is_female == "2") &
                                 (fam2.pheno == "1")))

        # Test quantitative phenotype
        out3 = new_temp_file()
        hl.export_plink(ds, out3, ind_id=ds.s, pheno=hl.float64(hl.len(ds.s)))
        fam3 = (hl.import_table(out3 + '.fam', no_header=True, impute=False, missing="")
                .rename(fam_mapping))

        self.assertTrue(fam3.all((fam3.fam_id == "0") & (fam3.pat_id == "0") &
                                 (fam3.mat_id == "0") & (fam3.is_female == "0") &
                                 (fam3.pheno != "0") & (fam3.pheno != "NA")))

        # Test non-default BIM arguments
        out4 = new_temp_file()
        hl.export_plink(ds, out4, varid="hello", cm_position=100)
        bim4 = (hl.import_table(out4 + '.bim', no_header=True, impute=False)
                .rename(bim_mapping))

        self.assertTrue(bim4.all((bim4.varid == "hello") & (bim4.cm_position == "100.0")))

        # Test call expr
        out5 = new_temp_file()
        ds_call = ds.annotate_entries(gt_fake=hl.call(0, 0))
        hl.export_plink(ds_call, out5, call=ds_call.gt_fake)
        ds_all_hom_ref = hl.import_plink(out5 + '.bed', out5 + '.bim', out5 + '.fam')
        nerrors = ds_all_hom_ref.aggregate_entries(hl.agg.count_where(~ds_all_hom_ref.GT.is_hom_ref()))
        self.assertTrue(nerrors == 0)

        # Test white-space in FAM id expr raises error
        with self.assertRaisesRegex(TypeError, "has spaces in the following values:"):
            hl.export_plink(ds, new_temp_file(), mat_id="hello world")

        # Test white-space in varid expr raises error
        with self.assertRaisesRegex(FatalError, "no white space allowed:"):
            hl.export_plink(ds, new_temp_file(), varid="hello world")

    def test_contig_recoding_defaults(self):
        hl.import_plink(resource('sex_mt_contigs.bed'),
                        resource('sex_mt_contigs.bim'),
                        resource('sex_mt_contigs.fam'),
                        reference_genome='GRCh37')

        hl.import_plink(resource('sex_mt_contigs.bed'),
                        resource('sex_mt_contigs.bim'),
                        resource('sex_mt_contigs.fam'),
                        reference_genome='GRCh38')

        rg_random = hl.ReferenceGenome("random", ['1', '23', '24', '25', '26'],
                                       {'1': 10, '23': 10, '24': 10, '25': 10, '26': 10})

        hl.import_plink(resource('sex_mt_contigs.bed'),
                        resource('sex_mt_contigs.bim'),
                        resource('sex_mt_contigs.fam'),
                        reference_genome='random')

    def test_export_plink_struct_locus(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.key_rows_by(locus=hl.struct(contig=hl.str(mt.row_idx), position=mt.row_idx), alleles=['A', 'T']).select_rows()
        mt = mt.key_cols_by(s=hl.str(mt.col_idx)).select_cols()
        mt = mt.annotate_entries(GT=hl.call(0, 0))

        out = new_temp_file()

        hl.export_plink(mt, out)
        mt2 = hl.import_plink(
            bed=out + '.bed',
            bim=out + '.bim',
            fam=out + '.fam',
            reference_genome=None).select_rows().select_cols()
        assert mt._same(mt2)


# this routine was used to generate resources random.gen, random.sample
# random.bgen was generated with qctool v2.0rc9:
# qctool -g random.gen -s random.sample -bgen-bits 8 -og random.bgen
#
# random-a.bgen, random-b.bgen, random-c.bgen was generated as follows:
# head -n 10 random.gen > random-a.gen; head -n 20 random.gen | tail -n 10 > random-b.gen; tail -n 10 random.gen > random-c.gen
# qctool -g random-a.gen -s random.sample -og random-a.bgen -bgen-bits 8
# qctool -g random-b.gen -s random.sample -og random-b.bgen -bgen-bits 8
# qctool -g random-c.gen -s random.sample -og random-c.bgen -bgen-bits 8
#
# random-*-disjoint.bgen was generated as follows:
# while read line; do echo $RANDOM $line; done < src/test/resources/random.gen | sort -n | cut -f2- -d' ' > random-shuffled.gen
# head -n 10 random-shuffled.gen > random-a-disjoint.gen; head -n 20 random-shuffled.gen | tail -n 10 > random-b-disjoint.gen; tail -n 10 random-shuffled.gen > random-c-disjoint.gen
# qctool -g random-a-disjoint.gen -s random.sample -og random-a-disjoint.bgen -bgen-bits 8
# qctool -g random-b-disjoint.gen -s random.sample -og random-b-disjoint.bgen -bgen-bits 8
# qctool -g random-c-disjoint.gen -s random.sample -og random-c-disjoint.bgen -bgen-bits 8
def generate_random_gen():
    mt = hl.utils.range_matrix_table(30, 10)
    mt = (mt.annotate_rows(locus = hl.locus('20', mt.row_idx + 1),
                           alleles = ['A', 'G'])
          .key_rows_by('locus', 'alleles'))
    mt = (mt.annotate_cols(s = hl.str(mt.col_idx))
          .key_cols_by('s'))
    # using totally random values leads rounding differences where
    # identical GEN values get rounded differently, leading to
    # differences in the GT call between import_{gen, bgen}
    mt = mt.annotate_entries(a = hl.int32(hl.rand_unif(0.0, 255.0)))
    mt = mt.annotate_entries(b = hl.int32(hl.rand_unif(0.0, 255.0 - mt.a)))
    mt = mt.transmute_entries(GP = hl.array([mt.a, mt.b, 255.0 - mt.a - mt.b]) / 255.0)
    # 20% missing
    mt = mt.filter_entries(hl.rand_bool(0.8))
    hl.export_gen(mt, 'random', precision=4)


class BGENTests(unittest.TestCase):
    def test_error_if_no_gp(self):
        mt = hl.balding_nichols_model(3, 3, 3)
        mt = mt.key_cols_by(s=hl.str(mt.sample_idx))
        tmp_path = new_temp_file(extension='bgen')
        with pytest.raises(ValueError, match="BGEN requires a GP"):
            hl.export_bgen(mt, tmp_path)

        with pytest.raises(ValueError, match="GEN requires a GP"):
            hl.export_gen(mt, tmp_path)

    def test_import_bgen_dosage_entry(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['dosage'])
        self.assertEqual(bgen.entry.dtype, hl.tstruct(dosage=hl.tfloat64))
        self.assertEqual(bgen.count_rows(), 199)
        self.assertEqual(bgen._force_count_rows(), 199)

    def test_import_bgen_GT_GP_entries(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GT', 'GP'],
                              sample_file=resource('example.sample'))
        self.assertEqual(bgen.entry.dtype, hl.tstruct(GT=hl.tcall, GP=hl.tarray(hl.tfloat64)))

    def test_import_bgen_no_entries(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=[],
                              sample_file=resource('example.sample'))
        self.assertEqual(bgen.entry.dtype, hl.tstruct())

    def test_import_bgen_no_reference(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GT', 'GP', 'dosage'],
                              index_file_map={resource('example.8bits.bgen'): resource('example.8bits.bgen-NO-REFERENCE-GENOME.idx2')})
        assert bgen.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32)
        assert bgen.count_rows() == 199

    def test_import_bgen_skip_invalid_loci_does_not_error_with_invalid_loci(self):
        # Note: the skip_invalid_loci.bgen has 16-bit probabilities, and Hail
        # will crash if the genotypes are decoded
        mt = hl.import_bgen(resource('skip_invalid_loci.bgen'),
                            entry_fields=[],
                            sample_file=resource('skip_invalid_loci.sample'))
        assert mt.rows().count() == 3

    def test_import_bgen_errors_with_invalid_loci(self):
        with hl.TemporaryFilename(suffix='.bgen') as f:
            hl.current_backend().fs.copy(resource('skip_invalid_loci.bgen'), f)
            with pytest.raises(FatalError, match='Invalid locus'):
                hl.index_bgen(f)
                mt = hl.import_bgen(f,
                                    entry_fields=[],
                                    sample_file=resource('skip_invalid_loci.sample'))
                mt.rows().count()

    def test_import_bgen_gavin_example(self):
        recoding = {'0{}'.format(i): str(i) for i in range(1, 10)}

        sample_file = resource('example.sample')
        genmt = hl.import_gen(resource('example.gen'), sample_file,
                              contig_recoding=recoding,
                              reference_genome="GRCh37")

        bgen_file = resource('example.8bits.bgen')
        bgenmt = hl.import_bgen(bgen_file, ['GT', 'GP'], sample_file)
        self.assertTrue(
            bgenmt._same(genmt, tolerance=1.0 / 255, absolute=True))

    def test_import_bgen_random(self):
        sample_file = resource('random.sample')
        genmt = hl.import_gen(resource('random.gen'), sample_file)

        bgenmt = hl.import_bgen(resource('random.bgen'), ['GT', 'GP'], sample_file)
        self.assertTrue(
            bgenmt._same(genmt, tolerance=1.0 / 255, absolute=True))

    def test_parallel_import(self):
        mt = hl.import_bgen(resource('parallelBgenExport.bgen'),
                            ['GT', 'GP'],
                            resource('parallelBgenExport.sample'))
        self.assertEqual(mt.count(), (16, 10))

    def test_import_bgen_dosage_and_gp_dosage_function_agree(self):
        recoding = {'0{}'.format(i): str(i) for i in range(1, 10)}

        sample_file = resource('example.sample')
        bgen_file = resource('example.8bits.bgen')

        bgenmt = hl.import_bgen(bgen_file, ['GP', 'dosage'], sample_file)
        et = bgenmt.entries()
        et = et.transmute(gp_dosage = hl.gp_dosage(et.GP))
        self.assertTrue(et.all(
            (hl.is_missing(et.dosage) & hl.is_missing(et.gp_dosage)) |
            (hl.abs(et.dosage - et.gp_dosage) < 1e-6)))

    def test_import_bgen_row_fields(self):
        default_row_fields = hl.import_bgen(resource('example.8bits.bgen'),
                                            entry_fields=['dosage'])
        self.assertEqual(default_row_fields.row.dtype,
                         hl.tstruct(locus=hl.tlocus('GRCh37'),
                                    alleles=hl.tarray(hl.tstr),
                                    rsid=hl.tstr,
                                    varid=hl.tstr))
        no_row_fields = hl.import_bgen(resource('example.8bits.bgen'),
                                       entry_fields=['dosage'],
                                       _row_fields=[])
        self.assertEqual(no_row_fields.row.dtype,
                         hl.tstruct(locus=hl.tlocus('GRCh37'),
                                    alleles=hl.tarray(hl.tstr)))
        varid_only = hl.import_bgen(resource('example.8bits.bgen'),
                                    entry_fields=['dosage'],
                                    _row_fields=['varid'])
        self.assertEqual(varid_only.row.dtype,
                         hl.tstruct(locus=hl.tlocus('GRCh37'),
                                    alleles=hl.tarray(hl.tstr),
                                    varid=hl.tstr))
        rsid_only = hl.import_bgen(resource('example.8bits.bgen'),
                                   entry_fields=['dosage'],
                                   _row_fields=['rsid'])
        self.assertEqual(rsid_only.row.dtype,
                         hl.tstruct(locus=hl.tlocus('GRCh37'),
                                    alleles=hl.tarray(hl.tstr),
                                    rsid=hl.tstr))

        self.assertTrue(default_row_fields.drop('varid')._same(rsid_only))
        self.assertTrue(default_row_fields.drop('rsid')._same(varid_only))
        self.assertTrue(
            default_row_fields.drop('varid', 'rsid')._same(no_row_fields))

    def test_import_bgen_variant_filtering_from_literals(self):
        bgen_file = resource('example.8bits.bgen')

        alleles = ['A', 'G']

        desired_variants = [
            hl.Struct(locus=hl.Locus('1', 2000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 2001), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 4000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 10000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 100001), alleles=alleles),
        ]

        expected_result = [
            hl.Struct(locus=hl.Locus('1', 2000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 2001), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 4000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 10000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 10000), alleles=alleles), # Duplicated variant
            hl.Struct(locus=hl.Locus('1', 100001), alleles=alleles),
        ]

        part_1 = hl.import_bgen(bgen_file,
                                ['GT'],
                                n_partitions=1, # forcing seek to be called
                                variants=desired_variants)
        self.assertEqual(part_1.rows().key_by('locus', 'alleles').select().collect(), expected_result)

        part_199 = hl.import_bgen(bgen_file,
                                ['GT'],
                                n_partitions=199, # forcing each variant to be its own partition for testing duplicates work properly
                                variants=desired_variants)
        self.assertEqual(part_199.rows().key_by('locus', 'alleles').select().collect(), expected_result)

        everything = hl.import_bgen(bgen_file, ['GT'])
        self.assertEqual(everything.count(), (199, 500))

        expected = everything.filter_rows(hl.set(desired_variants).contains(everything.row_key))

        self.assertTrue(expected._same(part_1))

    def test_import_bgen_locus_filtering_from_literals(self):
        bgen_file = resource('example.8bits.bgen')

        # Test with Struct(Locus)
        desired_loci = [hl.Struct(locus=hl.Locus('1', 10000))]

        expected_result = [
            hl.Struct(locus=hl.Locus('1', 10000), alleles=['A', 'G']),
            hl.Struct(locus=hl.Locus('1', 10000), alleles=['A', 'G']) # Duplicated variant
        ]

        locus_struct = hl.import_bgen(bgen_file,
                                      ['GT'],
                                      variants=desired_loci)
        self.assertEqual(locus_struct.rows().key_by('locus', 'alleles').select().collect(),
                         expected_result)

        # Test with Locus object
        desired_loci = [hl.Locus('1', 10000)]

        locus_object = hl.import_bgen(bgen_file,
                                      ['GT'],
                                      variants=desired_loci)
        self.assertEqual(locus_object.rows().key_by('locus', 'alleles').select().collect(),
                         expected_result)

    def test_import_bgen_variant_filtering_from_exprs(self):
        bgen_file = resource('example.8bits.bgen')

        everything = hl.import_bgen(bgen_file, ['GT'])
        # self.assertEqual(everything.count(), (199, 500))

        desired_variants = hl.struct(locus=everything.locus, alleles=everything.alleles)

        actual = hl.import_bgen(bgen_file,
                                ['GT'],
                                n_partitions=10,
                                variants=desired_variants) # filtering with everything

        self.assertTrue(everything._same(actual))

    def test_import_bgen_locus_filtering_from_exprs(self):
        bgen_file = resource('example.8bits.bgen')

        everything = hl.import_bgen(bgen_file, ['GT'])
        self.assertEqual(everything.count(), (199, 500))

        actual_struct = hl.import_bgen(bgen_file,
                                ['GT'],
                                variants=hl.struct(locus=everything.locus))

        self.assertTrue(everything._same(actual_struct))

        actual_locus = hl.import_bgen(bgen_file,
                                ['GT'],
                                variants=everything.locus)

        self.assertTrue(everything._same(actual_locus))

    def test_import_bgen_variant_filtering_from_table(self):
        bgen_file = resource('example.8bits.bgen')

        everything = hl.import_bgen(bgen_file, ['GT'])
        self.assertEqual(everything.count(), (199, 500))

        desired_variants = everything.rows()

        actual = hl.import_bgen(bgen_file,
                                ['GT'],
                                n_partitions=10,
                                variants=desired_variants) # filtering with everything

        self.assertTrue(everything._same(actual))

    def test_import_bgen_locus_filtering_from_table(self):
        bgen_file = resource('example.8bits.bgen')

        desired_loci = hl.Table.parallelize([{'locus': hl.Locus('1', 10000)}],
                                            schema=hl.tstruct(locus=hl.tlocus()),
                                            key='locus')

        expected_result = [
            hl.Struct(locus=hl.Locus('1', 10000), alleles=['A', 'G']),
            hl.Struct(locus=hl.Locus('1', 10000), alleles=['A', 'G'])  # Duplicated variant
        ]

        result = hl.import_bgen(bgen_file,
                                ['GT'],
                                variants=desired_loci)

        self.assertEqual(result.rows().key_by('locus', 'alleles').select().collect(),
                        expected_result)

    def test_import_bgen_empty_variant_filter(self):
        bgen_file = resource('example.8bits.bgen')

        actual = hl.import_bgen(bgen_file,
                                ['GT'],
                                n_partitions=10,
                                variants=[])
        self.assertEqual(actual.count_rows(), 0)

        nothing = hl.import_bgen(bgen_file, ['GT']).filter_rows(False)
        self.assertEqual(nothing.count(), (0, 500))

        desired_variants = hl.struct(locus=nothing.locus, alleles=nothing.alleles)

        actual = hl.import_bgen(bgen_file,
                                ['GT'],
                                n_partitions=10,
                                variants=desired_variants)
        self.assertEqual(actual.count_rows(), 0)

    # FIXME testing block_size (in MB) requires large BGEN
    def test_n_partitions(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['dosage'],
                              n_partitions=210)
        self.assertEqual(bgen.n_partitions(), 199) # only 199 variants in the file

    def test_drop(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['dosage'])

        dr = bgen.filter_rows(False)
        self.assertEqual(dr._force_count_rows(), 0)
        self.assertEqual(dr._force_count_cols(), 500)

        dc = bgen.filter_cols(False)
        self.assertEqual(dc._force_count_rows(), 199)
        self.assertEqual(dc._force_count_cols(), 0)

    def test_index_multiple_bgen_files_does_not_fail_and_is_importable(self):
        original_bgen_files = [resource('random-b.bgen'), resource('random-c.bgen'), resource('random-a.bgen')]
        with hl.TemporaryFilename(suffix='.bgen') as f, \
             hl.TemporaryFilename(suffix='.bgen') as g, \
             hl.TemporaryFilename(suffix='.bgen') as h:
            newly_indexed_bgen_files = [f, g, h]
            for source, temp in zip(original_bgen_files, newly_indexed_bgen_files):
                hl.current_backend().fs.copy(source, temp)

            sample_file = resource('random.sample')
            hl.index_bgen(newly_indexed_bgen_files)

            actual = hl.import_bgen(newly_indexed_bgen_files, ['GT', 'GP'], sample_file, n_partitions=3)
            expected = hl.import_gen(resource('random.gen'), sample_file)

            assert actual._same(expected, tolerance=1.0 / 255, absolute=True)

    def test_multiple_files_variant_filtering(self):
        bgen_file = [resource('random-b.bgen'), resource('random-c.bgen'), resource('random-a.bgen')]
        alleles = ['A', 'G']

        desired_variants = [
            hl.Struct(locus=hl.Locus('20', 11), alleles=alleles),
            hl.Struct(locus=hl.Locus('20', 13), alleles=alleles),
            hl.Struct(locus=hl.Locus('20', 29), alleles=alleles),
            hl.Struct(locus=hl.Locus('20', 28), alleles=alleles),
            hl.Struct(locus=hl.Locus('20', 1), alleles=alleles),
            hl.Struct(locus=hl.Locus('20', 12), alleles=alleles),
        ]

        actual = hl.import_bgen(bgen_file,
                                ['GT'],
                                n_partitions=10,
                                variants=desired_variants)
        assert actual.count_rows() == 6

        everything = hl.import_bgen(bgen_file,
                                    ['GT'])
        assert everything.count() == (30, 10)

        expected = everything.filter_rows(hl.set(desired_variants).contains(everything.row_key))

        assert expected._same(actual)

    def test_multiple_files_disjoint(self):
        sample_file = resource('random.sample')
        bgen_file = [resource('random-b-disjoint.bgen'), resource('random-c-disjoint.bgen'), resource('random-a-disjoint.bgen')]
        with pytest.raises(FatalError, match='Each BGEN file must contain a region of the genome disjoint from other files'):
            hl.import_bgen(bgen_file, ['GT', 'GP'], sample_file, n_partitions=3)

    def test_multiple_references_throws_error(self):
        sample_file = resource('random.sample')
        bgen_file1 = resource('random-b.bgen')
        bgen_file2 = resource('random-c.bgen')

        with pytest.raises(FatalError, match='Found multiple reference genomes were specified in the BGEN index files'):
            hl.import_bgen([bgen_file1, bgen_file2],
                           ['GT'],
                           sample_file=sample_file,
                           index_file_map={
                               resource('random-b.bgen'): resource('random-b.bgen-NO-REFERENCE-GENOME.idx2'),
                               resource('random-c.bgen'): resource('random-c.bgen.idx2'),
                           })

    def test_old_index_file_throws_error(self):
        sample_file = resource('random.sample')
        bgen_file = resource('random.bgen')

        with hl.TemporaryFilename() as f:
            hl.current_backend().fs.copy(bgen_file, f)
            with pytest.raises(FatalError, match='have no .idx2 index file'):
                hl.import_bgen(f, ['GT', 'GP'], sample_file, n_partitions=3)

            try:
                with hl.current_backend().fs.open(f + '.idx', 'wb') as fobj:
                    fobj.write(b'')

                with pytest.raises(FatalError, match='have no .idx2 index file'):
                    hl.import_bgen(f, ['GT', 'GP'], sample_file)
            finally:
                hl.current_backend().fs.remove(f + '.idx')

    def test_specify_different_index_file(self):
        sample_file = resource('random.sample')
        bgen_file = resource('random.bgen')

        with hl.TemporaryDirectory(suffix='.idx2', ensure_exists=False) as index_file:
            index_file_map = {bgen_file: index_file}
            hl.index_bgen(bgen_file,
                          index_file_map=index_file_map)
            mt = hl.import_bgen(bgen_file,
                                ['GT', 'GP'],
                                sample_file,
                                index_file_map=index_file_map)
            assert mt.count() == (30, 10)

    def test_index_bgen_errors_when_index_file_has_wrong_extension(self):
        bgen_file = resource('random.bgen')

        with hl.TemporaryFilename(suffix='.idx') as index_file:
            with pytest.raises(FatalError, match='missing a .idx2 file extension'):
                index_file_map = {bgen_file: index_file}
                hl.index_bgen(bgen_file, index_file_map=index_file_map)

    def test_export_bgen(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GP'],
                              sample_file=resource('example.sample'))

        with hl.TemporaryDirectory(ensure_exists=False) as tmpdir:
            tmp = tmpdir + '/dataset'
            hl.export_bgen(bgen, tmp)
            hl.index_bgen(tmp + '.bgen')
            bgen2 = hl.import_bgen(tmp + '.bgen',
                                   entry_fields=['GP'],
                                   sample_file=tmp + '.sample')
            assert bgen._same(bgen2)

    def test_export_bgen_zstd(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GP'],
                              sample_file=resource('example.sample'))
        with hl.TemporaryDirectory(prefix='zstd', ensure_exists=False) as tmpdir:
            tmp = tmpdir + '/dataset'
            hl.export_bgen(bgen, tmp, compression_codec='zstd')
            hl.index_bgen(tmp + '.bgen')
            bgen2 = hl.import_bgen(tmp + '.bgen',
                                   entry_fields=['GP'],
                                   sample_file=tmp + '.sample')
            assert bgen._same(bgen2)

    def test_export_bgen_parallel(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GP'],
                              sample_file=resource('example.sample'),
                              n_partitions=3)
        with hl.TemporaryDirectory(ensure_exists=False) as tmpdir:
            tmp = tmpdir + '/dataset'
            hl.export_bgen(bgen, tmp, parallel='header_per_shard')
            hl.index_bgen(tmp + '.bgen')
            bgen2 = hl.import_bgen(tmp + '.bgen',
                                   entry_fields=['GP'],
                                   sample_file=tmp + '.sample')
            assert bgen._same(bgen2)

    def test_export_bgen_from_vcf(self):
        mt = hl.import_vcf(resource('sample.vcf'))

        with hl.TemporaryDirectory(ensure_exists=False) as tmpdir:
            tmp = tmpdir + '/dataset'
            hl.export_bgen(mt, tmp,
                           gp=hl.or_missing(
                               hl.is_defined(mt.GT),
                               hl.map(lambda i: hl.if_else(mt.GT.unphased_diploid_gt_index() == i, 1.0, 0.0),
                                      hl.range(0, hl.triangle(hl.len(mt.alleles))))))


            hl.index_bgen(tmp + '.bgen')
            bgen2 = hl.import_bgen(tmp + '.bgen',
                                   entry_fields=['GT'],
                                   sample_file=tmp + '.sample')
            mt = mt.select_entries('GT').select_rows().select_cols()
            bgen2 = bgen2.unfilter_entries().select_rows() # drop varid, rsid
            assert bgen2._same(mt)

    def test_randomness(self):
        alleles = ['A', 'G']

        desired_variants = [
            hl.Struct(locus=hl.Locus('1', 2000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 2001), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 4000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 10000), alleles=alleles),
            hl.Struct(locus=hl.Locus('1', 100001), alleles=alleles),
        ]

        bgen1 = hl.import_bgen(resource('example.8bits.bgen'),
                               entry_fields=['GT'],
                               sample_file=resource('example.sample'),
                               n_partitions=3)
        bgen1 = bgen1.filter_rows(hl.literal(desired_variants).contains(bgen1.row_key))
        c1 = bgen1.filter_entries(hl.rand_bool(0.2, seed=1234))

        bgen2 = hl.import_bgen(resource('example.8bits.bgen'),
                               entry_fields=['GT'],
                               sample_file=resource('example.sample'),
                               n_partitions=5,
                               variants=desired_variants)

        c2 = bgen2.filter_entries(hl.rand_bool(0.2, seed=1234))
        assert c1._same(c2)


class GENTests(unittest.TestCase):
    def test_import_gen(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            contig_recoding={"01": "1"},
                            reference_genome = 'GRCh37').rows()
        self.assertTrue(gen.all(gen.locus.contig == "1"))
        self.assertEqual(gen.count(), 199)
        self.assertEqual(gen.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_gen_no_chromosome_in_file(self):
        gen = hl.import_gen(resource('no_chromosome.gen'),
                            resource('skip_invalid_loci.sample'),
                            chromosome="1",
                            reference_genome=None,
                            skip_invalid_loci=True)

        self.assertEqual(gen.aggregate_rows(hl.agg.all(gen.locus.contig == "1")), True)

    def test_import_gen_no_reference_specified(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            reference_genome=None)

        self.assertEqual(gen.locus.dtype,
                         hl.tstruct(contig=hl.tstr, position=hl.tint32))
        self.assertEqual(gen.count_rows(), 199)

    def test_import_gen_skip_invalid_loci(self):
        mt = hl.import_gen(resource('skip_invalid_loci.gen'),
                           resource('skip_invalid_loci.sample'),
                           reference_genome='GRCh37',
                           skip_invalid_loci=True)
        self.assertEqual(mt._force_count_rows(), 3)

        with self.assertRaisesRegex(FatalError, 'Invalid locus'):
            mt = hl.import_gen(resource('skip_invalid_loci.gen'),
                               resource('skip_invalid_loci.sample'))
            mt._force_count_rows()

    def test_export_gen(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            contig_recoding={"01": "1"},
                            reference_genome='GRCh37',
                            min_partitions=3)

        # permute columns so not in alphabetical order!
        import random
        indices = list(range(gen.count_cols()))
        random.shuffle(indices)
        gen = gen.choose_cols(indices)

        file = new_temp_file()
        hl.export_gen(gen, file)
        gen2 = hl.import_gen(file + '.gen',
                             sample_file=file + '.sample',
                             reference_genome='GRCh37',
                             min_partitions=3)

        self.assertTrue(gen._same(gen2, tolerance=3E-4, absolute=True))

    def test_export_gen_exprs(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            contig_recoding={"01": "1"},
                            reference_genome='GRCh37',
                            min_partitions=3).add_col_index().add_row_index()

        out1 = new_temp_file()
        hl.export_gen(gen, out1, id1=hl.str(gen.col_idx), id2=hl.str(gen.col_idx), missing=0.5,
                      varid=hl.str(gen.row_idx), rsid=hl.str(gen.row_idx), gp=[0.0, 1.0, 0.0])

        in1 = (hl.import_gen(out1 + '.gen', sample_file=out1 + '.sample', min_partitions=3)
               .add_col_index()
               .add_row_index())
        self.assertTrue(in1.aggregate_entries(hl.agg.fraction((hl.is_missing(in1.GP) | (in1.GP == [0.0, 1.0, 0.0])) == 1.0)))
        self.assertTrue(in1.aggregate_rows(hl.agg.fraction((in1.varid == hl.str(in1.row_idx)) &
                                                           (in1.rsid == hl.str(in1.row_idx)))) == 1.0)
        self.assertTrue(in1.aggregate_cols(hl.agg.fraction((in1.s == hl.str(in1.col_idx)))))


class LocusIntervalTests(unittest.TestCase):
    def test_import_locus_intervals(self):
        interval_file = resource('annotinterall.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome='GRCh37')
        nint = t.count()

        i = 0
        with hl.current_backend().fs.open(interval_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nint, i)
        self.assertEqual(t.interval.dtype.point_type, hl.tlocus('GRCh37'))

        tmp_file = new_temp_file(prefix="test", extension="interval_list")
        start = t.interval.start
        end = t.interval.end
        (t
         .key_by(interval=hl.locus_interval(start.contig, start.position, end.position, True, True))
         .select()
         .export(tmp_file, header=False))

        t2 = hl.import_locus_intervals(tmp_file)

        self.assertTrue(t.select()._same(t2))

    def test_import_locus_intervals_no_reference_specified(self):
        interval_file = resource('annotinterall.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome=None)
        self.assertEqual(t.count(), 2)
        self.assertEqual(t.interval.dtype.point_type, hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_locus_intervals_recoding(self):
        interval_file = resource('annotinterall.grch38.no.chr.interval_list')
        t = hl.import_locus_intervals(interval_file,
                                      contig_recoding={str(i): f'chr{i}' for i in [*range(1, 23), 'X', 'Y', 'M']},
                                      reference_genome='GRCh38')
        self.assertEqual(t._force_count(), 3)
        self.assertEqual(t.interval.dtype.point_type, hl.tlocus('GRCh38'))

    def test_import_locus_intervals_badly_defined_intervals(self):
        interval_file = resource('example3.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome='GRCh37', skip_invalid_intervals=True)
        self.assertEqual(t.count(), 21)

        t = hl.import_locus_intervals(interval_file, reference_genome=None, skip_invalid_intervals=True)
        self.assertEqual(t.count(), 22)

    def test_import_bed(self):
        bed_file = resource('example1.bed')
        bed = hl.import_bed(bed_file, reference_genome='GRCh37')

        nbed = bed.count()
        i = 0
        with hl.hadoop_open(bed_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    try:
                        int(line.split()[0])
                        i += 1
                    except:
                        pass
        self.assertEqual(nbed, i)

        self.assertEqual(bed.interval.dtype.point_type, hl.tlocus('GRCh37'))

        bed_file = resource('example2.bed')
        t = hl.import_bed(bed_file, reference_genome='GRCh37')
        self.assertEqual(t.interval.dtype.point_type, hl.tlocus('GRCh37'))
        self.assertEqual(list(t.key.dtype), ['interval'])
        self.assertEqual(list(t.row.dtype), ['interval','target'])

        expected = [hl.interval(hl.locus('20', 1), hl.locus('20', 11), True, False),   # 20    0 10      gene0
                    hl.interval(hl.locus('20', 2), hl.locus('20', 14000001), True, False),  # 20    1          14000000  gene1
                    hl.interval(hl.locus('20', 5), hl.locus('20', 6), False, False),  # 20    5   5   gene4
                    hl.interval(hl.locus('20', 17000001), hl.locus('20', 18000001), True, False),  # 20    17000000   18000000  gene2
                    hl.interval(hl.locus('20', 63025511), hl.locus('20', 63025520), True, True)]  # 20    63025510   63025520  gene3

        self.assertEqual(t.interval.collect(), hl.eval(expected))

    def test_import_bed_recoding(self):
        bed_file = resource('some-missing-chr-grch38.bed')
        bed = hl.import_bed(bed_file,
                            reference_genome='GRCh38',
                            contig_recoding={str(i): f'chr{i}' for i in [*range(1, 23), 'X', 'Y', 'M']})
        self.assertEqual(bed._force_count(), 5)
        self.assertEqual(bed.interval.dtype.point_type, hl.tlocus('GRCh38'))

    def test_import_bed_no_reference_specified(self):
        bed_file = resource('example1.bed')
        t = hl.import_bed(bed_file, reference_genome=None)
        self.assertEqual(t.count(), 3)
        self.assertEqual(t.interval.dtype.point_type, hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_bed_kwargs_to_import_table(self):
        bed_file = resource('example2.bed')
        t = hl.import_bed(bed_file, reference_genome='GRCh37', find_replace=('gene', ''))
        self.assertFalse('gene1' in t.aggregate(hl.agg.collect_as_set(t.target)))

    def test_import_bed_badly_defined_intervals(self):
        bed_file = resource('example4.bed')
        t = hl.import_bed(bed_file, reference_genome='GRCh37', skip_invalid_intervals=True)
        self.assertEqual(t.count(), 3)

        t = hl.import_bed(bed_file, reference_genome=None, skip_invalid_intervals=True)
        self.assertEqual(t.count(), 4)

    def test_pass_through_args(self):
        interval_file = resource('example3.interval_list')
        t = hl.import_locus_intervals(interval_file,
                                      reference_genome='GRCh37',
                                      skip_invalid_intervals=True,
                                      filter=r'target_\d\d')
        assert t.count() == 9


class ImportMatrixTableTests(unittest.TestCase):
    def test_import_matrix_table(self):
        mt = hl.import_matrix_table(doctest_resource('matrix1.tsv'),
                                    row_fields={'Barcode': hl.tstr, 'Tissue': hl.tstr, 'Days': hl.tfloat32})
        self.assertEqual(mt['Barcode']._indices, mt._row_indices)
        self.assertEqual(mt['Tissue']._indices, mt._row_indices)
        self.assertEqual(mt['Days']._indices, mt._row_indices)
        self.assertEqual(mt['col_id']._indices, mt._col_indices)
        self.assertEqual(mt['row_id']._indices, mt._row_indices)

        mt.count()

        row_fields = {'f0': hl.tstr, 'f1': hl.tstr, 'f2': hl.tfloat32}
        hl.import_matrix_table(doctest_resource('matrix2.tsv'),
                               row_fields=row_fields, row_key=[])._force_count_rows()
        hl.import_matrix_table(doctest_resource('matrix3.tsv'),
                               row_fields=row_fields,
                               no_header=True)._force_count_rows()
        hl.import_matrix_table(doctest_resource('matrix3.tsv'),
                               row_fields=row_fields,
                               no_header=True,
                               row_key=[])._force_count_rows()

    def test_import_matrix_table_no_cols(self):
        fields = {'Chromosome': hl.tstr, 'Position': hl.tint32, 'Ref': hl.tstr, 'Alt': hl.tstr, 'Rand1': hl.tfloat64, 'Rand2': hl.tfloat64}
        file = resource('sample2_va_nomulti.tsv')
        mt = hl.import_matrix_table(file, row_fields=fields, row_key=['Chromosome', 'Position'])
        t = hl.import_table(file, types=fields, key=['Chromosome', 'Position'])

        self.assertEqual(mt.count_cols(), 0)
        self.assertEqual(mt.count_rows(), 231)
        self.assertTrue(t._same(mt.rows()))

    def test_import_matrix_comment(self):
        no_comment = doctest_resource('matrix1.tsv')
        comment = doctest_resource('matrix1_comment.tsv')
        row_fields={'Barcode': hl.tstr, 'Tissue': hl.tstr, 'Days': hl.tfloat32}
        mt1 = hl.import_matrix_table(no_comment,
                                     row_fields=row_fields,
                                     row_key=[])
        mt2 = hl.import_matrix_table(comment,
                                     row_fields=row_fields,
                                     row_key=[],
                                     comment=['#', '%'])
        assert mt1._same(mt2)

    def test_headers_not_identical(self):
        with pytest.raises(ValueError, match='invalid header: lengths of headers differ'):
            hl.import_matrix_table([resource("sampleheader1.txt"), resource("sampleheader2.txt")],
                                   row_fields={'f0': hl.tstr}, row_key=['f0'])

    def test_headers_same_len_diff_elem(self):
        with pytest.raises(ValueError, match='invalid header: expected elements to be identical for all input paths'):
            hl.import_matrix_table([resource("sampleheader2.txt"),
                                   resource("sampleheaderdiffelem.txt")], row_fields={'f0': hl.tstr}, row_key=['f0'])

    def test_too_few_entries(self):
        def boom():
            hl.import_matrix_table(resource("samplesmissing.txt"),
                                   row_fields={'f0': hl.tstr},
                                   row_key=['f0']
                                   )._force_count_rows()
        with pytest.raises(HailUserError, match='unexpected end of line while reading entries'):
            boom()

    def test_wrong_row_field_type(self):
        with pytest.raises(HailUserError, match="error parsing value into int32 at row field 'f0'"):
            hl.import_matrix_table(resource("sampleheader1.txt"),
                                   row_fields={'f0': hl.tint32},
                                   row_key=['f0'])._force_count_rows()

    def test_wrong_entry_type(self):
        with pytest.raises(HailUserError, match="error parsing value into int32 at column id 'col000003'"):
            hl.import_matrix_table(resource("samplenonintentries.txt"),
                                   row_fields={'f0': hl.tstr},
                                   row_key=['f0'])._force_count_rows()

    def test_key_by_after_empty_key_import(self):
        fields = {'Chromosome':hl.tstr,
                  'Position': hl.tint32,
                  'Ref': hl.tstr,
                  'Alt': hl.tstr}
        mt = hl.import_matrix_table(resource('sample2_va_nomulti.tsv'),
                                    row_fields=fields,
                                    row_key=[],
                                    entry_type=hl.tfloat)
        mt = mt.key_rows_by('Chromosome', 'Position')
        assert 0.001 < abs(0.50965 - mt.aggregate_entries(hl.agg.mean(mt.x)))

    def test_key_by_after_empty_key_import(self):
        fields = {'Chromosome':hl.tstr,
                  'Position': hl.tint32,
                  'Ref': hl.tstr,
                  'Alt': hl.tstr}
        mt = hl.import_matrix_table(resource('sample2_va_nomulti.tsv'),
                                    row_fields=fields,
                                    row_key=[],
                                    entry_type=hl.tfloat)
        mt = mt.key_rows_by('Chromosome', 'Position')
        mt._force_count_rows()

    def test_devilish_nine_separated_eight_missing_file(self):
        fields = {'chr': hl.tstr,
                  '': hl.tint32,
                  'ref': hl.tstr,
                  'alt': hl.tstr}
        mt = hl.import_matrix_table(resource('import_matrix_table_devlish.ninesv'),
                                    row_fields=fields,
                                    row_key=['chr', ''],
                                    sep='9',
                                    missing='8')
        actual = mt.x.collect()
        expected = [
            1, 2, 3, 4,
            11, 12, 13, 14,
            21, 22, 23, 24,
            31, None, None, 34]
        assert actual == expected

        assert mt.count_rows() == len(mt.rows().collect())

        actual = mt.chr.collect()
        assert actual == ['chr1', 'chr1', 'chr1', None]
        actual = mt[''].collect()
        assert actual == [1, 10, 101, None]
        actual = mt.ref.collect()
        assert actual == ['A', 'AGT', None, 'CTA']
        actual = mt.alt.collect()
        assert actual == ['T', 'TGG', 'A', None]

    def test_empty_import_matrix_table(self):
        path = new_temp_file(extension='tsv.bgz')
        mt = hl.utils.range_matrix_table(0, 0)
        mt = mt.annotate_entries(x=1)
        mt.x.export(path)
        assert hl.import_matrix_table(path)._force_count_rows() == 0

        mt.x.export(path, header=False)
        assert hl.import_matrix_table(path, no_header=True)._force_count_rows() == 0

    def test_import_row_id_multiple_partitions(self):
        path = new_temp_file(extension='txt')
        (hl.utils.range_matrix_table(50, 50)
         .annotate_entries(x=1)
         .key_rows_by()
         .key_cols_by()
         .x
         .export(path, header=False, delimiter=' '))

        mt = hl.import_matrix_table(path,
                                    no_header=True,
                                    entry_type=hl.tint32,
                                    delimiter=' ',
                                    min_partitions=10)
        assert mt.row_id.collect() == list(range(50))

    def test_long_parsing(self):
        path = resource('text_matrix_longs.tsv')
        mt = hl.import_matrix_table(path, row_fields={'foo': hl.tint64})
        collected = mt.entries().collect()
        assert collected == [
            hl.utils.Struct(foo=7, row_id=0, col_id='s1', x=1234),
            hl.utils.Struct(foo=7, row_id=0, col_id='s2', x=2345)
        ]


@pytest.mark.parametrize("entry_fun", [hl.str, hl.int32, hl.float64])
@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("delimiter", [',', ' '])
@pytest.mark.parametrize("missing", ['.', '9'])
def test_import_matrix_table_round_trip(missing, delimiter, header, entry_fun):
    mt = hl.utils.range_matrix_table(10, 10, n_partitions=2)
    mt = mt.annotate_entries(x = entry_fun(mt.row_idx * mt.col_idx))
    mt = mt.annotate_rows(row_str = hl.str(mt.row_idx))
    mt = mt.annotate_rows(row_float = hl.float(mt.row_idx))

    entry_type = mt.x.dtype

    path = new_temp_file(extension='tsv')
    mt.key_rows_by(*mt.row).x.export(path,
                                     missing=missing,
                                     delimiter=delimiter,
                                     header=header)

    row_fields = {f: mt.row[f].dtype for f in mt.row}
    row_key = 'row_idx'

    if not header:
        pseudonym = {'row_idx': 'f0',
                     'row_str': 'f1',
                     'row_float': 'f2'}
        row_fields = {pseudonym[k]: v for k, v in row_fields.items()}
        row_key = pseudonym[row_key]
        mt = mt.rename(pseudonym)
    else:
        mt = mt.key_cols_by(col_idx=hl.str(mt.col_idx))

    actual = hl.import_matrix_table(
        path,
        row_fields=row_fields,
        row_key=row_key,
        entry_type=entry_type,
        missing=missing,
        no_header=not header,
        sep=delimiter)
    actual = actual.rename({'col_id': 'col_idx'})

    row_key = mt.row_key
    col_key = mt.col_key
    mt = mt.key_rows_by()
    mt = mt.annotate_entries(
        x = hl.if_else(hl.str(mt.x) == missing,
                       hl.missing(entry_type),
                       mt.x))
    mt = mt.annotate_rows(**{
        f: hl.if_else(hl.str(mt[f]) == missing,
                      hl.missing(mt[f].dtype),
                      mt[f])
        for f in mt.row})
    mt = mt.key_rows_by(*row_key)
    assert mt._same(actual)


class ImportLinesTest(unittest.TestCase):
    def test_import_lines(self):
        lines_table = hl.import_lines(resource('example.gen'))
        first_row = lines_table.head(1).collect()[0]
        assert lines_table.row.dtype == hl.tstruct(file=hl.tstr, text=hl.tstr)
        assert "01 SNPID_2 RSID_2 2000 A G 0 0 0 0.0278015 0.00863647 0.963531 0.0173645" in first_row.text
        assert "example.gen" in first_row.file
        assert lines_table._force_count() == 199

    def test_import_lines_multiple_files(self):
        lines_table = hl.import_lines((resource('first_half_example.gen'), resource('second_half_example.gen')))
        first_row = lines_table.head(1).collect()[0]
        last_row = lines_table.tail(1).collect()[0]
        assert "01 SNPID_2 RSID_2 2000 A G 0 0 0 0.0278015 0.00863647 0.963531 0.0173645" in first_row.text
        assert "first_half_example.gen" in first_row.file
        assert "second_half_example.gen" in last_row.file
        assert lines_table._force_count() == 199

    def test_import_lines_glob(self):
        lines_table = hl.import_lines(resource('*_half_example.gen'))
        assert lines_table._force_count() == 199

    def test_import_lines_bgz(self):
        lines_table = hl.import_lines(resource('sample.vcf.gz'), min_partitions=5, force_bgz=True)
        assert lines_table.n_partitions() == 5


class ImportTableTests(unittest.TestCase):
    def test_import_table_force_bgz(self):
        fs = hl.current_backend().fs
        f = new_temp_file(extension="bgz")
        t = hl.utils.range_table(10, 5)
        t.export(f)

        f2 = new_temp_file(extension="gz")
        fs.copy(f, f2)
        t2 = hl.import_table(f2, force_bgz=True, impute=True).key_by('idx')
        self.assertTrue(t._same(t2))

    def test_import_table_empty(self):
        try:
            rows = hl.import_table(resource('empty.tsv')).collect()
        except ValueError as err:
            assert f'Invalid file: no lines remaining after filters\n Files provided: {resource("empty.tsv")}' in err.args[0]
        else:
            assert False, rows

    def test_import_table_empty_with_header(self):
        assert [] == hl.import_table(resource('empty-with-header.tsv')).collect()

    def test_glob(self):
        tables = hl.import_table(resource('variantAnnotations.split.*.tsv'))
        assert tables.count() == 346

    def test_type_imputation(self):
        ht = hl.import_table(resource('type_imputation.tsv'), delimiter=' ', missing='.', impute=True)
        assert ht.row.dtype == hl.dtype('struct{1:int32,2:float64,3:str,4:str,5:str,6:bool,7:str}')

        ht = hl.import_table(resource('variantAnnotations.tsv'), impute=True)
        assert ht.row.dtype == hl.dtype(
            'struct{Chromosome: int32, Position: int32, Ref: str, Alt: str, Rand1: float64, Rand2: float64, Gene: str}')

        ht = hl.import_table(resource('variantAnnotations.tsv'), impute=True, types={'Chromosome': 'str'})
        assert ht.row.dtype == hl.dtype(
            'struct{Chromosome: str, Position: int32, Ref: str, Alt: str, Rand1: float64, Rand2: float64, Gene: str}')

        ht = hl.import_table(resource('variantAnnotations.alternateformat.tsv'), impute=True)
        assert ht.row.dtype == hl.dtype(
            'struct{`Chromosome:Position:Ref:Alt`: str, Rand1: float64, Rand2: float64, Gene: str}')

        ht = hl.import_table(resource('sampleAnnotations.tsv'), impute=True)
        assert ht.row.dtype == hl.dtype(
            'struct{Sample: str, Status: str, qPhen: int32}')

        ht = hl.import_table(resource('integer_imputation.txt'), impute=True, delimiter=r'\s+')
        assert ht.row.dtype == hl.dtype(
            'struct{A:int64, B:int32}')

    def test_import_export_identity(self):
        fs = hl.current_backend().fs
        ht = hl.import_table(resource('sampleAnnotations.tsv'))
        f = new_temp_file()
        ht.export(f)

        with fs.open(resource('sampleAnnotations.tsv'), 'r') as i1:
            expected = list(line.strip() for line in i1)
        with fs.open(f, 'r') as i2:
            observed = list(line.strip() for line in i2)

        assert expected == observed

    def small_dataset_1(self):
        data = [
            hl.Struct(Sample='Sample1',field1=5,field2=5),
            hl.Struct(Sample='Sample2',field1=3,field2=5),
            hl.Struct(Sample='Sample3',field1=2,field2=5),
            hl.Struct(Sample='Sample4',field1=1,field2=5),
        ]
        return hl.Table.parallelize(data, key='Sample')

    def test_source_file(self):
        ht = hl.import_table(resource('variantAnnotations.split.*.tsv'), source_file_field='source')
        ht = ht.add_index()
        assert ht.aggregate(hl.agg.all(
            hl.if_else(ht.idx < 239,
                       ht.source.endswith('variantAnnotations.split.1.tsv'),
                       ht.source.endswith('variantAnnotations.split.2.tsv'))))


    def test_read_write_identity(self):
        ht = self.small_dataset_1()
        f = new_temp_file(extension='ht')
        ht.write(f)
        assert ht._same(hl.read_table(f))

    def test_read_write_identity_keyed(self):
        ht = self.small_dataset_1().key_by()
        f = new_temp_file(extension='ht')
        ht.write(f)
        assert ht._same(hl.read_table(f))

    def test_import_same(self):
        ht = hl.import_table(resource('sampleAnnotations.tsv'))
        ht2 = hl.import_table(resource('sampleAnnotations.tsv'))
        assert ht._same(ht2)

    def test_error_with_context(self):
        with pytest.raises(HailUserError, match='cannot parse int32 from input string'):
            ht = hl.import_table(resource('tsv_errors.tsv'), types={'col1': 'int32'})
            ht._force_count()

        with pytest.raises(HailUserError, match='Expected 2 fields, found 1 field'):
            ht = hl.import_table(resource('tsv_errors.tsv'), impute=True)


class GrepTests(unittest.TestCase):
    @fails_local_backend()
    def test_grep_show_false(self):
        from hail.backend.service_backend import ServiceBackend
        if isinstance(hl.current_backend(), ServiceBackend):
            prefix = resource('')
        else:
            prefix = ''
        expected = {prefix + 'sampleAnnotations.tsv': ['HG00120\tCASE\t19599', 'HG00121\tCASE\t4832'],
                    prefix + 'sample2_rename.tsv': ['HG00120\tB_HG00120', 'HG00121\tB_HG00121'],
                    prefix + 'sampleAnnotations2.tsv': ['HG00120\t3919.8\t19589',
                                                        'HG00121\t966.4\t4822',
                                                        'HG00120_B\t3919.8\t19589',
                                                        'HG00121_B\t966.4\t4822',
                                                        'HG00120_B_B\t3919.8\t19589',
                                                        'HG00121_B_B\t966.4\t4822']}

        assert hl.grep('HG0012[0-1]', resource('*.tsv'), show=False) == expected


class AvroTests(unittest.TestCase):
    @fails_service_backend(reason='''
E                   java.io.NotSerializableException: org.apache.avro.Schema$RecordSchema
E                   	at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1184)
E                   	at java.io.ObjectOutputStream.writeArray(ObjectOutputStream.java:1378)
E                   	at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1174)
E                   	at java.io.ObjectOutputStream.defaultWriteFields(ObjectOutputStream.java:1548)
E                   	at java.io.ObjectOutputStream.writeSerialData(ObjectOutputStream.java:1509)
E                   	at java.io.ObjectOutputStream.writeOrdinaryObject(ObjectOutputStream.java:1432)
E                   	at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1178)
E                   	at java.io.ObjectOutputStream.writeArray(ObjectOutputStream.java:1378)
E                   	at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1174)
E                   	at java.io.ObjectOutputStream.defaultWriteFields(ObjectOutputStream.java:1548)
E                   	at java.io.ObjectOutputStream.writeSerialData(ObjectOutputStream.java:1509)
E                   	at java.io.ObjectOutputStream.writeOrdinaryObject(ObjectOutputStream.java:1432)
E                   	at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1178)
E                   	at java.io.ObjectOutputStream.writeObject(ObjectOutputStream.java:348)
E                   	at is.hail.backend.service.ServiceBackend.$anonfun$parallelizeAndComputeWithIndex$3(ServiceBackend.scala:119)
E                   	at is.hail.backend.service.ServiceBackend.$anonfun$parallelizeAndComputeWithIndex$3$adapted(ServiceBackend.scala:118)
E                   	at is.hail.utils.package$.using(package.scala:638)
E                   	at is.hail.backend.service.ServiceBackend.$anonfun$parallelizeAndComputeWithIndex$2(ServiceBackend.scala:118)
E                   	at scala.runtime.java8.JFunction0$mcV$sp.apply(JFunction0$mcV$sp.java:23)
E                   	at is.hail.services.package$.retryTransientErrors(package.scala:71)
E                   	at is.hail.backend.service.ServiceBackend.$anonfun$parallelizeAndComputeWithIndex$1(ServiceBackend.scala:117)
E                   	at scala.runtime.java8.JFunction0$mcV$sp.apply(JFunction0$mcV$sp.java:23)
E                   	at scala.concurrent.Future$.$anonfun$apply$1(Future.scala:659)
E                   	at scala.util.Success.$anonfun$map$1(Try.scala:255)
E                   	at scala.util.Success.map(Try.scala:213)
E                   	at scala.concurrent.Future.$anonfun$map$1(Future.scala:292)
E                   	at scala.concurrent.impl.Promise.liftedTree1$1(Promise.scala:33)
E                   	at scala.concurrent.impl.Promise.$anonfun$transform$1(Promise.scala:33)
E                   	at scala.concurrent.impl.CallbackRunnable.run(Promise.scala:64)
E                   	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
E                   	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
E                   	at java.lang.Thread.run(Thread.java:748)
''')
    def test_simple_avro(self):
        avro_file = resource('avro/weather.avro')
        fs = hl.current_backend().fs
        with DataFileReader(fs.open(avro_file, 'rb'), DatumReader()) as avro:
            expected = list(avro)
        data = hl.import_avro([avro_file]).collect()
        data = [dict(**s) for s in data]
        self.assertEqual(expected, data)


def test_matrix_and_table_read_intervals_with_hidden_key():
    f1 = new_temp_file()
    f2 = new_temp_file()
    f3 = new_temp_file()

    mt = hl.utils.range_matrix_table(50, 5, 10)
    mt = mt.key_rows_by(x=mt.row_idx, y=mt.row_idx + 2)
    mt.write(f1)

    hl.read_matrix_table(f1).key_rows_by('x').write(f2)
    hl.read_matrix_table(f1).key_rows_by('x').rows().write(f3)

    hl.read_matrix_table(f2, _intervals=[hl.Interval(0, 3)])._force_count_rows()
    hl.read_table(f3, _intervals=[hl.Interval(0, 3)])._force_count()
