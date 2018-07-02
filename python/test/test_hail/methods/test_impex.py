import hail as hl
from ..helpers import *
from hail.utils import new_temp_file, FatalError, run_command, uri_path
import unittest
import os

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

    def test_import_vcf(self):
        vcf = hl.split_multi_hts(
            hl.import_vcf(resource('sample2.vcf'),
                          reference_genome=hl.get_reference('GRCh38'),
                          contig_recoding={"22": "chr22"}))

        vcf_table = vcf.rows()
        self.assertTrue(vcf_table.all(vcf_table.locus.contig == "chr22"))
        self.assertTrue(vcf.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_vcf_no_reference_specified(self):
        vcf = hl.import_vcf(resource('sample2.vcf'),
                            reference_genome=None)
        self.assertTrue(vcf.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))
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

    def test_import_vcf_skip_invalid_loci(self):
        mt = hl.import_vcf(resource('skip_invalid_loci.vcf'), reference_genome='GRCh37',
                           skip_invalid_loci=True)
        self.assertTrue(mt._force_count_rows() == 3)

        with self.assertRaisesRegex(FatalError, 'Invalid locus'):
            hl.import_vcf(resource('skip_invalid_loci.vcf'))

    def test_export_vcf(self):
        dataset = hl.import_vcf(resource('sample.vcf.bgz'))
        vcf_metadata = hl.get_vcf_metadata(resource('sample.vcf.bgz'))
        hl.export_vcf(dataset, '/tmp/sample.vcf', metadata=vcf_metadata)
        dataset_imported = hl.import_vcf('/tmp/sample.vcf')
        self.assertTrue(dataset._same(dataset_imported))

        metadata_imported = hl.get_vcf_metadata('/tmp/sample.vcf')
        self.assertDictEqual(vcf_metadata, metadata_imported)


class PLINKTests(unittest.TestCase):
    def test_import_fam(self):
        fam_file = resource('sample.fam')
        nfam = hl.import_fam(fam_file).count()
        i = 0
        with open(fam_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nfam, i)
        
    def test_export_import_plink_same(self):
        mt = get_dataset()
        mt = mt.select_rows(rsid=hl.delimit([mt.locus.contig, hl.str(mt.locus.position), mt.alleles[0], mt.alleles[1]], ':'),
                            cm_position=15.0)
        mt = mt.select_cols(fam_id=hl.null(hl.tstr), pat_id=hl.null(hl.tstr), mat_id=hl.null(hl.tstr),
                            is_female=hl.null(hl.tbool), is_case=hl.null(hl.tbool))
        mt = mt.select_entries('GT')

        bfile = '/tmp/test_import_export_plink'
        hl.export_plink(mt, bfile, ind_id=mt.s, cm_position=mt.cm_position)

        mt_imported = hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam',
                                      a2_reference=True, reference_genome='GRCh37')
        self.assertTrue(mt._same(mt_imported))
        self.assertTrue(mt.aggregate_rows(hl.agg.all(mt.cm_position == 15.0)))

    def test_import_plink_empty_fam(self):
        mt = get_dataset().drop_cols()
        bfile = '/tmp/test_empty_fam'
        hl.export_plink(mt, bfile, ind_id=mt.s)
        with self.assertRaisesRegex(FatalError, "Empty .fam file"):
            hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam')

    def test_import_plink_empty_bim(self):
        mt = get_dataset().drop_rows()
        bfile = '/tmp/test_empty_bim'
        hl.export_plink(mt, bfile, ind_id=mt.s)
        with self.assertRaisesRegex(FatalError, ".bim file does not contain any variants"):
            hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam')

    def test_import_plink_a1_major(self):
        mt = get_dataset()
        bfile = '/tmp/sample_plink'
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

    def test_import_plink_contig_recoding_w_reference(self):
        vcf = hl.split_multi_hts(
            hl.import_vcf(resource('sample2.vcf'),
                          reference_genome=hl.get_reference('GRCh38'),
                          contig_recoding={"22": "chr22"}))

        hl.export_plink(vcf, '/tmp/sample_plink')

        bfile = '/tmp/sample_plink'
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
        self.assertTrue(
            plink.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_plink_skip_invalid_loci(self):
        mt = hl.import_plink(resource('skip_invalid_loci.bed'),
                             resource('skip_invalid_loci.bim'),
                             resource('skip_invalid_loci.fam'),
                             reference_genome='GRCh37',
                             skip_invalid_loci=True)
        self.assertTrue(mt._force_count_rows() == 3)

        with self.assertRaisesRegex(FatalError, 'Invalid locus'):
            hl.import_plink(resource('skip_invalid_loci.bed'),
                            resource('skip_invalid_loci.bim'),
                            resource('skip_invalid_loci.fam'))

    def test_export_plink(self):
        vcf_file = resource('sample.vcf')
        mt = hl.split_multi_hts(hl.import_vcf(vcf_file, min_partitions=10))

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
            

# this routine was used to generate resources random.gen, random.sample
# random.bgen was generated with qctool v2.0rc9:
# qctool -g random.gen -s random.sample -bgen-bits 8 -og random.bgen
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
    def test_import_bgen_dosage_entry(self):
        hl.index_bgen(resource('example.8bits.bgen'))

        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['dosage'],
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct(dosage=hl.tfloat64))
        self.assertEqual(bgen.count_rows(), 199)

    def test_import_bgen_GT_GP_entries(self):
        hl.index_bgen(resource('example.8bits.bgen'))

        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GT', 'GP'],
                              sample_file=resource('example.sample'),
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct(GT=hl.tcall, GP=hl.tarray(hl.tfloat64)))

    def test_import_bgen_no_entries(self):
        hl.index_bgen(resource('example.8bits.bgen'))

        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=[],
                              sample_file=resource('example.sample'),
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct())
        bgen._jvds.typecheck()

    def test_import_bgen_no_reference(self):
        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GT', 'GP', 'dosage'],
                              contig_recoding={'01': '1'},
                              reference_genome=None)
        self.assertEqual(bgen.locus.dtype, hl.tstruct(contig=hl.tstr, position=hl.tint32))
        self.assertEqual(bgen.count_rows(), 199)

    def test_import_bgen_skip_invalid_loci(self):
        hl.index_bgen(resource('skip_invalid_loci.bgen'))

        mt = hl.import_bgen(resource('skip_invalid_loci.bgen'),
                            entry_fields=[],
                            sample_file=resource('skip_invalid_loci.sample'),
                            reference_genome='GRCh37',
                            skip_invalid_loci=True)
        self.assertTrue(mt._force_count_rows() == 3)

        with self.assertRaisesRegex(FatalError, 'Invalid locus'):
            hl.import_bgen(resource('skip_invalid_loci.bgen'),
                           entry_fields=[],
                           sample_file=resource('skip_invalid_loci.sample'))

    def test_import_bgen_gavin_example(self):
        recoding = {'0{}'.format(i): str(i) for i in range(1, 10)}

        sample_file = resource('example.sample')
        genmt = hl.import_gen(resource('example.gen'), sample_file,
                              contig_recoding=recoding)

        bgen_file = resource('example.8bits.bgen')
        hl.index_bgen(bgen_file)
        bgenmt = hl.import_bgen(bgen_file, ['GT', 'GP'], sample_file,
                            contig_recoding=recoding)
        self.assertTrue(
            bgenmt._same(genmt, tolerance=1.0 / 255, absolute=True))

    def test_import_bgen_random(self):
        sample_file = resource('random.sample')
        genmt = hl.import_gen(resource('random.gen'), sample_file)

        bgen_file = resource('random.bgen')
        hl.index_bgen(bgen_file)
        bgenmt = hl.import_bgen(bgen_file, ['GT', 'GP'], sample_file)
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
        hl.index_bgen(bgen_file)

        bgenmt = hl.import_bgen(bgen_file, ['GP', 'dosage'], sample_file,
                                contig_recoding=recoding)
        et = bgenmt.entries()
        et = et.transmute(gp_dosage = hl.gp_dosage(et.GP))
        self.assertTrue(et.all(
            (hl.is_missing(et.dosage) & hl.is_missing(et.gp_dosage)) |
            (hl.abs(et.dosage - et.gp_dosage) < 1e-6)))

    def test_import_bgen_row_fields(self):
        default_row_fields = hl.import_bgen(resource('example.8bits.bgen'),
                                            entry_fields=['dosage'],
                                            contig_recoding={'01': '1'},
                                            reference_genome='GRCh37')
        self.assertEqual(default_row_fields.row.dtype,
                         hl.tstruct(locus=hl.tlocus('GRCh37'),
                                    alleles=hl.tarray(hl.tstr),
                                    rsid=hl.tstr,
                                    varid=hl.tstr))
        no_row_fields = hl.import_bgen(resource('example.8bits.bgen'),
                                       entry_fields=['dosage'],
                                       contig_recoding={'01': '1'},
                                       reference_genome='GRCh37',
                                       _row_fields=[])
        self.assertEqual(no_row_fields.row.dtype,
                         hl.tstruct(locus=hl.tlocus('GRCh37'),
                                    alleles=hl.tarray(hl.tstr)))
        varid_only = hl.import_bgen(resource('example.8bits.bgen'),
                                    entry_fields=['dosage'],
                                    contig_recoding={'01': '1'},
                                    reference_genome='GRCh37',
                                    _row_fields=['varid'])
        self.assertEqual(varid_only.row.dtype,
                         hl.tstruct(locus=hl.tlocus('GRCh37'),
                                    alleles=hl.tarray(hl.tstr),
                                    varid=hl.tstr))
        rsid_only = hl.import_bgen(resource('example.8bits.bgen'),
                                   entry_fields=['dosage'],
                                   contig_recoding={'01': '1'},
                                   reference_genome='GRCh37',
                                   _row_fields=['rsid'])
        self.assertEqual(rsid_only.row.dtype,
                         hl.tstruct(locus=hl.tlocus('GRCh37'),
                                    alleles=hl.tarray(hl.tstr),
                                    rsid=hl.tstr))

        self.assertTrue(default_row_fields.drop('varid')._same(rsid_only))
        self.assertTrue(default_row_fields.drop('rsid')._same(varid_only))
        self.assertTrue(
            default_row_fields.drop('varid', 'rsid')._same(no_row_fields))

    def test_import_bgen_row_fields(self):
        mt = hl.import_bgen(resource('example.8bits.bgen'),
                            entry_fields=['dosage'],
                            contig_recoding={'01': '1'},
                            reference_genome='GRCh37',
                            _row_fields=['rsid', 'file_row_idx'])
        self.assertEqual(mt.file_row_idx.take(10), [99, 0, 100, 1, 101, 2, 102, 3, 103, 4])

        # the rsids are numbered 2 to 200 and corresond to the order of the
        # variants in the file (the loci are out of order in this file)
        #
        # the rsids look like: "RSID_99"
        rsids = mt.rsid.collect()
        self.assertEqual(mt.file_row_idx.collect(),
                         [int(rsid[5:]) - 2 for rsid in rsids])

    def test_import_bgen_variant_filtering(self):
        desired_variant_indexes = [1,2,3,5,7,9,11,13,17,198]
        actual = hl.import_bgen(resource('example.8bits.bgen'),
                                ['GT'],
                                contig_recoding={'01': '1'},
                                reference_genome=None,
                                min_partitions=10,
                                _row_fields=['file_row_idx'],
                                _variants_per_file={ resource('example.8bits.bgen') : desired_variant_indexes})
        # doing the expected import_bgen second catches the case where the
        # hadoop configuraiton is polluted with old data from the
        # _variants_per_file
        everything = hl.import_bgen(resource('example.8bits.bgen'),
                                    ['GT'],
                                    contig_recoding={'01': '1'},
                                    reference_genome=None,
                                    _row_fields=['file_row_idx'])
        self.assertEqual(everything.count(), (199, 500))

        expected = everything.filter_rows(
            hl.set(desired_variant_indexes).contains(hl.int32(everything.file_row_idx)))

        self.assertTrue(expected._same(actual))
        self.assertEqual((hl.str(actual.locus.contig) + ":" + hl.str(actual.locus.position)).collect(),
                         ['1:3000', '1:4000', '1:5000', '1:7000', '1:9000',
                          '1:11000', '1:13000', '1:15000', '1:19000', '1:100001'])


class GENTests(unittest.TestCase):
    def test_import_gen(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            contig_recoding={"01": "1"},
                            reference_genome = 'GRCh37').rows()
        self.assertTrue(gen.all(gen.locus.contig == "1"))
        self.assertEqual(gen.count(), 199)
        self.assertEqual(gen.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_gen_no_reference_specified(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            reference_genome=None)

        self.assertTrue(gen.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))
        self.assertEqual(gen.count_rows(), 199)

    def test_import_gen_skip_invalid_loci(self):
        mt = hl.import_gen(resource('skip_invalid_loci.gen'),
                           resource('skip_invalid_loci.sample'),
                           reference_genome='GRCh37',
                           skip_invalid_loci=True)
        self.assertTrue(mt._force_count_rows() == 3)

        with self.assertRaisesRegex(FatalError, 'Invalid locus'):
            hl.import_gen(resource('skip_invalid_loci.gen'),
                          resource('skip_invalid_loci.sample'))

    def test_export_gen(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            contig_recoding={"01": "1"},
                            reference_genome='GRCh37',
                            min_partitions=3)

        file = '/tmp/test_export_gen'
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
        self.assertTrue(in1.aggregate_entries(hl.agg.fraction(in1.GP == [0.0, 1.0, 0.0])) == 1.0)
        self.assertTrue(in1.aggregate_rows(hl.agg.fraction((in1.varid == hl.str(in1.row_idx)) &
                                                           (in1.rsid == hl.str(in1.row_idx)))) == 1.0)
        self.assertTrue(in1.aggregate_cols(hl.agg.fraction((in1.s == hl.str(in1.col_idx)))))


class LocusIntervalTests(unittest.TestCase):
    def test_import_locus_intervals(self):
        interval_file = resource('annotinterall.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome='GRCh37')
        nint = t.count()

        i = 0
        with open(interval_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nint, i)
        self.assertEqual(t.interval.dtype.point_type, hl.tlocus('GRCh37'))

        tmp_file = new_temp_file(prefix="test", suffix="interval_list")
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
        self.assertTrue(t.count() == 2)
        self.assertEqual(t.interval.dtype.point_type, hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_locus_intervals_badly_defined_intervals(self):
        interval_file = resource('example3.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome='GRCh37', skip_invalid_intervals=True)
        self.assertTrue(t.count() == 21)

        t = hl.import_locus_intervals(interval_file, reference_genome=None, skip_invalid_intervals=True)
        self.assertTrue(t.count() == 22)

    def test_import_bed(self):
        bed_file = resource('example1.bed')
        bed = hl.import_bed(bed_file, reference_genome='GRCh37')

        nbed = bed.count()
        i = 0
        with open(bed_file) as f:
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
        self.assertTrue(list(t.key.dtype) == ['interval'])
        self.assertTrue(list(t.row.dtype) == ['interval','target'])

    def test_import_bed_no_reference_specified(self):
        bed_file = resource('example1.bed')
        t = hl.import_bed(bed_file, reference_genome=None)
        self.assertTrue(t.count() == 3)
        self.assertEqual(t.interval.dtype.point_type, hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_bed_badly_defined_intervals(self):
        bed_file = resource('example4.bed')
        t = hl.import_bed(bed_file, reference_genome='GRCh37', skip_invalid_intervals=True)
        self.assertTrue(t.count() == 3)

        t = hl.import_bed(bed_file, reference_genome=None, skip_invalid_intervals=True)
        self.assertTrue(t.count() == 4)
        

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
                               row_fields=row_fields, row_key=[]).count()
        hl.import_matrix_table(doctest_resource('matrix3.tsv'),
                               row_fields=row_fields,
                               no_header=True).count()
        hl.import_matrix_table(doctest_resource('matrix3.tsv'),
                               row_fields=row_fields,
                               no_header=True,
                               row_key=[]).count()
        self.assertRaises(hl.utils.FatalError,
                          hl.import_matrix_table,
                          doctest_resource('matrix3.tsv'),
                          row_fields=row_fields,
                          no_header=True,
                          row_key=['foo'])


class ImportTableTests(unittest.TestCase):
    def test_import_table_force_bgz(self):
        f = new_temp_file(suffix=".bgz")
        t = hl.utils.range_table(10, 5)
        t.export(f)

        f2 = new_temp_file(suffix=".gz")
        run_command(["cp", uri_path(f), uri_path(f2)])
        t2 = hl.import_table(f2, force_bgz=True, impute=True).key_by('idx')
        self.assertTrue(t._same(t2))
