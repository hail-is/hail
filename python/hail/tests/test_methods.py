import unittest

import hail as hl
import hail.expr.aggregators as agg
from subprocess import DEVNULL, call as syscall
import numpy as np
from struct import unpack
import hail.utils as utils
from hail.linalg import BlockMatrix
from math import sqrt
from .utils import resource, doctest_resource, startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

class Tests(unittest.TestCase):
    _dataset = None

    def get_dataset(self):
        if Tests._dataset is None:
            Tests._dataset = hl.split_multi_hts(hl.import_vcf(resource('sample.vcf')))
        return Tests._dataset

    def test_ibd(self):
        dataset = self.get_dataset()

        def plinkify(ds, min=None, max=None):
            vcf = utils.new_temp_file(prefix="plink", suffix="vcf")
            plinkpath = utils.new_temp_file(prefix="plink")
            hl.export_vcf(ds, vcf)
            threshold_string = "{} {}".format("--min {}".format(min) if min else "",
                                              "--max {}".format(max) if max else "")

            plink_command = "plink --double-id --allow-extra-chr --vcf {} --genome full --out {} {}" \
                .format(utils.uri_path(vcf),
                        utils.uri_path(plinkpath),
                        threshold_string)
            result_file = utils.uri_path(plinkpath + ".genome")

            syscall(plink_command, shell=True, stdout=DEVNULL, stderr=DEVNULL)

            ### format of .genome file is:
            # _, fid1, iid1, fid2, iid2, rt, ez, z0, z1, z2, pihat, phe,
            # dst, ppc, ratio, ibs0, ibs1, ibs2, homhom, hethet (+ separated)

            ### format of ibd is:
            # i (iid1), j (iid2), ibd: {Z0, Z1, Z2, PI_HAT}, ibs0, ibs1, ibs2
            results = {}
            with open(result_file) as f:
                f.readline()
                for line in f:
                    row = line.strip().split()
                    results[(row[1], row[3])] = (list(map(float, row[6:10])),
                                                 list(map(int, row[14:17])))
            return results

        def compare(ds, min=None, max=None):
            plink_results = plinkify(ds, min, max)
            hail_results = hl.identity_by_descent(ds, min=min, max=max).collect()

            for row in hail_results:
                key = (row.i, row.j)
                self.assertAlmostEqual(plink_results[key][0][0], row.ibd.Z0, places=4)
                self.assertAlmostEqual(plink_results[key][0][1], row.ibd.Z1, places=4)
                self.assertAlmostEqual(plink_results[key][0][2], row.ibd.Z2, places=4)
                self.assertAlmostEqual(plink_results[key][0][3], row.ibd.PI_HAT, places=4)
                self.assertEqual(plink_results[key][1][0], row.ibs0)
                self.assertEqual(plink_results[key][1][1], row.ibs1)
                self.assertEqual(plink_results[key][1][2], row.ibs2)

        compare(dataset)
        compare(dataset, min=0.0, max=1.0)
        dataset = dataset.annotate_rows(dummy_maf=0.01)
        hl.identity_by_descent(dataset, dataset['dummy_maf'], min=0.0, max=1.0)
        hl.identity_by_descent(dataset, hl.float32(dataset['dummy_maf']), min=0.0, max=1.0)

    def test_impute_sex_same_as_plink(self):
        import subprocess as sp

        ds = hl.import_vcf(resource('x-chromosome.vcf'))

        sex = hl.impute_sex(ds.GT, include_par=True)

        vcf_file = utils.uri_path(utils.new_temp_file(prefix="plink", suffix="vcf"))
        out_file = utils.uri_path(utils.new_temp_file(prefix="plink"))

        hl.export_vcf(ds, vcf_file)

        try:
            out = sp.check_output(
                ["plink", "--vcf", vcf_file, "--const-fid", "--check-sex",
                 "--silent", "--out", out_file],
                stderr=sp.STDOUT)
        except sp.CalledProcessError as e:
            print(e.output)
            raise e

        plink_sex = hl.import_table(out_file + '.sexcheck',
                                    delimiter=' +',
                                    types={'SNPSEX': hl.tint32,
                                           'F': hl.tfloat64})
        plink_sex = plink_sex.select('IID', 'SNPSEX', 'F')
        plink_sex = plink_sex.select(
            s=plink_sex.IID,
            is_female=hl.cond(plink_sex.SNPSEX == 2,
                              True,
                              hl.cond(plink_sex.SNPSEX == 1,
                                      False,
                                      hl.null(hl.tbool))),
            f_stat=plink_sex.F).key_by('s')

        sex = sex.select(s=sex.s,
                         is_female=sex.is_female,
                         f_stat=sex.f_stat)

        self.assertTrue(plink_sex._same(sex.select_globals(), tolerance=1e-3))

        ds = ds.annotate_rows(aaf=(agg.call_stats(ds.GT, ds.alleles)).AF[1])

        self.assertTrue(hl.impute_sex(ds.GT)._same(hl.impute_sex(ds.GT, aaf='aaf')))

    def test_linreg(self):
        dataset = hl.import_vcf(resource('regressionLinear.vcf'))

        phenos = hl.import_table(resource('regressionLinear.pheno'),
                                 types={'Pheno': hl.tfloat64},
                                 key='Sample')
        covs = hl.import_table(resource('regressionLinear.cov'),
                               types={'Cov1': hl.tfloat64, 'Cov2': hl.tfloat64},
                               key='Sample')

        dataset = dataset.annotate_cols(pheno=phenos[dataset.s].Pheno, cov=covs[dataset.s])
        dataset = hl.linear_regression(dataset,
                                       ys=dataset.pheno,
                                       x=dataset.GT.n_alt_alleles(),
                                       covariates=[dataset.cov.Cov1, dataset.cov.Cov2 + 1 - 1])

        dataset.count_rows()

    def test_trio_matrix(self):
        """
        This test depends on certain properties of the trio matrix VCF and
        pedigree structure. This test is NOT a valid test if the pedigree
        includes quads: the trio_matrix method will duplicate the parents
        appropriately, but the genotypes_table and samples_table orthogonal
        paths would require another duplication/explode that we haven't written.
        """
        ped = hl.Pedigree.read(resource('triomatrix.fam'))
        ht = hl.import_fam(resource('triomatrix.fam'))

        mt = hl.import_vcf(resource('triomatrix.vcf'))
        mt = mt.annotate_cols(fam=ht[mt.s])

        dads = ht.filter(hl.is_defined(ht.pat_id))
        dads = dads.select(dads.pat_id, is_dad=True).key_by('pat_id')

        moms = ht.filter(hl.is_defined(ht.mat_id))
        moms = moms.select(moms.mat_id, is_mom=True).key_by('mat_id')

        et = (mt.entries()
            .key_by('s')
            .join(dads, how='left')
            .join(moms, how='left'))
        et = et.annotate(is_dad=hl.is_defined(et.is_dad),
                         is_mom=hl.is_defined(et.is_mom))

        et = (et
            .group_by(et.locus, et.alleles, fam=et.fam.fam_id)
            .aggregate(data=hl.agg.collect(hl.struct(
            role=hl.case().when(et.is_dad, 1).when(et.is_mom, 2).default(0),
            g=hl.struct(GT=et.GT, AD=et.AD, DP=et.DP, GQ=et.GQ, PL=et.PL)))))

        et = et.filter(hl.len(et.data) == 3)
        et = et.select('locus', 'alleles', 'fam', 'data').explode('data')

        tt = hl.trio_matrix(mt, ped, complete_trios=True).entries()
        tt = tt.annotate(fam=tt.proband.fam.fam_id,
                         data=[hl.struct(role=0, g=tt.proband_entry.select('GT', 'AD', 'DP', 'GQ', 'PL')),
                               hl.struct(role=1, g=tt.father_entry.select('GT', 'AD', 'DP', 'GQ', 'PL')),
                               hl.struct(role=2, g=tt.mother_entry.select('GT', 'AD', 'DP', 'GQ', 'PL'))])
        tt = tt.select('locus', 'alleles', 'fam', 'data').explode('data')
        tt = tt.filter(hl.is_defined(tt.data.g)).key_by('locus', 'alleles', 'fam')

        self.assertTrue(et._same(tt))

        # test annotations
        e_cols = (mt.cols()
            .join(dads, how='left')
            .join(moms, how='left'))
        e_cols = e_cols.annotate(is_dad=hl.is_defined(e_cols.is_dad),
                                 is_mom=hl.is_defined(e_cols.is_mom))
        e_cols = (e_cols.group_by(fam=e_cols.fam.fam_id)
            .aggregate(data=hl.agg.collect(hl.struct(role=hl.case()
                                                     .when(e_cols.is_dad, 1).when(e_cols.is_mom, 2).default(0),
                                                     sa=e_cols.row.select(*mt.col)))))
        e_cols = e_cols.filter(hl.len(e_cols.data) == 3).select('fam', 'data').explode('data')

        t_cols = hl.trio_matrix(mt, ped, complete_trios=True).cols()
        t_cols = t_cols.annotate(fam=t_cols.proband.fam.fam_id,
                                 data=[
                                     hl.struct(role=0, sa=t_cols.proband),
                                     hl.struct(role=1, sa=t_cols.father),
                                     hl.struct(role=2, sa=t_cols.mother)]).select('fam', 'data').explode('data')
        t_cols = t_cols.filter(hl.is_defined(t_cols.data.sa)).key_by('fam')

        self.assertTrue(e_cols._same(t_cols))

    def test_sample_qc(self):
        dataset = self.get_dataset()
        dataset = hl.sample_qc(dataset)

    def test_grm(self):
        tolerance = 0.001

        def load_id_file(path):
            ids = []
            with hl.hadoop_read(path) as f:
                for l in f:
                    r = l.strip().split('\t')
                    self.assertEqual(len(r), 2)
                    ids.append(r[1])
            return ids

        def load_rel(ns, path):
            rel = np.zeros((ns, ns))
            with hl.hadoop_read(path) as f:
                for i, l in enumerate(f):
                    for j, n in enumerate(map(float, l.strip().split('\t'))):
                        rel[i, j] = n
                    self.assertEqual(j, i)
                self.assertEqual(i, ns - 1)
            return rel

        def load_grm(ns, nv, path):
            m = np.zeros((ns, ns))
            with utils.hadoop_read(path) as f:
                i = 0
                for l in f:
                    row = l.strip().split('\t')
                    self.assertEqual(int(row[2]), nv)
                    m[int(row[0]) - 1, int(row[1]) - 1] = float(row[3])
                    i += 1

                self.assertEqual(i, ns * (ns + 1) / 2)
            return m

        def load_bin(ns, path):
            m = np.zeros((ns, ns))
            with utils.hadoop_read_binary(path) as f:
                for i in range(ns):
                    for j in range(i + 1):
                        b = f.read(4)
                        self.assertEqual(len(b), 4)
                        m[i, j] = unpack('<f', bytearray(b))[0]
                left = f.read()
                self.assertEqual(len(left), 0)
            return m

        b_file = utils.new_temp_file(prefix="plink")
        rel_file = utils.new_temp_file(prefix="test", suffix="rel")
        rel_id_file = utils.new_temp_file(prefix="test", suffix="rel.id")
        grm_file = utils.new_temp_file(prefix="test", suffix="grm")
        grm_bin_file = utils.new_temp_file(prefix="test", suffix="grm.bin")
        grm_nbin_file = utils.new_temp_file(prefix="test", suffix="grm.N.bin")

        dataset = self.get_dataset()
        n_samples = dataset.count_cols()
        dataset = dataset.annotate_rows(AC=agg.sum(dataset.GT.n_alt_alleles()),
                                        n_called=agg.count_where(hl.is_defined(dataset.GT)))
        dataset = dataset.filter_rows((dataset.AC > 0) & (dataset.AC < 2 * dataset.n_called))
        dataset = dataset.filter_rows(dataset.n_called == n_samples).persist()

        hl.export_plink(dataset, b_file, id=dataset.s)

        sample_ids = [row.s for row in dataset.cols().select('s').collect()]
        n_variants = dataset.count_rows()
        self.assertGreater(n_variants, 0)

        grm = hl.genetic_relatedness_matrix(dataset)
        grm.export_id_file(rel_id_file)

        ############
        ### rel

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-rel --out {}'''
                .format(utils.uri_path(b_file), utils.uri_path(p_file)), shell=True, stdout=DEVNULL, stderr=DEVNULL)
        self.assertEqual(load_id_file(p_file + ".rel.id"), sample_ids)

        grm.export_rel(rel_file)
        self.assertEqual(load_id_file(rel_id_file), sample_ids)
        self.assertTrue(np.allclose(load_rel(n_samples, p_file + ".rel"),
                                    load_rel(n_samples, rel_file),
                                    atol=tolerance))

        ############
        ### gcta-grm

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-grm-gz --out {}'''
                .format(utils.uri_path(b_file), utils.uri_path(p_file)), shell=True, stdout=DEVNULL, stderr=DEVNULL)
        self.assertEqual(load_id_file(p_file + ".grm.id"), sample_ids)

        grm.export_gcta_grm(grm_file)
        self.assertTrue(np.allclose(load_grm(n_samples, n_variants, p_file + ".grm.gz"),
                                    load_grm(n_samples, n_variants, grm_file),
                                    atol=tolerance))

        ############
        ### gcta-grm-bin

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-grm-bin --out {}'''
                .format(utils.uri_path(b_file), utils.uri_path(p_file)), shell=True, stdout=DEVNULL, stderr=DEVNULL)

        self.assertEqual(load_id_file(p_file + ".grm.id"), sample_ids)

        grm.export_gcta_grm_bin(grm_bin_file, grm_nbin_file)

        self.assertTrue(np.allclose(load_bin(n_samples, p_file + ".grm.bin"),
                                    load_bin(n_samples, grm_bin_file),
                                    atol=tolerance))
        self.assertTrue(np.allclose(load_bin(n_samples, p_file + ".grm.N.bin"),
                                    load_bin(n_samples, grm_nbin_file),
                                    atol=tolerance))

    def test_block_matrix_from_numpy(self):
        ndarray = np.matrix([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], dtype=np.float64)

        for block_size in [1, 2, 5, 1024]:
            block_matrix = BlockMatrix.from_numpy(ndarray, block_size)
            assert (block_matrix.n_rows == 3)
            assert (block_matrix.n_cols == 5)
            assert (block_matrix.to_numpy() == ndarray).all()

    def test_rrm(self):
        seed = 0
        n1 = 100
        m1 = 200
        k = 3
        fst = .9

        dataset = hl.balding_nichols_model(k,
                                           n1,
                                           m1,
                                           fst=(k * [fst]),
                                           seed=seed,
                                           n_partitions=4)
        dataset = dataset.annotate_cols(s = hl.str(dataset.sample_idx)).key_cols_by('s')

        def direct_calculation(ds):
            ds = BlockMatrix.from_entry_expr(ds['GT'].n_alt_alleles()).to_numpy()

            # filter out constant rows
            isconst = lambda r: any([all([(gt < c + .01) and (gt > c - .01) for gt in r]) for c in range(3)])
            ds = np.array([row for row in ds if not isconst(row)])

            nvariants, nsamples = ds.shape
            sumgt = lambda r: sum([i for i in r if i >= 0])
            sumsq = lambda r: sum([i ** 2 for i in r if i >= 0])

            mean = [sumgt(row) / nsamples for row in ds]
            stddev = [sqrt(sumsq(row) / nsamples - mean[i] ** 2)
                      for i, row in enumerate(ds)]

            mat = np.array([[(g - mean[i]) / stddev[i] for g in row] for i, row in enumerate(ds)])

            rrm = (mat.T @ mat) / nvariants

            return rrm

        def hail_calculation(ds):
            rrm = hl.realized_relationship_matrix(ds['GT'])
            fn = utils.new_temp_file(suffix='.tsv')

            rrm.export_tsv(fn)
            data = []
            with open(utils.uri_path(fn)) as f:
                f.readline()
                for line in f:
                    row = line.strip().split()
                    data.append(list(map(float, row)))

            return np.array(data)

        manual = direct_calculation(dataset)
        rrm = hail_calculation(dataset)

        self.assertTrue(np.allclose(manual, rrm))

    def test_pca(self):
        dataset = hl.balding_nichols_model(3, 100, 100)
        eigenvalues, scores, loadings = hl.pca(dataset.GT.n_alt_alleles(), k=2, compute_loadings=True)

        self.assertEqual(len(eigenvalues), 2)
        self.assertTrue(isinstance(scores, hl.Table))
        self.assertEqual(scores.count(), 100)
        self.assertTrue(isinstance(loadings, hl.Table))
        self.assertEqual(loadings.count(), 100)

        _, _, loadings = hl.pca(dataset.GT.n_alt_alleles(), k=2, compute_loadings=False)
        self.assertEqual(loadings, None)

    def test_pcrelate(self):
        dataset = hl.balding_nichols_model(3, 100, 100)
        dataset = dataset.annotate_cols(sample_idx = hl.str(dataset.sample_idx))
        t = hl.pc_relate(dataset, 2, 0.05, block_size=64, statistics="phi")

        self.assertTrue(isinstance(t, hl.Table))
        t.count()

    def test_rename_duplicates(self):
        dataset = self.get_dataset()  # FIXME - want to rename samples with same id
        renamed_ids = hl.rename_duplicates(dataset).cols().select('s').collect()
        self.assertTrue(len(set(renamed_ids)), len(renamed_ids))

    def test_split_multi_hts(self):
        ds1 = hl.import_vcf(resource('split_test.vcf'))
        ds1 = hl.split_multi_hts(ds1)
        ds2 = hl.import_vcf(resource('split_test_b.vcf'))
        df = ds1.rows()
        self.assertTrue(df.all((df.locus.position == 1180) | df.was_split))
        ds1 = ds1.drop('was_split', 'a_index')
        self.assertTrue(ds1._same(ds2))

        ds = self.get_dataset()
        ds = ds.annotate_entries(X=ds.GT)
        with self.assertRaises(utils.FatalError):
            hl.split_multi_hts(ds)

    def test_mendel_errors(self):
        dataset = self.get_dataset()
        men, fam, ind, var = hl.mendel_errors(dataset, hl.Pedigree.read(resource('sample.fam')))
        men.select('fam_id', 's', 'code')
        fam.select('pat_id', 'children')
        self.assertEqual(list(ind.key), ['s'])
        self.assertEqual(list(var.key), ['locus', 'alleles'])
        dataset.annotate_rows(mendel=var[dataset.locus, dataset.alleles]).count_rows()

    def test_export_vcf(self):
        dataset = hl.import_vcf(resource('sample.vcf.bgz'))
        vcf_metadata = hl.get_vcf_metadata(resource('sample.vcf.bgz'))
        hl.export_vcf(dataset, '/tmp/sample.vcf', metadata=vcf_metadata)
        dataset_imported = hl.import_vcf('/tmp/sample.vcf')
        self.assertTrue(dataset._same(dataset_imported))

        metadata_imported = hl.get_vcf_metadata('/tmp/sample.vcf')
        self.assertDictEqual(vcf_metadata, metadata_imported)

    def test_concordance(self):
        dataset = self.get_dataset()
        glob_conc, cols_conc, rows_conc = hl.concordance(dataset, dataset)

        self.assertEqual(sum([sum(glob_conc[i]) for i in range(5)]), dataset.count_rows() * dataset.count_cols())

        counts = dataset.aggregate_entries(hl.Struct(n_het=agg.count(agg.filter(dataset.GT.is_het(), dataset.GT)),
                                                     n_hom_ref=agg.count(agg.filter(dataset.GT.is_hom_ref(), dataset.GT)),
                                                     n_hom_var=agg.count(agg.filter(dataset.GT.is_hom_var(), dataset.GT)),
                                                     nNoCall=agg.count(
                                                         agg.filter(hl.is_missing(dataset.GT), dataset.GT))))

        self.assertEqual(glob_conc[0][0], 0)
        self.assertEqual(glob_conc[1][1], counts.nNoCall)
        self.assertEqual(glob_conc[2][2], counts.n_hom_ref)
        self.assertEqual(glob_conc[3][3], counts.n_het)
        self.assertEqual(glob_conc[4][4], counts.n_hom_var)
        [self.assertEqual(glob_conc[i][j], 0) for i in range(5) for j in range(5) if i != j]

        self.assertTrue(cols_conc.all(hl.sum(hl.flatten(cols_conc.concordance)) == dataset.count_rows()))
        self.assertTrue(rows_conc.all(hl.sum(hl.flatten(rows_conc.concordance)) == dataset.count_cols()))

        cols_conc.write('/tmp/foo.kt', overwrite=True)
        rows_conc.write('/tmp/foo.kt', overwrite=True)

    def test_import_locus_intervals(self):
        interval_file = resource('annotinterall.interval_list')
        intervals = hl.import_locus_intervals(interval_file, reference_genome='GRCh37')
        nint = intervals.count()

        i = 0
        with open(interval_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nint, i)

        self.assertEqual(intervals.interval.dtype.point_type, hl.tlocus('GRCh37'))

    def test_import_locus_intervals_no_reference_specified(self):
        interval_file = resource('annotinterall.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome=None)
        self.assertEqual(t.interval.dtype.point_type, hl.tstruct(contig=hl.tstr, position=hl.tint32))

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

    def test_import_bed_no_reference_specified(self):
        bed_file = resource('example1.bed')
        t = hl.import_bed(bed_file, reference_genome=None)
        self.assertEqual(t.interval.dtype.point_type, hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_fam(self):
        fam_file = resource('sample.fam')
        nfam = hl.import_fam(fam_file).count()
        i = 0
        with open(fam_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nfam, i)

    def test_export_plink(self):
        ds = self.get_dataset()

        hl.export_plink(ds, '/tmp/plink_example', id=ds.s)

        hl.export_plink(ds, '/tmp/plink_example2', id=ds.s, fam_id=ds.s, pat_id="nope",
                        mat_id="nada", is_female=True, is_case=False)

        hl.export_plink(ds, '/tmp/plink_example3', id=ds.s, fam_id=ds.s, pat_id="nope",
                        mat_id="nada", is_female=True, quant_pheno=hl.float64(hl.len(ds.s)))

        self.assertRaises(ValueError,
                          lambda: hl.export_plink(ds, '/tmp/plink_example', is_case=True, quant_pheno=0.0))

        self.assertRaises(ValueError, lambda: hl.export_plink(ds, '/tmp/plink_example', foo=0.0))

        self.assertRaises(TypeError, lambda: hl.export_plink(ds, '/tmp/plink_example', is_case=0.0))

        # FIXME still resolving: these should throw an error due to unexpected row / entry indicies, still looking into why a more cryptic error is being thrown
        # self.assertRaises(ExpressionException, lambda: hl.export_plink(ds, '/tmp/plink_example', id = ds.locus.contig))
        # self.assertRaises(ExpressionException, lambda: hl.export_plink(ds, '/tmp/plink_example', id = ds.GT))

    def test_tdt(self):
        pedigree = hl.Pedigree.read(resource('tdt.fam'))
        tdt_tab = (hl.transmission_disequilibrium_test(
            hl.split_multi_hts(hl.import_vcf(resource('tdt.vcf'), min_partitions=4)),
            pedigree))

        truth = hl.import_table(
            resource('tdt_results.tsv'),
            types={'POSITION': hl.tint32, 'T': hl.tint32, 'U': hl.tint32,
                   'Chi2': hl.tfloat64, 'Pval': hl.tfloat64})
        truth = (truth
            .transmute(locus=hl.locus(truth.CHROM, truth.POSITION),
                       alleles=[truth.REF, truth.ALT])
            .key_by('locus', 'alleles'))

        if tdt_tab.count() != truth.count():
            self.fail('Result has {} rows but should have {} rows'.format(tdt_tab.count(), truth.count()))

        bad = (tdt_tab.filter(hl.is_nan(tdt_tab.p_value), keep=False)
            .join(truth.filter(hl.is_nan(truth.Pval), keep=False), how='outer'))

        bad = bad.filter(~(
                (bad.t == bad.T) &
                (bad.u == bad.U) &
                (hl.abs(bad.chi2 - bad.Chi2) < 0.001) &
                (hl.abs(bad.p_value - bad.Pval) < 0.001)))

        if bad.count() != 0:
            bad.order_by(hl.asc(bad.v)).show()
            self.fail('Found rows in violation of the predicate (see show output)')

    def test_maximal_independent_set(self):
        # prefer to remove nodes with higher index
        t = hl.utils.range_table(10)
        graph = t.select(i=hl.int64(t.idx), j=hl.int64(t.idx + 10), bad_type=hl.float32(t.idx))

        mis_table = hl.maximal_independent_set(graph.i, graph.j, True, lambda l, r: l - r)
        mis = [row['node'] for row in mis_table.collect()]
        self.assertEqual(sorted(mis), list(range(0, 10)))
        self.assertEqual(mis_table.row.dtype, hl.tstruct(node=hl.tint64))

        self.assertRaises(ValueError, lambda: hl.maximal_independent_set(graph.i, graph.bad_type, True))
        self.assertRaises(ValueError, lambda: hl.maximal_independent_set(graph.i, hl.utils.range_table(10).idx, True))
        self.assertRaises(ValueError, lambda: hl.maximal_independent_set(hl.literal(1), hl.literal(2), True))

    def test_filter_alleles(self):
        # poor man's Gen
        paths = [resource('sample.vcf'),
                 resource('multipleChromosomes.vcf'),
                 resource('sample2.vcf')]
        for path in paths:
            ds = hl.import_vcf(path)
            self.assertEqual(
                hl.FilterAlleles(hl.range(0, ds.alleles.length() - 1).map(lambda i: False))
                    .filter()
                    .count_rows(), 0)
            self.assertEqual(
                hl.FilterAlleles(hl.range(0, ds.alleles.length() - 1).map(lambda i: True))
                    .filter()
                    .count_rows(), ds.count_rows())

    def test_filter_alleles2(self):
        # 1 variant: A:T,G
        ds = hl.import_vcf(resource('filter_alleles/input.vcf'))

        fa = hl.FilterAlleles(ds.alleles[1:].map(lambda aa: aa == "T"))
        fa.subset_entries_hts()
        self.assertTrue(
            hl.import_vcf(resource('filter_alleles/keep_allele1_subset.vcf'))._same(fa.filter()))

        fa = hl.FilterAlleles(ds.alleles[1:].map(lambda aa: aa == "G"))
        # test fa.annotate_rows
        fa.annotate_rows(new_to_old=fa.new_to_old)
        fa.subset_entries_hts()
        self.assertTrue(
            (hl.import_vcf(resource('filter_alleles/keep_allele2_subset.vcf'))
                .annotate_rows(new_to_old=[0, 2])
                ._same(fa.filter())))

        # also test keep=False
        fa = hl.FilterAlleles(ds.alleles[1:].map(lambda aa: aa == "G"), keep=False)
        fa.downcode_entries_hts()
        self.assertTrue(
            hl.import_vcf(resource('filter_alleles/keep_allele1_downcode.vcf'))._same(fa.filter()))

        fa = hl.FilterAlleles(ds.alleles[1:].map(lambda aa: aa == "G"))
        fa.downcode_entries_hts()
        self.assertTrue(
            hl.import_vcf(resource('filter_alleles/keep_allele2_downcode.vcf'))._same(fa.filter()))

    def test_ld_prune(self):
        ds = hl.split_multi_hts(
            hl.import_vcf(resource('sample.vcf')))
        hl.ld_prune(ds, 8).count_rows()

    def test_entries_table(self):
        n_rows, n_cols = 5, 3
        rows = [{'i': i, 'j': j, 'entry': float(i + j)} for i in range(n_rows) for j in range(n_cols)]
        schema = hl.tstruct(i=hl.tint32, j=hl.tint32, entry=hl.tfloat64)
        table = hl.Table.parallelize([hl.struct(i=row['i'], j=row['j'], entry=row['entry']) for row in rows], schema)
        table = table.annotate(i=hl.int64(table.i),
                               j=hl.int64(table.j))

        ndarray = np.reshape(list(map(lambda row: row['entry'], rows)), (n_rows, n_cols))

        for block_size in [1, 2, 1024]:
            block_matrix = BlockMatrix.from_numpy(ndarray, block_size)
            entries_table = block_matrix.entries()
            self.assertEqual(entries_table.count(), n_cols * n_rows)
            self.assertEqual(len(entries_table.row), 3)
            self.assertTrue(table._same(entries_table))

    def test_min_rep(self):
        # FIXME actually test
        ds = self.get_dataset()
        hl.min_rep(ds).count()

    def test_filter_intervals(self):
        ds = hl.import_vcf(resource('sample.vcf'), min_partitions=20)

        self.assertEqual(
            hl.filter_intervals(ds, [hl.parse_locus_interval('20:10639222-10644705')]).count_rows(), 3)

        intervals = [hl.parse_locus_interval('20:10639222-10644700'),
                     hl.parse_locus_interval('20:10644700-10644705')]
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 3)

        intervals = hl.array([hl.parse_locus_interval('20:10639222-10644700'),
                              hl.parse_locus_interval('20:10644700-10644705')])
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 3)

        intervals = hl.array([hl.parse_locus_interval('20:10639222-10644700').value,
                              hl.parse_locus_interval('20:10644700-10644705')])
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 3)

        intervals = [hl.parse_locus_interval('[20:10019093-10026348]').value,
                     hl.parse_locus_interval('[20:17705793-17716416]').value]
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 4)

    def test_filter_intervals_compound_partition_key(self):
        ds = hl.import_vcf(resource('sample.vcf'), min_partitions=20)
        ds = (ds.annotate_rows(variant=hl.struct(locus=ds.locus, alleles=ds.alleles))
              .key_rows_by('locus', 'alleles'))

        intervals = [hl.Interval(hl.Struct(locus=hl.Locus('20', 10639222), alleles=['A', 'T']),
                                 hl.Struct(locus=hl.Locus('20', 10644700), alleles=['A', 'T']))]
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 3)

    def test_balding_nichols_model(self):
        from hail.stats import TruncatedBetaDist

        ds = hl.balding_nichols_model(2, 20, 25, 3,
                                      pop_dist=[1.0, 2.0],
                                      fst=[.02, .06],
                                      af_dist=TruncatedBetaDist(a=0.01, b=2.0, min=0.05, max=0.95),
                                      seed=1)

        self.assertEqual(ds.count_cols(), 20)
        self.assertEqual(ds.count_rows(), 25)
        self.assertEqual(ds.n_partitions(), 3)

        glob = ds.globals
        self.assertEqual(glob.n_populations.value, 2)
        self.assertEqual(glob.n_samples.value, 20)
        self.assertEqual(glob.n_variants.value, 25)
        self.assertEqual(glob.pop_dist.value, [1, 2])
        self.assertEqual(glob.fst.value, [.02, .06])
        self.assertEqual(glob.seed.value, 1)
        self.assertEqual(glob.ancestral_af_dist.value,
                         hl.Struct(type='TruncatedBetaDist', a=0.01, b=2.0, min=0.05, max=0.95))

    def test_skat(self):
        ds2 = hl.import_vcf(resource('sample2.vcf'))

        covariatesSkat = (hl.import_table(resource("skat.cov"), impute=True)
            .key_by("Sample"))

        phenotypesSkat = (hl.import_table(resource("skat.pheno"),
                                          types={"Pheno": hl.tfloat64},
                                          missing="0")
            .key_by("Sample"))

        intervalsSkat = (hl.import_locus_intervals(resource("skat.interval_list")))

        weightsSkat = (hl.import_table(resource("skat.weights"),
                                       types={"locus": hl.tlocus(),
                                              "weight": hl.tfloat64})
            .key_by("locus"))

        ds = hl.split_multi_hts(ds2)
        ds = ds.annotate_rows(gene=intervalsSkat[ds.locus],
                              weight=weightsSkat[ds.locus].weight)
        ds = ds.annotate_cols(pheno=phenotypesSkat[ds.s].Pheno,
                              cov=covariatesSkat[ds.s])
        ds = ds.annotate_cols(pheno=hl.cond(ds.pheno == 1.0,
                                            False,
                                            hl.cond(ds.pheno == 2.0,
                                                    True,
                                                    hl.null(hl.tbool))))

        hl.skat(ds,
                key_expr=ds.gene,
                weight_expr=ds.weight,
                y=ds.pheno,
                x=ds.GT.n_alt_alleles(),
                covariates=[ds.cov.Cov1, ds.cov.Cov2],
                logistic=False).count()

        hl.skat(ds,
                key_expr=ds.gene,
                weight_expr=ds.weight,
                y=ds.pheno,
                x=hl.pl_dosage(ds.PL),
                covariates=[ds.cov.Cov1, ds.cov.Cov2],
                logistic=True).count()

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

    def test_import_bgen(self):
        hl.index_bgen(resource('example.v11.bgen'))

        bgen_rows = hl.import_bgen(resource('example.v11.bgen'),
                                   entry_fields=['GT', 'GP'],
                                   sample_file=resource('example.sample'),
                                   contig_recoding={'01': '1'},
                                   reference_genome='GRCh37').rows()
        self.assertTrue(bgen_rows.all(bgen_rows.locus.contig == '1'))
        self.assertEqual(bgen_rows.count(), 199)

        hl.index_bgen(resource('example.8bits.bgen'))

        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['dosage'],
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct(dosage=hl.tfloat64))

        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GT', 'GP'],
                              sample_file=resource('example.sample'),
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct(GT=hl.tcall, GP=hl.tarray(hl.tfloat64)))
        self.assertEqual(bgen.count_rows(), 199)

        hl.index_bgen(resource('example.10bits.bgen'))
        bgen = hl.import_bgen(resource('example.10bits.bgen'),
                              entry_fields=['GT', 'GP', 'dosage'],
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct(GT=hl.tcall, GP=hl.tarray(hl.tfloat64), dosage=hl.tfloat64))
        self.assertEqual(bgen.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_bgen_no_reference_specified(self):
        bgen = hl.import_bgen(resource('example.10bits.bgen'),
                              entry_fields=['GT', 'GP', 'dosage'],
                              contig_recoding={'01': '1'},
                              reference_genome=None)
        self.assertTrue(bgen.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))
        self.assertEqual(bgen.count_rows(), 199)

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

    def test_import_plink(self):
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
        self.assertTrue(plink.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))

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
