import unittest

import hail as hl
from ..helpers import *


class Tests(unittest.TestCase):
    @test_timeout(local=3 * 60, batch=6 * 60)
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
        mt = mt.annotate_cols(fam=ht[mt.s].fam_id)

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
            .group_by(et.locus, et.alleles, fam=et.fam)
            .aggregate(data=hl.agg.collect(hl.struct(
            role=hl.case().when(et.is_dad, 1).when(et.is_mom, 2).default(0),
            g=hl.struct(GT=et.GT, AD=et.AD, DP=et.DP, GQ=et.GQ, PL=et.PL)))))

        et = et.filter(hl.len(et.data) == 3)
        et = et.select('data').explode('data')

        tt = hl.trio_matrix(mt, ped, complete_trios=True).entries().key_by('locus', 'alleles')
        tt = tt.annotate(fam=tt.proband.fam,
                         data=[hl.struct(role=0, g=tt.proband_entry.select('GT', 'AD', 'DP', 'GQ', 'PL')),
                               hl.struct(role=1, g=tt.father_entry.select('GT', 'AD', 'DP', 'GQ', 'PL')),
                               hl.struct(role=2, g=tt.mother_entry.select('GT', 'AD', 'DP', 'GQ', 'PL'))])
        tt = tt.select('fam', 'data').explode('data')
        tt = tt.filter(hl.is_defined(tt.data.g)).key_by('locus', 'alleles', 'fam')

        self.assertEqual(et.key.dtype, tt.key.dtype)
        self.assertEqual(et.row.dtype, tt.row.dtype)
        self.assertTrue(et._same(tt))

        # test annotations
        e_cols = (mt.cols()
                  .join(dads, how='left')
                  .join(moms, how='left'))
        e_cols = e_cols.annotate(is_dad=hl.is_defined(e_cols.is_dad),
                                 is_mom=hl.is_defined(e_cols.is_mom))
        e_cols = (e_cols.group_by(fam=e_cols.fam)
                  .aggregate(data=hl.agg.collect(hl.struct(role=hl.case()
                                                           .when(e_cols.is_dad, 1).when(e_cols.is_mom, 2).default(0),
                                                           sa=hl.struct(**e_cols.row.select(*mt.col))))))
        e_cols = e_cols.filter(hl.len(e_cols.data) == 3).select('data').explode('data')

        t_cols = hl.trio_matrix(mt, ped, complete_trios=True).cols()
        t_cols = t_cols.annotate(fam=t_cols.proband.fam,
                                 data=[
                                     hl.struct(role=0, sa=t_cols.proband),
                                     hl.struct(role=1, sa=t_cols.father),
                                     hl.struct(role=2, sa=t_cols.mother)]).key_by('fam').select('data').explode('data')
        t_cols = t_cols.filter(hl.is_defined(t_cols.data.sa))

        self.assertEqual(e_cols.key.dtype, t_cols.key.dtype)
        self.assertEqual(e_cols.row.dtype, t_cols.row.dtype)
        self.assertTrue(e_cols._same(t_cols))

    def test_trio_matrix_null_keys(self):
        ped = hl.Pedigree.read(resource('triomatrix.fam'))
        ht = hl.import_fam(resource('triomatrix.fam'))

        mt = hl.import_vcf(resource('triomatrix.vcf'))
        mt = mt.annotate_cols(fam=ht[mt.s].fam_id)

        # Make keys all null
        mt = mt.key_cols_by(s=hl.missing(hl.tstr))

        tt = hl.trio_matrix(mt, ped, complete_trios=True)
        self.assertEqual(tt.count_cols(), 0)

    def test_trio_matrix_incomplete_trios(self):
        ped = hl.Pedigree.read(resource('triomatrix.fam'))
        mt = hl.import_vcf(resource('triomatrix.vcf'))
        hl.trio_matrix(mt, ped, complete_trios=False)

    @test_timeout(3 * 60, local=4 * 60, batch=4 * 60)
    def test_mendel_errors(self):
        mt = hl.import_vcf(resource('mendel.vcf'))
        ped = hl.Pedigree.read(resource('mendel.fam'))

        men, fam, ind, var = hl.mendel_errors(mt['GT'], ped)

        self.assertEqual(men.key.dtype, hl.tstruct(locus=mt.locus.dtype,
                                                   alleles=hl.tarray(hl.tstr),
                                                   s=hl.tstr))
        self.assertEqual(men.row.dtype, hl.tstruct(locus=mt.locus.dtype,
                                                   alleles=hl.tarray(hl.tstr),
                                                   s=hl.tstr,
                                                   fam_id=hl.tstr,
                                                   mendel_code=hl.tint))
        self.assertEqual(fam.key.dtype, hl.tstruct(pat_id=hl.tstr,
                                                   mat_id=hl.tstr))
        self.assertEqual(fam.row.dtype, hl.tstruct(pat_id=hl.tstr,
                                                   mat_id=hl.tstr,
                                                   fam_id=hl.tstr,
                                                   children=hl.tint,
                                                   errors=hl.tint64,
                                                   snp_errors=hl.tint64))
        self.assertEqual(ind.key.dtype, hl.tstruct(s=hl.tstr))
        self.assertEqual(ind.row.dtype, hl.tstruct(s=hl.tstr,
                                                   fam_id=hl.tstr,
                                                   errors=hl.tint64,
                                                   snp_errors=hl.tint64))
        self.assertEqual(var.key.dtype, hl.tstruct(locus=mt.locus.dtype,
                                                   alleles=hl.tarray(hl.tstr)))
        self.assertEqual(var.row.dtype, hl.tstruct(locus=mt.locus.dtype,
                                                   alleles=hl.tarray(hl.tstr),
                                                   errors=hl.tint64))

        self.assertEqual(men.count(), 41)
        self.assertEqual(fam.count(), 2)
        self.assertEqual(ind.count(), 7)
        self.assertEqual(var.count(), mt.count_rows())

        self.assertEqual(set(fam.select('children', 'errors', 'snp_errors').collect()),
                         {
                             hl.utils.Struct(pat_id='Dad1', mat_id='Mom1', children=2,
                                             errors=41, snp_errors=39),
                             hl.utils.Struct(pat_id='Dad2', mat_id='Mom2', children=1,
                                             errors=0, snp_errors=0)
                         })

        self.assertEqual(set(ind.select('errors', 'snp_errors').collect()),
                         {
                             hl.utils.Struct(s='Son1', errors=23, snp_errors=22),
                             hl.utils.Struct(s='Dtr1', errors=18, snp_errors=17),
                             hl.utils.Struct(s='Dad1', errors=19, snp_errors=18),
                             hl.utils.Struct(s='Mom1', errors=22, snp_errors=21),
                             hl.utils.Struct(s='Dad2', errors=0, snp_errors=0),
                             hl.utils.Struct(s='Mom2', errors=0, snp_errors=0),
                             hl.utils.Struct(s='Son2', errors=0, snp_errors=0)
                         })

        to_keep = hl.set([
            (hl.Locus("1", 1), ['C', 'CT']),
            (hl.Locus("1", 2), ['C', 'T']),
            (hl.Locus("X", 1), ['C', 'T']),
            (hl.Locus("X", 3), ['C', 'T']),
            (hl.Locus("Y", 1), ['C', 'T']),
            (hl.Locus("Y", 3), ['C', 'T'])
        ])
        self.assertEqual(var.filter(to_keep.contains((var.locus, var.alleles)))
                         .order_by('locus')
                         .select('locus', 'alleles', 'errors').collect(),
                         [
                             hl.utils.Struct(locus=hl.Locus("1", 1), alleles=['C', 'CT'], errors=2),
                             hl.utils.Struct(locus=hl.Locus("1", 2), alleles=['C', 'T'], errors=1),
                             hl.utils.Struct(locus=hl.Locus("X", 1), alleles=['C', 'T'], errors=2),
                             hl.utils.Struct(locus=hl.Locus("X", 3), alleles=['C', 'T'], errors=1),
                             hl.utils.Struct(locus=hl.Locus("Y", 1), alleles=['C', 'T'], errors=1),
                             hl.utils.Struct(locus=hl.Locus("Y", 3), alleles=['C', 'T'], errors=1),
                         ])

        ped2 = hl.Pedigree.read(resource('mendelWithMissingSex.fam'))
        men2, _, _, _ = hl.mendel_errors(mt['GT'], ped2)

        self.assertTrue(men2.filter(men2.s == 'Dtr1')._same(men.filter(men.s == 'Dtr1')))

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
        bad.describe()

        bad = bad.filter(~(
                (bad.t == bad.T) &
                (bad.u == bad.U) &
                (hl.abs(bad.chi_sq - bad.Chi2) < 0.001) &
                (hl.abs(bad.p_value - bad.Pval) < 0.001)))

        if bad.count() != 0:
            bad.order_by(hl.asc(bad.v)).show()
            self.fail('Found rows in violation of the predicate (see show output)')

    def test_de_novo(self):
        mt = hl.import_vcf(resource('denovo.vcf'))
        mt = mt.filter_rows(mt.locus.in_y_par(), keep=False)  # de_novo_finder doesn't know about y PAR
        ped = hl.Pedigree.read(resource('denovo.fam'))
        r = hl.de_novo(mt, ped, mt.info.ESP)
        r = r.select(
            prior=r.prior,
            kid_id=r.proband.s,
            dad_id=r.father.s,
            mom_id=r.mother.s,
            p_de_novo=r.p_de_novo,
            confidence=r.confidence).key_by('locus', 'alleles', 'kid_id', 'dad_id', 'mom_id')

        truth = hl.import_table(resource('denovo.out'), impute=True, comment='#')
        truth = truth.select(
            locus=hl.locus(truth['Chr'], truth['Pos']),
            alleles=[truth['Ref'], truth['Alt']],
            kid_id=truth['Child_ID'],
            dad_id=truth['Dad_ID'],
            mom_id=truth['Mom_ID'],
            p_de_novo=truth['Prob_dn'],
            confidence=truth['Validation_Likelihood'].split('_')[0]).key_by('locus', 'alleles', 'kid_id', 'dad_id',
                                                                            'mom_id')

        j = r.join(truth, how='outer')
        self.assertTrue(j.all((j.confidence == j.confidence_1) & (hl.abs(j.p_de_novo - j.p_de_novo_1) < 1e-4)))
