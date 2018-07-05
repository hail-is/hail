import hail as hl
import unittest
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):

    def test_ld_score(self):

        ht = hl.import_table(doctest_resource('ldsc.annot'),
                             types={'BP': hl.tint,
                                    'CM': hl.tfloat,
                                    'univariate': hl.tfloat,
                                    'binary': hl.tfloat,
                                    'continuous': hl.tfloat})
        ht = ht.annotate(locus=hl.locus(ht.CHR, ht.BP))
        ht = ht.key_by('locus')

        mt = hl.import_plink(bed=doctest_resource('ldsc.bed'),
                             bim=doctest_resource('ldsc.bim'),
                             fam=doctest_resource('ldsc.fam'))
        mt = mt.annotate_rows(stats=hl.agg.stats(mt.GT.n_alt_alleles()))
        mt = mt.annotate_rows(univariate=1,
                              binary=ht[mt.locus].binary,
                              continuous=ht[mt.locus].continuous)

        mt = mt.annotate_entries(GT_std=hl.or_else(
            (mt.GT.n_alt_alleles() - mt.stats.mean)/mt.stats.stdev, 0.0))

        ht_scores = hl.experimental.ld_score(entry_expr=mt.GT_std,
                                             annotation_exprs=[mt.univariate,
                                                               mt.binary,
                                                               mt.continuous],
                                             position_expr=mt.cm_position,
                                             window_size=1)

        chr20_firsts = ht_scores.aggregate(
            hl.struct(univariate=hl.agg.collect(
                        hl.agg.filter((ht_scores.locus.contig == '20') &
                                      (ht_scores.locus.position == 82079),
                                      ht_scores.univariate))[0],
                      binary=hl.agg.collect(
                        hl.agg.filter((ht_scores.locus.contig == '20') &
                                      (ht_scores.locus.position == 82079),
                                      ht_scores.binary))[0],
                      continuous=hl.agg.collect(
                        hl.agg.filter((ht_scores.locus.contig == '20') &
                                      (ht_scores.locus.position == 82079),
                                      ht_scores.continuous))[0]))

        self.assertAlmostEqual(chr20_firsts.univariate, 1.601, places=3)
        self.assertAlmostEqual(chr20_firsts.binary, 1.152, places=3)
        self.assertAlmostEqual(chr20_firsts.continuous, 73.014, places=3)

        chr22_firsts = ht_scores.aggregate(
            hl.struct(univariate=hl.agg.collect(
                        hl.agg.filter((ht_scores.locus.contig == '22') &
                                      (ht_scores.locus.position == 16894090),
                                      ht_scores.univariate))[0],
                      binary=hl.agg.collect(
                        hl.agg.filter((ht_scores.locus.contig == '22') &
                                      (ht_scores.locus.position == 16894090),
                                      ht_scores.binary))[0],
                      continuous=hl.agg.collect(
                        hl.agg.filter((ht_scores.locus.contig == '22') &
                                      (ht_scores.locus.position == 16894090),
                                      ht_scores.continuous))[0]))

        self.assertAlmostEqual(chr22_firsts.univariate, 1.140, places=3)
        self.assertAlmostEqual(chr22_firsts.binary, 1.107, places=3)
        self.assertAlmostEqual(chr22_firsts.continuous, 102.174, places=3)

        means = ht_scores.aggregate(hl.struct(
            univariate=hl.agg.mean(ht_scores.univariate),
            binary=hl.agg.mean(ht_scores.binary),
            continuous=hl.agg.mean(ht_scores.continuous)))

        self.assertAlmostEqual(means.univariate, 3.507, places=3)
        self.assertAlmostEqual(means.binary, 0.965, places=3)
        self.assertAlmostEqual(means.continuous, 176.528, places=3)
