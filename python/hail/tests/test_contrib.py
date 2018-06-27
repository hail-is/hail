import unittest

import hail as hl
from .utils import resource, startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

class Tests(unittest.TestCase):

    def test_ld_score(self):
        
        ht = hl.import_table(resource('ldsc.annot'), types={'BP': hl.tint, 'CM': hl.tfloat, 'univariate': hl.tfloat, 'binary': hl.tfloat, 'continuous': hl.tfloat})
        ht = ht.annotate(locus=hl.locus(ht.CHR, ht.BP))
        ht = ht.key_by('locus')

        mt = hl.import_plink(bed=resource('ldsc.bed'), bim=resource('ldsc.bim'), fam=resource('ldsc.fam'))
        mt = mt.annotate_rows(stats=hl.agg.stats(mt.GT.n_alt_alleles()))
        mt = mt.annotate_rows(univariate=ht[mt.locus].univariate, binary=ht[mt.locus].binary, continuous=ht[mt.locus].continuous)

        ht_scores = hl.ld_score(entry_expr=hl.or_else((mt.GT.n_alt_alleles() - mt.stats.mean)/mt.stats.stdev, 0.0),
                                annotation_exprs=[mt.univariate, mt.binary, mt.continuous],
                                position_expr=mt.cm_position,
                                window_size=1)

        first_scores = ht_scores.aggregate(hl.struct(univariate=hl.agg.collect(hl.agg.filter((ht_scores.locus.contig == '20') & 
                                                                                             (ht_scores.locus.position == 82079), ht_scores.univariate))[0],
                                                     binary=hl.agg.collect(hl.agg.filter((ht_scores.locus.contig == '20') & 
                                                                                         (ht_scores.locus.position == 82079), ht_scores.binary))[0],
                                                     continuous=hl.agg.collect(hl.agg.filter((ht_scores.locus.contig == '20') & 
                                                                                             (ht_scores.locus.position == 82079), ht_scores.continuous))[0]))

        self.assertAlmostEqual(first_scores.univariate, 1.601, places=3)
        self.assertAlmostEqual(first_scores.binary, 1.152, places=3)
        self.assertAlmostEqual(first_scores.continuous, 73.014, places=3)

        mean_scores = ht_scores.aggregate(hl.struct(univariate=hl.agg.mean(ht_scores.univariate),
                                                    binary=hl.agg.mean(ht_scores.binary),
                                                    continuous=hl.agg.mean(ht_scores.continuous)))

        self.assertAlmostEqual(mean_scores.univariate, 3.826, places=3)
        self.assertAlmostEqual(mean_scores.binary, 0.996, places=3)
        self.assertAlmostEqual(mean_scores.continuous, 199.469, places=3)
