import unittest

import hail as hl

from test.hail.helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):

    def setUp(self):
        schema = hl.tstruct(locus=hl.tstr, alleles=hl.tarray(hl.tstr),
                            gene=hl.tstr,
                            phenotype=hl.tstr, pval=hl.tfloat32)
        ht = hl.Table.parallelize([
            {'locus': '1:904165', 'alleles': hl.array(['C', 'T']),
             'gene': 'A_GENE', 'phenotype': 'height',
             'pval': 0.1},
            {'locus': '1:909917', 'alleles': hl.array(['A', 'G']),
             'gene': 'A_GENE', 'phenotype': 'height',
             'pval': 0.002},
        ], schema)
        ht = ht.annotate(locus=hl.parse_locus(ht.locus))

        self.mt = ht.to_matrix_table(['locus', 'alleles'], ['phenotype'],
                                     ['gene'])

    def test_format_manhattan(self):
        manhat_mt = hl.experimental.format_manhattan(self.mt.locus,
                                                     self.mt.phenotype,
                                                     self.mt.pval)

        expected_color = "#000000"
        expected_min_nlp = -hl.log(0.1)
        expected_max_nlp = -hl.log(0.002)

        entries = (hl.Table.parallelize([
            {'locus': '1:904165', 'alleles': ['C', 'T'], 'gene': 'A_GENE',
             'phenotype': 'height',
             'pval': 0.1,
             'global_position': 904164, 'color': expected_color,
             'neg_log_pval': -hl.log(0.1), 'under_threshold': False},
            {'locus': '1:909917', 'alleles': ['A', 'G'], 'gene': 'A_GENE',
             'phenotype': 'height',
             'pval': 0.002,
             'global_position': 909916, 'color': expected_color,
             'neg_log_pval': -hl.log(0.002), 'under_threshold': False}],
            hl.tstruct(locus=hl.tstr, alleles=hl.tarray(hl.tstr),
                       gene=hl.tstr,
                       phenotype=hl.tstr,
                       pval=hl.tfloat32, global_position=hl.tint64,
                       color=hl.tstr,
                       neg_log_pval=hl.tfloat64,
                       under_threshold=hl.tbool)
        ))
        expected_mt = (entries.annotate(locus=hl.parse_locus(entries.locus))
            .to_matrix_table(['locus', 'alleles'], ['phenotype'],
                             ['gene', 'global_position', 'color'])
            .annotate_cols(min_nlp=expected_min_nlp,
                           max_nlp=expected_max_nlp)
            .annotate_globals(
            gp_range=hl.struct(min=904164, max=909916)))

        self.assertTrue(manhat_mt.drop('label')._same(expected_mt))
