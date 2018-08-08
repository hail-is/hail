import unittest

import hail as hl

from hail.plot.slippyplot.tile_generator import TileGenerator

from test.hail.helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def setUp(self):
        ht = hl.Table.parallelize([
            {'locus': '1:904165', 'alleles': ['G', 'A'], 'gene': 'A_GENE',
             'phenotype': 'trait1', 'pval': 0.128},
            {'locus': '1:909917', 'alleles': ['G', 'A'], 'gene': 'A_GENE',
             'phenotype': 'trait1', 'pval': 0.766},
        ],
            hl.tstruct(locus=hl.tstr,
                       alleles=hl.tarray(hl.tstr),
                       gene=hl.tstr,
                       phenotype=hl.tstr,
                       pval=hl.tfloat64)
        )
        ht = ht.annotate(locus=hl.parse_locus(ht.locus))

        self.mt = ht.to_matrix_table(['locus', 'alleles'], ['phenotype'],
                                     ['gene'])

        self.colors = {
            '1': "#08ad4d", '2': "#cc0648", '3': "#bbdd11", '4': "#4a87d6",
            '5': "#6f50b7", '6': "#e0c10f", '7': "#d10456", '8': "#2779d8",
            '9': "#9e0631", '10': "#5fcc06", '11': "#4915a8", '12': "#0453d3",
            '13': "#7faf26", '14': "#d17b0c", '15': "#526d13", '16': "#e82019",
            '17': "#125b07", '18': "#12e2c3", '19': "#914ae2", '20': "#95ce10",
            '21': "#af1ca8", '22': "#eaca3a", 'X': "#1c8caf"}

    def test_init(self):
        mt = (hl.experimental.format_manhattan(
            self.mt.locus, self.mt.phenotype, self.mt.pval, colors=self.colors))

        tg = TileGenerator(
            mt,
            dest='destination/to/put/tiles/in',
            empty_tile_path='/Users/maccum/manhattan_data/empty_tile.png',
            regen=True)

    def test_map_value_onto_range(self):
        TG = TileGenerator
        self.assertEqual(TG.map_value_onto_range(9, [5, 10], [10, 20]), 18)
        self.assertEqual(TG.map_value_onto_range(9, [5, 10], [-10, -20]), -18)
        self.assertEqual(TG.map_value_onto_range(9, [5, 10], [-5, 5]), 3)
        self.assertEqual(TG.map_value_onto_range(9, [5, 10], [5, -5]), -3)
        self.assertEqual(TG.map_value_onto_range(-9, [-5, -10], [10, 20]), 18)
        self.assertEqual(TG.map_value_onto_range(3, [2.5, 5], [-15, -10]), -14)

    def test_calculate_gp_range(self):
        mt = (hl.Table.parallelize(
            [{'global_position': 10, 'color': "#08ad4d", 'phenotype': 'trait1',
              'pval': 0.5},
             {'global_position': 190, 'color': "#08ad4d", 'phenotype': 'trait1',
              'pval': 0.6}],
            hl.tstruct(global_position=hl.tint32, color=hl.tstr,
                       phenotype=hl.tstr, pval=hl.tfloat32)
        ).to_matrix_table(['global_position'], ['phenotype'], ['color'])
              .annotate_globals(gp_range=hl.struct(min=10, max=190)))

        mt = mt.annotate_entries(neg_log_pval=-hl.log(mt.pval))

        tg = TileGenerator(mt,
                           dest='temp/plot_temp/',
                           empty_tile_path='/Users/maccum/manhattan_data/empty_tile.png',
                           regen=True,
                           x_margin=10,
                           y_margin=5)

        self.assertEqual(tg.calculate_gp_range(column=0, num_cols=4), [0, 50])
        self.assertEqual(tg.calculate_gp_range(column=1, num_cols=4), [50, 100])
        self.assertEqual(tg.calculate_gp_range(column=2, num_cols=4),
                         [100, 150])
        self.assertEqual(tg.calculate_gp_range(column=3, num_cols=4),
                         [150, 200])

    def test_filter_by_coordinates(self):
        global_positions = [0, 11, 26, 34, 49, 63]
        neg_log_pvals = [3.4, 10.5, 2, 8.9, 1.6, 4.7]
        rows = []
        for i in range(len(global_positions)):
            row = {'global_position': global_positions[i],
                   'neg_log_pval': neg_log_pvals[i], 'color': '#F73A12',
                   'phenotype': 'trait1'}
            rows.append(row)
        mt = hl.Table.parallelize(
            rows, hl.tstruct(global_position=hl.tint64,
                             neg_log_pval=hl.tfloat64,
                             color=hl.tstr,
                             phenotype=hl.tstr)
        ).to_matrix_table(['global_position'], ['phenotype'], ['color'])

        ht = TileGenerator.filter_by_coordinates(
            mt, gp_range=[12, 49], nlp_range=[0, 10], phenotype='trait1'
        ).key_by('global_position').select('neg_log_pval')

        self.assertEqual(set(ht.collect()),
                         {hl.struct(global_position=34, neg_log_pval=8.9).value,
                          hl.struct(global_position=26, neg_log_pval=2.0).value,
                          hl.struct(global_position=49,
                                    neg_log_pval=1.6).value})

    def test_filter_by_pixel(self):
        global_positions = [15000, 15001]
        neg_log_pvals = [250.5, 250.7]
        rows = []
        for i in range(len(global_positions)):
            row = {'global_position': global_positions[i],
                   'neg_log_pval': neg_log_pvals[i], 'color': '#F73A12'}
            rows.append(row)
        ht = hl.Table.parallelize(rows, hl.tstruct(global_position=hl.tint64,
                                                   neg_log_pval=hl.tfloat64,
                                                   color=hl.tstr))
        filtered_ht = TileGenerator.filter_by_pixel(ht, gp_range=[10000, 20000],
                                                    nlp_range=[150.5, 350.5])
        self.assertTrue(len(filtered_ht.collect()) == 1)

    def test_collect_values(self):
        global_positions = [0, 11, 26, 34, 49, 63]
        neg_log_pvals = [3.4, 10.5, 2, 8.9, 1.6, 4.7]
        rows = []
        for i in range(len(global_positions)):
            row = {'global_position': global_positions[i],
                   'neg_log_pval': neg_log_pvals[i], 'color': '#F73A12',
                   'phenotype': 'trait1'}
            rows.append(row)
        ht = hl.Table.parallelize(
            rows, hl.tstruct(global_position=hl.tint64,
                             neg_log_pval=hl.tfloat64,
                             color=hl.tstr,
                             phenotype=hl.tstr)
        )

        global_positions, neg_log_pvals, colors = TileGenerator.collect_values(
            ht)
        self.assertEqual(global_positions, [0, 11, 26, 34, 49, 63])
        self.assertEqual(neg_log_pvals, [3.4, 10.5, 2, 8.9, 1.6, 4.7])
        self.assertEqual(colors,
                         ['#F73A12', '#F73A12', '#F73A12', '#F73A12', '#F73A12',
                          '#F73A12'])

    def test_generate_tiles(self):
        # TODO: test actual image output in some way?
        # TODO: write better image test case for visual testing (show point overlap between tiles, for example)
        mt = (hl.experimental.format_manhattan(
            self.mt.locus, self.mt.phenotype, self.mt.pval, colors=self.colors))

        destination_folder = 'temp/plots'
        tg = TileGenerator(mt,
                           dest=destination_folder,
                           regen=True,
                           log_path='temp/plot_gen.log')
        tg.generate_tile_layer(phenotype="trait1", zoom=2, new_log_file=True)
        self.assertTrue(os.path.exists(destination_folder + "/trait1/2/0.png"))
        self.assertTrue(os.path.exists(destination_folder + "/trait1/2/1.png"))
        self.assertTrue(os.path.exists(destination_folder + "/trait1/2/2.png"))
        self.assertTrue(os.path.exists(destination_folder + "/trait1/2/3.png"))

    def test_empty_tiles_not_computed(self):
        # TODO
        pass
