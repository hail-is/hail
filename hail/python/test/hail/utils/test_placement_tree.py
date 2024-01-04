import unittest

import hail as hl

from hail.utils.placement_tree import PlacementTree


class Tests(unittest.TestCase):
    def test_realistic(self):
        dtype = hl.dtype(
            '''struct{
locus: locus<GRCh37>,
alleles: array<str>,
rsid: str,
qual: float64,
filters: set<str>,
info: struct{
  NEGATIVE_TRAIN_SITE: bool,
  HWP: float64,
  AC: array<int32>},
empty_struct: struct{
},
variant_qc: struct{
  dp_stats: struct{
    mean: float64,
    stdev: float64,
    min: float64,
    max: float64},
  gq_stats: struct{
    mean: float64,
    stdev: float64,
    min: float64,
    max: float64},
  AC: array<int32>,
  AF: array<float64>,
  AN: int32,
  homozygote_count: array<int32>,
  call_rate: float64}}'''
        )
        tree = PlacementTree.from_named_type('row', dtype)
        grid = tree.to_grid()
        assert len(grid) == 4

        row1 = grid[1]
        assert len(row1) == 8
        for i in range(5):
            assert row1[i] == (None, 1)
        assert row1[5] == (None, 3)
        assert row1[7] == ('variant_qc', 13)

        row2 = grid[2]
        assert len(row2) == 14
        for i in range(5):
            assert row2[i] == (None, 1)
        assert row2[5] == ('info', 3)
        assert row2[7] == ('dp_stats', 4)
        assert row2[8] == ('gq_stats', 4)
        for i in range(9, 13):
            assert row2[i] == (None, 1)

        row3 = grid[3]
        assert row3 == [
            ('locus', 1),
            ('alleles', 1),
            ('rsid', 1),
            ('qual', 1),
            ('filters', 1),
            ('NEGATIVE_TRAIN_SITE', 1),
            ('HWP', 1),
            ('AC', 1),
            ('mean', 1),
            ('stdev', 1),
            ('min', 1),
            ('max', 1),
            ('mean', 1),
            ('stdev', 1),
            ('min', 1),
            ('max', 1),
            ('AC', 1),
            ('AF', 1),
            ('AN', 1),
            ('homozygote_count', 1),
            ('call_rate', 1),
        ]
