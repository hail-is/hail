import unittest

import hail as hl
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):

    @staticmethod
    def get_groupable_matrix():
        rt = hl.utils.range_matrix_table(n_rows=100, n_cols=20)
        rt = rt.annotate_globals(foo="foo")
        rt = rt.annotate_rows(group1=rt['row_idx'] % 6,
                              group2=hl.Struct(a=rt['row_idx'] % 6,
                                               b="foo"))
        rt = rt.annotate_cols(group3=rt['col_idx'] % 6,
                              group4=hl.Struct(a=rt['col_idx'] % 6,
                                               b="foo"))
        return rt.annotate_entries(c=rt['row_idx'],
                                   d=rt['col_idx'],
                                   e="foo",
                                   f=rt['group1'],
                                   g=rt['group2']['a'],
                                   h=rt['group3'],
                                   i=rt['group4']['a'])

    def test_errors_caught_correctly(self):
        from hail.expr.expressions import ExpressionException

        mt = self.get_groupable_matrix()
        self.assertRaises(ExpressionException, mt.group_rows_by, mt['group1'] + 1)
        self.assertRaises(ExpressionException, mt.group_cols_by, mt['group1'])
        self.assertRaises(ExpressionException, mt.group_cols_by, mt['group3'] + 1)
        self.assertRaises(ExpressionException, mt.group_rows_by, mt['group3'])
        self.assertRaises(ExpressionException, mt.group_rows_by, group3=mt['group1'])
        self.assertRaises(ExpressionException, mt.group_cols_by, group1=mt['group3'])
        self.assertRaises(ExpressionException, mt.group_rows_by, foo=mt['group1'])
        self.assertRaises(ExpressionException, mt.group_cols_by, foo=mt['group3'])

        a = mt.group_rows_by(group5=(mt['group2']['a'] + 1))
        self.assertRaises(ExpressionException, a.aggregate, group3=hl.agg.sum(mt['c']))
        self.assertRaises(ExpressionException, a.aggregate, group5=hl.agg.sum(mt['c']))
        self.assertRaises(ExpressionException, a.aggregate, foo=hl.agg.sum(mt['c']))

        b = mt.group_cols_by(group5=(mt['group4']['a'] + 1))
        self.assertRaises(ExpressionException, b.aggregate, group1=hl.agg.sum(mt['c']))
        self.assertRaises(ExpressionException, b.aggregate, group5=hl.agg.sum(mt['c']))
        self.assertRaises(ExpressionException, b.aggregate, foo=hl.agg.sum(mt['c']))

    def test_fields_work_correctly(self):
        mt = self.get_groupable_matrix()
        a = mt.group_rows_by(mt['group1']).aggregate(c=hl.agg.sum(mt['c']))
        self.assertEqual(a.count_rows(), 6)
        self.assertTrue('group1' in a.row_key)

        b = mt.group_cols_by(mt['group3']).aggregate(c=hl.agg.sum(mt['c']))
        self.assertEqual(b.count_cols(), 6)
        self.assertTrue('group3' in b.col_key)

    def test_nested_fields_work_correctly(self):
        mt = self.get_groupable_matrix()
        a = mt.group_rows_by(mt['group2']['a']).aggregate(c=hl.agg.sum(mt['c']))
        self.assertEqual(a.count_rows(), 6)
        self.assertTrue('a' in a.row_key)

        b = mt.group_cols_by(mt['group4']['a']).aggregate(c=hl.agg.sum(mt['c']))
        self.assertEqual(b.count_cols(), 6)
        self.assertTrue('a' in b.col_key)

    def test_named_fields_work_correctly(self):
        mt = self.get_groupable_matrix()
        a = mt.group_rows_by(group5=(mt['group2']['a'] + 1)).aggregate(c=hl.agg.sum(mt['c']))
        self.assertEqual(a.count_rows(), 6)
        self.assertTrue('group5' in a.row_key)

        b = mt.group_cols_by(group5=(mt['group4']['a'] + 1)).aggregate(c=hl.agg.sum(mt['c']))
        self.assertEqual(b.count_cols(), 6)
        self.assertTrue('group5' in b.col_key)

    def test_joins_work_correctly(self):
        mt = hl.utils.range_matrix_table(4, 4)
        mt = mt.annotate_globals(glob=5)

        mt2 = hl.utils.range_matrix_table(4, 4)
        mt2 = mt2.annotate_entries(x=mt2.row_idx + mt2.col_idx)
        mt2 = mt2.annotate_rows(row_idx2=mt2.row_idx)
        mt2 = mt2.annotate_cols(col_idx2=mt2.col_idx)

        col_result = (mt.group_cols_by(group=mt2.cols()[mt.col_idx].col_idx2 < 2)
                      .aggregate(sum=hl.agg.sum(mt2[mt.row_idx, mt.col_idx].x + mt.glob) + mt.glob - 15))

        col_expected = (
            hl.Table.parallelize(
                [{'row_idx': 0, 'group': True, 'sum': 1},
                 {'row_idx': 0, 'group': False, 'sum': 5},
                 {'row_idx': 1, 'group': True, 'sum': 3},
                 {'row_idx': 1, 'group': False, 'sum': 7},
                 {'row_idx': 2, 'group': True, 'sum': 5},
                 {'row_idx': 2, 'group': False, 'sum': 9},
                 {'row_idx': 3, 'group': True, 'sum': 7},
                 {'row_idx': 3, 'group': False, 'sum': 11}],
                hl.tstruct(row_idx=hl.tint32, group=hl.tbool, sum=hl.tint64)
            ).annotate_globals(glob=5).key_by('row_idx', 'group')
        )

        self.assertTrue(col_result.entries()._same(col_expected))

        row_result = (mt.group_rows_by(group=mt2.rows()[mt.row_idx].row_idx2 < 2)
                      .aggregate(sum=hl.agg.sum(mt2[mt.row_idx, mt.col_idx].x + mt.glob) + mt.glob - 15))

        row_expected = (
            hl.Table.parallelize(
                [{'group': True, 'col_idx': 0, 'sum': 1},
                 {'group': True, 'col_idx': 1, 'sum': 3},
                 {'group': True, 'col_idx': 2, 'sum': 5},
                 {'group': True, 'col_idx': 3, 'sum': 7},
                 {'group': False, 'col_idx': 0, 'sum': 5},
                 {'group': False, 'col_idx': 1, 'sum': 7},
                 {'group': False, 'col_idx': 2, 'sum': 9},
                 {'group': False, 'col_idx': 3, 'sum': 11}],
                hl.tstruct(group=hl.tbool, col_idx=hl.tint32, sum=hl.tint64)
            ).annotate_globals(glob=5).key_by('group', 'col_idx')
        )

        self.assertTrue(row_result.entries()._same(row_expected))
