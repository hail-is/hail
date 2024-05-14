import unittest

import hail as hl

from ..helpers import qobtest


class GroupedTableTests(unittest.TestCase):
    @qobtest
    def test_aggregate_by(self):
        ht = hl.utils.range_table(4)
        ht = ht.annotate(foo=0, group=ht.idx < 2, bar='hello').annotate_globals(glob=5)
        grouped = ht.group_by(ht.group)
        result = grouped.aggregate(sum=hl.agg.sum(ht.idx + ht.glob) + ht.glob - 15, max=hl.agg.max(ht.idx))

        expected = (
            hl.Table.parallelize(
                [{'group': True, 'sum': 1, 'max': 1}, {'group': False, 'sum': 5, 'max': 3}],
                hl.tstruct(group=hl.tbool, sum=hl.tint64, max=hl.tint32),
            )
            .annotate_globals(glob=5)
            .key_by('group')
        )

        self.assertTrue(result._same(expected))

        with self.assertRaises(ValueError):
            grouped.aggregate(group=hl.agg.sum(ht.idx))

    def test_aggregate_by_with_joins(self):
        ht = hl.utils.range_table(4)
        ht2 = hl.utils.range_table(4)
        ht2 = ht2.annotate(idx2=ht2.idx)

        ht = ht.annotate_globals(glob=5)
        grouped = ht.group_by(group=ht2[ht.idx].idx2 < 2)
        result = grouped.aggregate(
            sum=hl.agg.sum(ht2[ht.idx].idx2 + ht.glob) + ht.glob - 15, max=hl.agg.max(ht2[ht.idx].idx2)
        )

        expected = (
            hl.Table.parallelize(
                [{'group': True, 'sum': 1, 'max': 1}, {'group': False, 'sum': 5, 'max': 3}],
                hl.tstruct(group=hl.tbool, sum=hl.tint64, max=hl.tint32),
            )
            .annotate_globals(glob=5)
            .key_by('group')
        )

        self.assertTrue(result._same(expected))

    def test_issue_2446_takeby(self):
        t = hl.utils.range_table(10)
        result = t.group_by(foo=5).aggregate(x=hl.agg.take(t.idx, 3, ordering=t.idx))
        self.assertTrue(result.collect()[0].x == [0, 1, 2])
