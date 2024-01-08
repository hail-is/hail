import math
import operator
import random
import unittest

import pytest

import hail as hl
import hail.expr.aggregators as agg
from hail import ir
from hail.utils.java import Env
from hail.utils.misc import new_temp_file

from ..helpers import (
    convert_struct_to_dict,
    create_all_values_matrix_table,
    fails_local_backend,
    fails_service_backend,
    get_dataset,
    qobtest,
    resource,
    schema_eq,
    test_timeout,
)


class Tests(unittest.TestCase):
    def get_mt(self, min_partitions=None) -> hl.MatrixTable:
        return hl.import_vcf(resource("sample.vcf"), min_partitions=min_partitions)

    @qobtest
    def test_range_count(self):
        self.assertEqual(hl.utils.range_matrix_table(7, 13).count(), (7, 13))

    def test_row_key_field_show_runs(self):
        ds = self.get_mt()
        ds.locus.show()

    def test_update(self):
        mt = self.get_mt()
        mt = mt.select_entries(dp=mt.DP, gq=mt.GQ)
        self.assertTrue(schema_eq(mt.entry.dtype, hl.tstruct(dp=hl.tint32, gq=hl.tint32)))

    def test_annotate(self):
        mt = self.get_mt()
        mt = mt.annotate_globals(foo=5)

        self.assertEqual(mt.globals.dtype, hl.tstruct(foo=hl.tint32))

        mt = mt.annotate_rows(x1=agg.count(), x2=agg.fraction(False), x3=agg.count_where(True), x4=mt.info.AC + mt.foo)

        mt = mt.annotate_cols(apple=6)
        mt = mt.annotate_cols(y1=agg.count(), y2=agg.fraction(False), y3=agg.count_where(True), y4=mt.foo + mt.apple)

        expected_schema = hl.tstruct(
            s=hl.tstr, apple=hl.tint32, y1=hl.tint64, y2=hl.tfloat64, y3=hl.tint64, y4=hl.tint32
        )

        self.assertTrue(
            schema_eq(mt.col.dtype, expected_schema),
            "expected: " + str(mt.col.dtype) + "\nactual: " + str(expected_schema),
        )

        mt = mt.select_entries(z1=mt.x1 + mt.foo, z2=mt.x1 + mt.y1 + mt.foo)
        self.assertTrue(schema_eq(mt.entry.dtype, hl.tstruct(z1=hl.tint64, z2=hl.tint64)))

    def test_annotate_globals(self):
        mt = hl.utils.range_matrix_table(1, 1)
        ht = hl.utils.range_table(1, 1)
        data = [
            (5, hl.tint, operator.eq),
            (float('nan'), hl.tfloat32, lambda x, y: str(x) == str(y)),
            (float('inf'), hl.tfloat64, lambda x, y: str(x) == str(y)),
            (float('-inf'), hl.tfloat64, lambda x, y: str(x) == str(y)),
            (1.111, hl.tfloat64, operator.eq),
            (
                [hl.Struct(**{'a': None, 'b': 5}), hl.Struct(**{'a': 'hello', 'b': 10})],
                hl.tarray(hl.tstruct(a=hl.tstr, b=hl.tint)),
                operator.eq,
            ),
        ]

        for x, t, f in data:
            self.assertTrue(f(hl.eval(mt.annotate_globals(foo=hl.literal(x, t)).foo), x), f"{x}, {t}")
            self.assertTrue(f(hl.eval(ht.annotate_globals(foo=hl.literal(x, t)).foo), x), f"{x}, {t}")

    def test_head_no_empty_partitions(self):
        mt = hl.utils.range_matrix_table(10, 10)

        tmp_file = new_temp_file(extension='mt')

        mt.write(tmp_file)
        mt_readback = hl.read_matrix_table(tmp_file)
        for mt_ in [mt, mt_readback]:
            assert mt_.head(1).count_rows() == 1
            assert mt_.head(1)._force_count_rows() == 1
            assert mt_.head(100).count_rows() == 10
            assert mt_.head(100)._force_count_rows() == 10

    def test_head_empty_partitions_at_front(self):
        mt = hl.utils.range_matrix_table(20, 10, 20)
        mt = mt.filter_rows(mt.row_idx > 9)

        tmp_file = new_temp_file(extension='mt')

        mt.write(tmp_file)
        mt_readback = hl.read_matrix_table(tmp_file)
        for mt_ in [mt, mt_readback]:
            assert mt_.head(1).count_rows() == 1
            assert mt_.head(1)._force_count_rows() == 1
            assert mt_.head(100).count_rows() == 10
            assert mt_.head(100)._force_count_rows() == 10

    def test_head_rows_and_cols(self):
        mt1 = hl.utils.range_matrix_table(10, 10)
        assert mt1.head(1, 2).count() == (1, 2)

    def test_head_rows(self):
        mt1 = hl.utils.range_matrix_table(10, 10)
        assert mt1.head(1, None).count() == (1, 10)

    def test_head_cols(self):
        mt1 = hl.utils.range_matrix_table(10, 10)
        assert mt1.head(None, 1).count() == (10, 1)

    @test_timeout(batch=5 * 60)
    def test_tail_no_empty_partitions(self):
        mt = hl.utils.range_matrix_table(10, 10)

        tmp_file = new_temp_file(extension='mt')
        mt.write(tmp_file)
        mt_readback = hl.read_matrix_table(tmp_file)
        for mt_ in [mt, mt_readback]:
            assert mt_.tail(1).count_rows() == 1
            assert mt_.tail(1)._force_count_rows() == 1
            assert mt_.tail(100).count_rows() == 10
            assert mt_.tail(100)._force_count_rows() == 10

    @test_timeout(batch=5 * 60)
    def test_tail_empty_partitions_at_front(self):
        mt = hl.utils.range_matrix_table(20, 10, 20)
        mt = mt.filter_rows(mt.row_idx > 9)

        tmp_file = new_temp_file(extension='mt')
        mt.write(tmp_file)
        mt_readback = hl.read_matrix_table(tmp_file)
        for mt_ in [mt, mt_readback]:
            assert mt_.tail(1).count_rows() == 1
            assert mt_.tail(1)._force_count_rows() == 1
            assert mt_.tail(100).count_rows() == 10
            assert mt_.tail(100)._force_count_rows() == 10

    def test_tail_cols(self):
        mt1 = hl.utils.range_matrix_table(10, 10)
        assert mt1.tail(1, 2).count() == (1, 2)
        assert mt1.tail(1, None).count() == (1, 10)
        assert mt1.tail(None, 1).count() == (10, 1)

    def test_tail_entries(self):
        mt = hl.utils.range_matrix_table(100, 30)
        mt = mt.filter_cols(mt.col_idx != 29)

        def tail(*args):
            ht = mt.tail(*args).entries()
            return ht.aggregate(hl.agg.collect_as_set(hl.tuple([ht.row_idx, ht.col_idx])))

        def expected(n, m):
            return set((i, j) for i in range(100 - n, 100) for j in range(29 - m, 29))

        assert tail(None, 10) == expected(100, 10)
        assert tail(30, None) == expected(30, 29)
        assert tail(30, 10) == expected(30, 10)

    def test_tail_scan(self):
        mt = hl.utils.range_matrix_table(30, 40)
        mt = mt.annotate_rows(i=hl.scan.count())
        mt = mt.annotate_cols(j=hl.scan.count())
        mt = mt.tail(10, 11)
        ht = mt.entries()
        assert ht.aggregate(agg.collect_as_set(hl.tuple([ht.i, ht.j]))) == set(
            (i, j) for i in range(20, 30) for j in range(29, 40)
        )

    def test_filter(self):
        mt = self.get_mt()
        mt = mt.annotate_globals(foo=5)
        mt = mt.annotate_rows(x1=agg.count())
        mt = mt.annotate_cols(y1=agg.count())
        mt = mt.annotate_entries(z1=mt.DP)

        mt = mt.filter_rows((mt.x1 == 5) & (agg.count() == 3) & (mt.foo == 2))
        mt = mt.filter_cols((mt.y1 == 5) & (agg.count() == 3) & (mt.foo == 2))
        mt = mt.filter_entries((mt.z1 < 5) & (mt.y1 == 3) & (mt.x1 == 5) & (mt.foo == 2))
        mt.count_rows()

    def test_aggregate_rows(self):
        mt = self.get_mt()

        mt = mt.annotate_globals(foo=5)
        mt = mt.annotate_rows(x1=agg.count())
        mt = mt.annotate_cols(y1=agg.count())
        mt = mt.annotate_entries(z1=mt.DP)

        qv = mt.aggregate_rows(agg.count())
        self.assertEqual(qv, 346)

        mt.aggregate_rows(hl.Struct(x=agg.collect(mt.locus.contig), y=agg.collect(mt.x1)))

    def test_aggregate_cols(self):
        mt = self.get_mt()

        mt = mt.annotate_globals(foo=5)
        mt = mt.annotate_rows(x1=agg.count())
        mt = mt.annotate_cols(y1=agg.count())
        mt = mt.annotate_entries(z1=mt.DP)

        qs = mt.aggregate_cols(agg.count())
        self.assertEqual(qs, 100)
        qs = hl.eval(mt.aggregate_cols(agg.count(), _localize=False))
        self.assertEqual(qs, 100)

        mt.aggregate_cols(hl.Struct(x=agg.collect(mt.s), y=agg.collect(mt.y1)))

    def test_aggregate_cols_order(self):
        path = new_temp_file(extension='mt')
        mt = hl.utils.range_matrix_table(3, 3)
        mt = mt.choose_cols([2, 1, 0])
        mt = mt.checkpoint(path)
        assert mt.aggregate_cols(hl.agg.collect(mt.col_idx)) == [0, 1, 2]
        mt = mt.key_cols_by()
        assert mt.aggregate_cols(hl.agg.collect(mt.col_idx)) == [2, 1, 0]

    def test_aggregate_entries(self):
        mt = self.get_mt()

        mt = mt.annotate_globals(foo=5)
        mt = mt.annotate_rows(x1=agg.count())
        mt = mt.annotate_cols(y1=agg.count())
        mt = mt.annotate_entries(z1=mt.DP)

        qg = mt.aggregate_entries(agg.count())
        self.assertEqual(qg, 34600)

        mt.aggregate_entries(
            hl.Struct(x=agg.filter(False, agg.collect(mt.y1)), y=agg.filter(hl.rand_bool(0.1), agg.collect(mt.GT)))
        )
        self.assertIsNotNone(mt.aggregate_entries(hl.agg.take(mt.s, 1)[0]))

    def test_aggregate_rows_array_agg(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.annotate_rows(maf_flag=hl.empty_array('bool'))
        mt.aggregate_rows(hl.agg.array_agg(lambda x: hl.agg.counter(x), mt.maf_flag))

    def test_aggregate_rows_bn_counter(self):
        r = hl.balding_nichols_model(3, 10, 10).rows()
        r.aggregate(hl.agg.counter(r.locus.in_x_nonpar()))

    def test_col_agg_no_rows(self):
        mt = hl.utils.range_matrix_table(3, 3).filter_rows(False)
        mt = mt.annotate_cols(x=hl.agg.count())
        assert mt.x.collect() == [0, 0, 0]

    def test_col_collect(self):
        mt = hl.utils.range_matrix_table(3, 3)
        mt.cols().collect()

    def test_aggregate_ir(self):
        ds = hl.utils.range_matrix_table(5, 5).annotate_globals(g1=5).annotate_entries(e1=3)

        x = [("col_idx", lambda e: ds.aggregate_cols(e)), ("row_idx", lambda e: ds.aggregate_rows(e))]

        for name, f in x:
            r = f(
                hl.struct(
                    x=agg.sum(ds[name]) + ds.g1,
                    y=agg.filter(ds[name] % 2 != 0, agg.sum(ds[name] + 2)) + ds.g1,
                    z=agg.sum(ds.g1 + ds[name]) + ds.g1,
                    mean=agg.mean(ds[name]),
                )
            )
            self.assertEqual(convert_struct_to_dict(r), {u'x': 15, u'y': 13, u'z': 40, u'mean': 2.0})

            r = f(5)
            self.assertEqual(r, 5)

            r = f(hl.missing(hl.tint32))
            self.assertEqual(r, None)

            r = f(agg.filter(ds[name] % 2 != 0, agg.sum(ds[name] + 2)) + ds.g1)
            self.assertEqual(r, 13)

        r = ds.aggregate_entries(
            agg.filter((ds.row_idx % 2 != 0) & (ds.col_idx % 2 != 0), agg.sum(ds.e1 + ds.g1 + ds.row_idx + ds.col_idx))
            + ds.g1
        )
        self.assertTrue(r, 48)

    def test_select_entries(self):
        mt = hl.utils.range_matrix_table(10, 10, n_partitions=4)
        mt = mt.annotate_entries(a=hl.struct(b=mt.row_idx, c=mt.col_idx), foo=mt.row_idx * 10 + mt.col_idx)
        mt = mt.select_entries(mt.a.b, mt.a.c, mt.foo)
        mt = mt.annotate_entries(bc=mt.b * 10 + mt.c)
        mt_entries = mt.entries()

        assert mt_entries.all(mt_entries.bc == mt_entries.foo)

    def test_select_cols(self):
        mt = hl.utils.range_matrix_table(3, 5, n_partitions=4)
        mt = mt.annotate_entries(e=mt.col_idx * mt.row_idx)
        mt = mt.annotate_globals(g=1)
        mt = mt.annotate_cols(
            sum=agg.sum(mt.e + mt.col_idx + mt.row_idx + mt.g) + mt.col_idx + mt.g,
            count=agg.count_where(mt.e % 2 == 0),
            foo=agg.count(),
        )

        result = convert_struct_to_dict(mt.cols().collect()[-2])
        self.assertEqual(result, {'col_idx': 3, 'sum': 28, 'count': 2, 'foo': 3})

    def test_drop(self):
        mt = self.get_mt()
        mt = mt.annotate_globals(foo=5)
        mt = mt.annotate_cols(bar=5)
        mt1 = mt.drop('GT', 'info', 'foo', 'bar')
        self.assertTrue('foo' not in mt1.globals)
        self.assertTrue('info' not in mt1.row)
        self.assertTrue('bar' not in mt1.col)
        self.assertTrue('GT' not in mt1.entry)
        mt1._force_count_rows()

        mt2 = mt.drop(mt.GT, mt.info, mt.foo, mt.bar)
        self.assertTrue('foo' not in mt2.globals)
        self.assertTrue('info' not in mt2.row)
        self.assertTrue('bar' not in mt2.col)
        self.assertTrue('GT' not in mt2.entry)
        mt2._force_count_rows()

    def test_explode_rows(self):
        mt = hl.utils.range_matrix_table(4, 4)
        mt = mt.annotate_entries(e=mt.row_idx * 10 + mt.col_idx)

        self.assertTrue(mt.annotate_rows(x=[1]).explode_rows('x').drop('x')._same(mt))

        self.assertEqual(mt.annotate_rows(x=hl.empty_array('int')).explode_rows('x').count_rows(), 0)
        self.assertEqual(mt.annotate_rows(x=hl.missing('array<int>')).explode_rows('x').count_rows(), 0)
        self.assertEqual(mt.annotate_rows(x=hl.range(0, mt.row_idx)).explode_rows('x').count_rows(), 6)
        mt = mt.annotate_rows(x=hl.struct(y=hl.range(0, mt.row_idx)))
        self.assertEqual(mt.explode_rows(mt.x.y).count_rows(), 6)

    def test_explode_cols(self):
        mt = hl.utils.range_matrix_table(4, 4)
        mt = mt.annotate_entries(e=mt.row_idx * 10 + mt.col_idx)

        self.assertTrue(mt.annotate_cols(x=[1]).explode_cols('x').drop('x')._same(mt))

        self.assertEqual(mt.annotate_cols(x=hl.empty_array('int')).explode_cols('x').count_cols(), 0)
        self.assertEqual(mt.annotate_cols(x=hl.missing('array<int>')).explode_cols('x').count_cols(), 0)
        self.assertEqual(mt.annotate_cols(x=hl.range(0, mt.col_idx)).explode_cols('x').count_cols(), 6)

    def test_explode_key_errors(self):
        mt = hl.utils.range_matrix_table(1, 1).key_cols_by(a=[1]).key_rows_by(b=[1])
        with self.assertRaises(ValueError):
            mt.explode_cols('a')
        with self.assertRaises(ValueError):
            mt.explode_rows('b')

    def test_group_by_field_lifetimes(self):
        mt = hl.utils.range_matrix_table(3, 3)
        mt2 = mt.group_rows_by(row_idx='100').aggregate(x=hl.agg.collect_as_set(mt.row_idx + 5))
        assert mt2.aggregate_entries(hl.agg.all(mt2.x == hl.set({5, 6, 7})))

        mt3 = mt.group_cols_by(col_idx='100').aggregate(x=hl.agg.collect_as_set(mt.col_idx + 5))
        assert mt3.aggregate_entries(hl.agg.all(mt3.x == hl.set({5, 6, 7})))

    def test_aggregate_cols_by(self):
        mt = hl.utils.range_matrix_table(2, 4)
        mt = mt.annotate_cols(group=mt.col_idx < 2).annotate_globals(glob=5)
        grouped = mt.group_cols_by(mt.group)
        result = grouped.aggregate(sum=hl.agg.sum(mt.row_idx * 2 + mt.col_idx + mt.glob) + 3)

        expected = (
            hl.Table.parallelize(
                [
                    {'row_idx': 0, 'group': True, 'sum': 14},
                    {'row_idx': 0, 'group': False, 'sum': 18},
                    {'row_idx': 1, 'group': True, 'sum': 18},
                    {'row_idx': 1, 'group': False, 'sum': 22},
                ],
                hl.tstruct(row_idx=hl.tint, group=hl.tbool, sum=hl.tint64),
            )
            .annotate_globals(glob=5)
            .key_by('row_idx', 'group')
        )

        self.assertTrue(result.entries()._same(expected))

    def test_aggregate_cols_by_init_op(self):
        mt = hl.import_vcf(resource('sample.vcf'))
        cs = mt.group_cols_by(mt.s).aggregate(cs=hl.agg.call_stats(mt.GT, mt.alleles))
        cs._force_count_rows()  # should run without error

    def test_aggregate_cols_scope_violation(self):
        mt = get_dataset()
        with pytest.raises(hl.expr.ExpressionException) as exc:
            mt.aggregate_cols(hl.agg.filter(False, hl.agg.sum(mt.GT.is_non_ref())))
        assert "scope violation" in str(exc.value)

    def test_aggregate_rows_by(self):
        mt = hl.utils.range_matrix_table(4, 2)
        mt = mt.annotate_rows(group=mt.row_idx < 2).annotate_globals(glob=5)
        grouped = mt.group_rows_by(mt.group)
        result = grouped.aggregate(sum=hl.agg.sum(mt.col_idx * 2 + mt.row_idx + mt.glob) + 3)

        expected = (
            hl.Table.parallelize(
                [
                    {'col_idx': 0, 'group': True, 'sum': 14},
                    {'col_idx': 1, 'group': True, 'sum': 18},
                    {'col_idx': 0, 'group': False, 'sum': 18},
                    {'col_idx': 1, 'group': False, 'sum': 22},
                ],
                hl.tstruct(group=hl.tbool, col_idx=hl.tint, sum=hl.tint64),
            )
            .annotate_globals(glob=5)
            .key_by('group', 'col_idx')
        )

        self.assertTrue(result.entries()._same(expected))

    @qobtest
    def test_collect_cols_by_key(self):
        mt = hl.utils.range_matrix_table(3, 3)
        col_dict = hl.literal({0: [1], 1: [2, 3], 2: [4, 5, 6]})
        mt = mt.annotate_cols(foo=col_dict.get(mt.col_idx)).explode_cols('foo')
        mt = mt.annotate_entries(bar=mt.row_idx * mt.foo)

        grouped = mt.collect_cols_by_key()

        self.assertListEqual(
            grouped.cols().order_by('col_idx').collect(),
            [hl.Struct(col_idx=0, foo=[1]), hl.Struct(col_idx=1, foo=[2, 3]), hl.Struct(col_idx=2, foo=[4, 5, 6])],
        )
        self.assertListEqual(
            grouped.entries().select('bar').order_by('row_idx', 'col_idx').collect(),
            [
                hl.Struct(row_idx=0, col_idx=0, bar=[0]),
                hl.Struct(row_idx=0, col_idx=1, bar=[0, 0]),
                hl.Struct(row_idx=0, col_idx=2, bar=[0, 0, 0]),
                hl.Struct(row_idx=1, col_idx=0, bar=[1]),
                hl.Struct(row_idx=1, col_idx=1, bar=[2, 3]),
                hl.Struct(row_idx=1, col_idx=2, bar=[4, 5, 6]),
                hl.Struct(row_idx=2, col_idx=0, bar=[2]),
                hl.Struct(row_idx=2, col_idx=1, bar=[4, 6]),
                hl.Struct(row_idx=2, col_idx=2, bar=[8, 10, 12]),
            ],
        )

    def test_collect_cols_by_key_with_rand(self):
        mt = hl.utils.range_matrix_table(3, 3)
        mt = mt.annotate_cols(x=hl.rand_norm())
        mt = mt.collect_cols_by_key()
        mt = mt.annotate_cols(x=hl.rand_norm())
        mt.cols().collect()

    def test_weird_names(self):
        ds = self.get_mt()
        exprs = {'a': 5, '   a    ': 5, r'\%!^!@#&#&$%#$%': [5], '$': 5, 'ß': 5}

        ds.annotate_globals(**exprs)
        ds.select_globals(**exprs)

        ds.annotate_cols(**exprs)
        ds1 = ds.select_cols(**exprs)

        ds.annotate_rows(**exprs)
        ds2 = ds.select_rows(**exprs)

        ds.annotate_entries(**exprs)
        ds.select_entries(**exprs)

        ds1.explode_cols(r'\%!^!@#&#&$%#$%')
        ds1.explode_cols(ds1[r'\%!^!@#&#&$%#$%'])
        ds1.group_cols_by(ds1.a).aggregate(**{'*``81': agg.count()})

        ds1.drop(r'\%!^!@#&#&$%#$%')
        ds1.drop(ds1[r'\%!^!@#&#&$%#$%'])

        ds2.explode_rows(r'\%!^!@#&#&$%#$%')
        ds2.explode_rows(ds2[r'\%!^!@#&#&$%#$%'])
        ds2.group_rows_by(ds2.a).aggregate(**{'*``81': agg.count()})

    def test_semi_anti_join_rows(self):
        mt = hl.utils.range_matrix_table(10, 3)
        ht = hl.utils.range_table(3)
        mt2 = mt.key_rows_by(k1=mt.row_idx, k2=hl.str(mt.row_idx * 2))
        ht2 = ht.key_by(k1=ht.idx, k2=hl.str(ht.idx * 2))

        assert mt.semi_join_rows(ht).count() == (3, 3)
        assert mt.anti_join_rows(ht).count() == (7, 3)
        assert mt2.semi_join_rows(ht).count() == (3, 3)
        assert mt2.anti_join_rows(ht).count() == (7, 3)
        assert mt2.semi_join_rows(ht2).count() == (3, 3)
        assert mt2.anti_join_rows(ht2).count() == (7, 3)

        with pytest.raises(ValueError, match='semi_join_rows: cannot join'):
            mt.semi_join_rows(ht2)
        with pytest.raises(ValueError, match='semi_join_rows: cannot join'):
            mt.semi_join_rows(ht.key_by())

        with pytest.raises(ValueError, match='anti_join_rows: cannot join'):
            mt.anti_join_rows(ht2)
        with pytest.raises(ValueError, match='anti_join_rows: cannot join'):
            mt.anti_join_rows(ht.key_by())

    def test_semi_anti_join_cols(self):
        mt = hl.utils.range_matrix_table(3, 10)
        ht = hl.utils.range_table(3)
        mt2 = mt.key_cols_by(k1=mt.col_idx, k2=hl.str(mt.col_idx * 2))
        ht2 = ht.key_by(k1=ht.idx, k2=hl.str(ht.idx * 2))

        assert mt.semi_join_cols(ht).count() == (3, 3)
        assert mt.anti_join_cols(ht).count() == (3, 7)
        assert mt2.semi_join_cols(ht).count() == (3, 3)
        assert mt2.anti_join_cols(ht).count() == (3, 7)
        assert mt2.semi_join_cols(ht2).count() == (3, 3)
        assert mt2.anti_join_cols(ht2).count() == (3, 7)

        with pytest.raises(ValueError, match='semi_join_cols: cannot join'):
            mt.semi_join_cols(ht2)
        with pytest.raises(ValueError, match='semi_join_cols: cannot join'):
            mt.semi_join_cols(ht.key_by())

        with pytest.raises(ValueError, match='anti_join_cols: cannot join'):
            mt.anti_join_cols(ht2)
        with pytest.raises(ValueError, match='anti_join_cols: cannot join'):
            mt.anti_join_cols(ht.key_by())

    def test_joins(self):
        mt = self.get_mt().select_rows(x1=1, y1=1)
        mt2 = mt.select_rows(x2=1, y2=2)
        mt2 = mt2.select_cols(c1=1, c2=2)

        mt = mt.annotate_rows(y2=mt2.index_rows(mt.row_key).y2)
        mt = mt.annotate_cols(c2=mt2.index_cols(mt.s).c2)

        mt = mt.annotate_cols(c2=mt2.index_cols(hl.str(mt.s)).c2)

        rt = mt.rows()
        ct = mt.cols()

        mt.annotate_rows(**rt[mt.locus, mt.alleles])

        self.assertTrue(rt.all(rt.y2 == 2))
        self.assertTrue(ct.all(ct.c2 == 2))

    def test_joins_with_key_structs(self):
        mt = self.get_mt()

        rows = mt.rows()
        cols = mt.cols()

        self.assertEqual(rows[mt.locus, mt.alleles].take(1), rows[mt.row_key].take(1))
        self.assertEqual(cols[mt.s].take(1), cols[mt.col_key].take(1))

        self.assertEqual(mt.index_rows(mt.row_key).take(1), mt.index_rows(mt.locus, mt.alleles).take(1))
        self.assertEqual(mt.index_cols(mt.col_key).take(1), mt.index_cols(mt.s).take(1))
        self.assertEqual(mt[mt.row_key, mt.col_key].take(1), mt[(mt.locus, mt.alleles), mt.s].take(1))

    def test_index_keyless(self):
        mt = hl.utils.range_matrix_table(3, 3)
        with self.assertRaisesRegex(hl.expr.ExpressionException, "MatrixTable row key: *<<<empty key>>>"):
            mt.key_rows_by().index_rows(mt.row_idx)
        with self.assertRaisesRegex(hl.expr.ExpressionException, "MatrixTable col key: *<<<empty key>>>"):
            mt.key_cols_by().index_cols(mt.col_idx)

    def test_table_join(self):
        ds = self.get_mt()
        # test different row schemas
        self.assertTrue(ds.union_cols(ds.drop(ds.info)).count_rows(), 346)

    def test_table_product_join(self):
        left = hl.utils.range_matrix_table(5, 1)
        right = hl.utils.range_table(5)
        right = right.annotate(i=hl.range(right.idx + 1, 5)).explode('i').key_by('i')
        left = left.annotate_rows(matches=right.index(left.row_key, all_matches=True))
        rows = left.rows()
        self.assertTrue(rows.all(rows.matches.map(lambda x: x.idx) == hl.range(0, rows.row_idx)))

    @qobtest
    def test_naive_coalesce(self):
        mt = self.get_mt(min_partitions=8)
        self.assertEqual(mt.n_partitions(), 8)
        repart = mt.naive_coalesce(2)
        self.assertTrue(mt._same(repart))

    def test_coalesce_with_no_rows(self):
        mt = self.get_mt().filter_rows(False)
        self.assertEqual(mt.repartition(1).count_rows(), 0)

    def test_literals_rebuild(self):
        mt = hl.utils.range_matrix_table(1, 1)
        mt = mt.annotate_rows(
            x=hl.if_else(hl.literal([1, 2, 3])[mt.row_idx] < hl.rand_unif(10, 11), mt.globals, hl.struct())
        )
        mt._force_count_rows()

    def test_globals_lowering(self):
        mt = hl.utils.range_matrix_table(1, 1).annotate_globals(x=1)
        lit = hl.literal(hl.utils.Struct(x=0))

        mt.annotate_rows(foo=hl.agg.collect(mt.globals == lit))._force_count_rows()
        mt.annotate_cols(foo=hl.agg.collect(mt.globals == lit))._force_count_rows()
        mt.filter_rows(mt.globals == lit)._force_count_rows()
        mt.filter_cols(mt.globals == lit)._force_count_rows()
        mt.filter_entries(mt.globals == lit)._force_count_rows()
        (
            mt.group_rows_by(mt.row_idx)
            .aggregate_rows(foo=hl.agg.collect(mt.globals == lit))
            .aggregate(bar=hl.agg.collect(mt.globals == lit))
            ._force_count_rows()
        )
        (
            mt.group_cols_by(mt.col_idx)
            .aggregate_cols(foo=hl.agg.collect(mt.globals == lit))
            .aggregate(bar=hl.agg.collect(mt.globals == lit))
            ._force_count_rows()
        )

    def test_unions_1(self):
        dataset = hl.import_vcf(resource('sample2.vcf'))

        ds1 = dataset.filter_rows(dataset.locus.position % 2 == 1)
        ds2 = dataset.filter_rows(dataset.locus.position % 2 == 0)

        datasets = [ds1, ds2]
        r1 = ds1.union_rows(ds2)
        r2 = hl.MatrixTable.union_rows(*datasets)

        self.assertTrue(r1._same(r2))

        with self.assertRaises(ValueError):
            ds1.filter_cols(ds1.s.endswith('5')).union_rows(ds2)

    def test_unions_2(self):
        dataset = hl.import_vcf(resource('sample2.vcf'))

        ds = dataset.union_cols(dataset).union_cols(dataset)
        for s, count in ds.aggregate_cols(agg.counter(ds.s)).items():
            self.assertEqual(count, 3)

    def test_union_cols_example(self):
        joined = hl.import_vcf(resource('joined.vcf'))

        left = hl.import_vcf(resource('joinleft.vcf'))
        right = hl.import_vcf(resource('joinright.vcf'))

        self.assertTrue(left.union_cols(right)._same(joined))

    def test_union_cols_distinct(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.key_rows_by(x=mt.row_idx // 2)
        assert mt.union_cols(mt).count_rows() == 5

    def test_union_cols_no_error_on_duplicate_names(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.annotate_rows(both='hi')
        mt2 = mt.annotate_rows(both=3, right_only='abc')
        mt = mt.annotate_rows(left_only='123')
        mt = mt.union_cols(mt2, drop_right_row_fields=False)
        assert 'both' in mt.row_value
        assert 'left_only' in mt.row_value
        assert 'right_only' in mt.row_value
        assert len(mt.row_value) == 4

    def test_union_cols_outer(self):
        r, c = 10, 10
        mt = hl.utils.range_matrix_table(2 * r, c)
        mt = mt.annotate_entries(entry=hl.tuple([mt.row_idx, mt.col_idx]))
        mt = mt.annotate_rows(left=mt.row_idx)
        mt2 = hl.utils.range_matrix_table(2 * r, c)
        mt2 = mt2.key_rows_by(row_idx=mt2.row_idx + r)
        mt2 = mt2.key_cols_by(col_idx=mt2.col_idx + c)
        mt2 = mt2.annotate_entries(entry=hl.tuple([mt2.row_idx, mt2.col_idx]))
        mt2 = mt2.annotate_rows(right=mt2.row_idx)
        expected = hl.utils.range_matrix_table(3 * r, 2 * c)
        missing = hl.missing(hl.ttuple(hl.tint, hl.tint))
        expected = expected.annotate_entries(
            entry=hl.if_else(
                expected.col_idx < c,
                hl.if_else(expected.row_idx < 2 * r, hl.tuple([expected.row_idx, expected.col_idx]), missing),
                hl.if_else(expected.row_idx >= r, hl.tuple([expected.row_idx, expected.col_idx]), missing),
            )
        )
        expected = expected.annotate_rows(
            left=hl.if_else(expected.row_idx < 2 * r, expected.row_idx, hl.missing(hl.tint)),
            right=hl.if_else(expected.row_idx >= r, expected.row_idx, hl.missing(hl.tint)),
        )
        assert mt.union_cols(mt2, row_join_type='outer', drop_right_row_fields=False)._same(expected)

    def test_union_rows_different_col_schema(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt2 = hl.utils.range_matrix_table(10, 10)

        mt2 = mt2.annotate_cols(x=mt2.col_idx + 1)
        mt2 = mt2.annotate_globals(g="foo")

        self.assertEqual(mt.union_rows(mt2).count_rows(), 20)

    def test_index(self):
        ds = self.get_mt(min_partitions=8)
        self.assertEqual(ds.n_partitions(), 8)
        ds = ds.add_row_index('rowidx').add_col_index('colidx')

        for i, struct in enumerate(ds.cols().select('colidx').collect()):
            self.assertEqual(i, struct.colidx)
        for i, struct in enumerate(ds.rows().select('rowidx').collect()):
            self.assertEqual(i, struct.rowidx)

    def test_choose_cols(self):
        ds = self.get_mt()
        indices = list(range(ds.count_cols()))
        random.shuffle(indices)

        old_order = ds.key_cols_by()['s'].collect()
        self.assertEqual(ds.choose_cols(indices).key_cols_by()['s'].collect(), [old_order[i] for i in indices])

        self.assertEqual(ds.choose_cols(list(range(10))).s.collect(), old_order[:10])

    def test_choose_cols_vs_explode(self):
        ds = self.get_mt()

        ds2 = ds.annotate_cols(foo=[0, 0]).explode_cols('foo').drop('foo')

        self.assertTrue(ds.choose_cols(sorted(list(range(ds.count_cols())) * 2))._same(ds2))

    def test_distinct_by_row(self):
        orig_mt = hl.utils.range_matrix_table(10, 10)
        mt = orig_mt.key_rows_by(row_idx=orig_mt.row_idx // 2)
        self.assertTrue(mt.distinct_by_row().count_rows() == 5)

        self.assertTrue(orig_mt.union_rows(orig_mt).distinct_by_row()._same(orig_mt))

    def test_distinct_by_col(self):
        orig_mt = hl.utils.range_matrix_table(10, 10)
        mt = orig_mt.key_cols_by(col_idx=orig_mt.col_idx // 2)
        self.assertTrue(mt.distinct_by_col().count_cols() == 5)

        self.assertTrue(orig_mt.union_cols(orig_mt).distinct_by_col()._same(orig_mt))

    def test_aggregation_with_no_aggregators(self):
        mt = hl.utils.range_matrix_table(3, 3)
        self.assertEqual(mt.group_rows_by(mt.row_idx).aggregate().count_rows(), 3)
        self.assertEqual(mt.group_cols_by(mt.col_idx).aggregate().count_cols(), 3)

    def test_computed_key_join_1(self):
        ds = self.get_mt()
        kt = hl.Table.parallelize(
            [{'key': 0, 'value': True}, {'key': 1, 'value': False}],
            hl.tstruct(key=hl.tint32, value=hl.tbool),
            key=['key'],
        )
        ds = ds.annotate_rows(key=ds.locus.position % 2)
        ds = ds.annotate_rows(value=kt[ds['key']]['value'])
        rt = ds.rows()
        self.assertTrue(rt.all(((rt.locus.position % 2) == 0) == rt['value']))

    def test_computed_key_join_multiple_keys(self):
        ds = self.get_mt()
        kt = hl.Table.parallelize(
            [
                {'key1': 0, 'key2': 0, 'value': 0},
                {'key1': 1, 'key2': 0, 'value': 1},
                {'key1': 0, 'key2': 1, 'value': -2},
                {'key1': 1, 'key2': 1, 'value': -1},
            ],
            hl.tstruct(key1=hl.tint32, key2=hl.tint32, value=hl.tint32),
            key=['key1', 'key2'],
        )
        ds = ds.annotate_rows(key1=ds.locus.position % 2, key2=ds.info.DP % 2)
        ds = ds.annotate_rows(value=kt[ds.key1, ds.key2]['value'])
        rt = ds.rows()
        self.assertTrue(rt.all((rt.locus.position % 2) - 2 * (rt.info.DP % 2) == rt['value']))

    def test_computed_key_join_duplicate_row_keys(self):
        ds = self.get_mt()
        kt = hl.Table.parallelize(
            [{'culprit': 'InbreedingCoeff', 'foo': 'bar', 'value': 'IB'}],
            hl.tstruct(culprit=hl.tstr, foo=hl.tstr, value=hl.tstr),
            key=['culprit', 'foo'],
        )
        ds = ds.annotate_rows(dsfoo='bar', info=ds.info.annotate(culprit=[ds.info.culprit, "foo"]))
        ds = ds.explode_rows(ds.info.culprit)
        ds = ds.annotate_rows(value=kt[ds.info.culprit, ds.dsfoo]['value'])
        rt = ds.rows()
        self.assertTrue(
            rt.all(hl.if_else(rt.info.culprit == "InbreedingCoeff", rt['value'] == "IB", hl.is_missing(rt['value'])))
        )

    def test_interval_join(self):
        left = hl.utils.range_matrix_table(50, 1, n_partitions=10)
        intervals = hl.utils.range_table(4)
        intervals = intervals.key_by(interval=hl.interval(intervals.idx * 10, intervals.idx * 10 + 5))
        left = left.annotate_rows(interval_matches=intervals.index(left.row_key))
        rows = left.rows()
        self.assertTrue(
            rows.all(
                hl.case()
                .when(rows.row_idx % 10 < 5, rows.interval_matches.idx == rows.row_idx // 10)
                .default(hl.is_missing(rows.interval_matches))
            )
        )

    def test_interval_product_join(self):
        left = hl.utils.range_matrix_table(50, 1, n_partitions=8)
        intervals = hl.utils.range_table(25)
        intervals = intervals.key_by(
            interval=hl.interval(
                1 + (intervals.idx // 5) * 10 + (intervals.idx % 5), (1 + intervals.idx // 5) * 10 - (intervals.idx % 5)
            )
        )
        intervals = intervals.annotate(i=intervals.idx % 5)
        left = left.annotate_rows(interval_matches=intervals.index(left.row_key, all_matches=True))
        rows = left.rows()
        self.assertTrue(
            rows.all(
                hl.sorted(rows.interval_matches.map(lambda x: x.i))
                == hl.range(0, hl.min(rows.row_idx % 10, 10 - rows.row_idx % 10))
            )
        )

    def test_entry_join_self(self):
        mt1 = hl.utils.range_matrix_table(10, 10, n_partitions=4).choose_cols([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        mt1 = mt1.annotate_entries(x=10 * mt1.row_idx + mt1.col_idx)

        self.assertEqual(mt1[mt1.row_idx, mt1.col_idx].dtype, mt1.entry.dtype)

        mt_join = mt1.annotate_entries(x2=mt1[mt1.row_idx, mt1.col_idx].x)
        mt_join_entries = mt_join.entries()

        self.assertTrue(mt_join_entries.all(mt_join_entries.x == mt_join_entries.x2))

    def test_entry_join_const(self):
        mt1 = hl.utils.range_matrix_table(10, 10, n_partitions=4)
        mt1 = mt1.annotate_entries(x=mt1.row_idx + mt1.col_idx)

        mt2 = hl.utils.range_matrix_table(1, 1, n_partitions=1)
        mt2 = mt2.annotate_entries(foo=10101)

        mt_join = mt1.annotate_entries(**mt2[mt1.row_idx // 100, mt1.col_idx // 100])
        mt_join_entries = mt_join.entries()
        self.assertTrue(mt_join_entries.all(mt_join_entries['foo'] == 10101))

    def test_entry_join_missingness(self):
        mt1 = hl.utils.range_matrix_table(10, 10, n_partitions=4)
        mt1 = mt1.annotate_entries(x=mt1.row_idx + mt1.col_idx)

        mt2 = mt1.filter_cols(mt1.col_idx % 2 == 0)
        mt2 = mt2.filter_rows(mt2.row_idx % 2 == 0)
        mt_join = mt1.annotate_entries(x2=mt2[mt1.row_idx, mt1.col_idx].x * 10)
        mt_join_entries = mt_join.entries()

        kept = mt_join_entries.filter((mt_join_entries.row_idx % 2 == 0) & (mt_join_entries.col_idx % 2 == 0))
        removed = mt_join_entries.filter(~((mt_join_entries.row_idx % 2 == 0) & (mt_join_entries.col_idx % 2 == 0)))

        self.assertTrue(kept.all(hl.is_defined(kept.x2) & (kept.x2 == kept.x * 10)))
        self.assertTrue(removed.all(hl.is_missing(removed.x2)))

    def test_entries_table_length_and_fields(self):
        mt = hl.utils.range_matrix_table(10, 10, n_partitions=4)
        mt = mt.annotate_entries(x=mt.col_idx + mt.row_idx)
        et = mt.entries()
        self.assertEqual(et.count(), 100)
        self.assertTrue(et.all(et.x == et.col_idx + et.row_idx))

    def test_entries_table_no_keys(self):
        mt = hl.utils.range_matrix_table(2, 2)
        mt = mt.annotate_entries(x=(mt.row_idx, mt.col_idx))

        original_order = [
            hl.utils.Struct(row_idx=0, col_idx=0, x=(0, 0)),
            hl.utils.Struct(row_idx=0, col_idx=1, x=(0, 1)),
            hl.utils.Struct(row_idx=1, col_idx=0, x=(1, 0)),
            hl.utils.Struct(row_idx=1, col_idx=1, x=(1, 1)),
        ]

        assert mt.entries().collect() == original_order
        assert mt.key_cols_by().entries().collect() == original_order
        assert mt.key_rows_by().key_cols_by().entries().collect() == original_order
        assert mt.key_rows_by().entries().collect() == sorted(original_order, key=lambda x: x.col_idx)

    def test_entries_table_without_of_order_row_key_fields(self):
        mt = hl.utils.range_matrix_table(10, 10, 1)
        mt = mt.select_rows(key2=0, key1=mt.row_idx)
        mt = mt.key_rows_by(mt.key1, mt.key2)
        mt.entries()._force_count()

    def test_filter_cols_required_entries(self):
        mt1 = hl.utils.range_matrix_table(10, 10, n_partitions=4)
        mt1 = mt1.filter_cols(mt1.col_idx < 3)
        self.assertEqual(len(mt1.entries().collect()), 30)

    def test_filter_cols_with_global_references(self):
        mt = hl.utils.range_matrix_table(10, 10)
        s = hl.literal({1, 3, 5, 7})
        self.assertEqual(mt.filter_cols(s.contains(mt.col_idx)).count_cols(), 4)

    def test_filter_cols_agg(self):
        mt = hl.utils.range_matrix_table(10, 10)
        assert mt.filter_cols(hl.agg.count() > 5).count_cols() == 10

    def test_vcf_regression(self):
        ds = hl.import_vcf(resource('33alleles.vcf'))
        self.assertEqual(ds.filter_rows(ds.alleles.length() == 2).count_rows(), 0)

    def test_field_groups(self):
        ds = self.get_mt()

        df = ds.annotate_rows(row_struct=ds.row).rows()
        self.assertTrue(df.all((df.info == df.row_struct.info) & (df.qual == df.row_struct.qual)))

        ds2 = ds.add_col_index()
        df = ds2.annotate_cols(col_struct=ds2.col).cols()
        self.assertTrue(df.all((df.col_idx == df.col_struct.col_idx)))

        df = ds.annotate_entries(entry_struct=ds.entry).entries()
        self.assertTrue(
            df.all(((hl.is_missing(df.GT) | (df.GT == df.entry_struct.GT)) & (df.AD == df.entry_struct.AD)))
        )

    @test_timeout(batch=5 * 60)
    def test_filter_partitions(self):
        ds = self.get_mt(min_partitions=8)
        self.assertEqual(ds.n_partitions(), 8)
        self.assertEqual(ds._filter_partitions([0, 1, 4]).n_partitions(), 3)
        self.assertEqual(ds._filter_partitions(range(3)).n_partitions(), 3)
        self.assertEqual(ds._filter_partitions([4, 5, 7], keep=False).n_partitions(), 5)
        self.assertTrue(
            ds._same(
                hl.MatrixTable.union_rows(
                    ds._filter_partitions([0, 3, 7]), ds._filter_partitions([0, 3, 7], keep=False)
                )
            )
        )

    def test_from_rows_table(self):
        mt = hl.import_vcf(resource('sample.vcf'))
        mt = mt.annotate_globals(foo='bar')
        rt = mt.rows()
        rm = hl.MatrixTable.from_rows_table(rt)
        self.assertTrue(rm._same(mt.filter_cols(False).select_entries().key_cols_by().select_cols()))

    def test_sample_rows(self):
        ds = self.get_mt()
        ds_small = ds.sample_rows(0.01)
        self.assertTrue(ds_small.count_rows() < ds.count_rows())

    def test_read_stored_cols(self):
        ds = self.get_mt()
        ds = ds.annotate_globals(x='foo')
        f = new_temp_file(extension='mt')
        ds.write(f)
        t = hl.read_table(f + '/cols')
        self.assertTrue(ds.cols().key_by()._same(t))

    def test_read_stored_rows(self):
        ds = self.get_mt()
        ds = ds.annotate_globals(x='foo')
        f = new_temp_file(extension='mt')
        ds.write(f)
        t = hl.read_table(f + '/rows')
        self.assertTrue(ds.rows()._same(t))

    def test_read_stored_globals(self):
        ds = self.get_mt()
        ds = ds.annotate_globals(x=5, baz='foo')
        f = new_temp_file(extension='mt')
        ds.write(f)
        t = hl.read_table(f + '/globals')
        self.assertTrue(ds.globals_table()._same(t))

    def test_indexed_read(self):
        mt = hl.utils.range_matrix_table(2000, 100, 10)
        f = new_temp_file(extension='mt')
        mt.write(f)
        mt1 = hl.read_matrix_table(f)
        mt2 = hl.read_matrix_table(
            f,
            _intervals=[
                hl.Interval(start=150, end=250, includes_start=True, includes_end=False),
                hl.Interval(start=250, end=500, includes_start=True, includes_end=False),
            ],
        )
        self.assertEqual(mt2.n_partitions(), 2)
        self.assertTrue(mt1.filter_rows((mt1.row_idx >= 150) & (mt1.row_idx < 500))._same(mt2))

        mt2 = hl.read_matrix_table(
            f,
            _intervals=[
                hl.Interval(start=150, end=250, includes_start=True, includes_end=False),
                hl.Interval(start=250, end=500, includes_start=True, includes_end=False),
            ],
            _filter_intervals=True,
        )
        self.assertEqual(mt2.n_partitions(), 3)
        self.assertTrue(mt1.filter_rows((mt1.row_idx >= 150) & (mt1.row_idx < 500))._same(mt2))

    def test_indexed_read_vcf(self):
        vcf = self.get_mt(10)
        f = new_temp_file(extension='mt')
        vcf.write(f)
        l1, l2, l3, l4 = (
            hl.Locus('20', 10000000),
            hl.Locus('20', 11000000),
            hl.Locus('20', 13000000),
            hl.Locus('20', 14000000),
        )
        mt = hl.read_matrix_table(
            f,
            _intervals=[
                hl.Interval(start=l1, end=l2),
                hl.Interval(start=l3, end=l4),
            ],
        )
        self.assertEqual(mt.n_partitions(), 2)
        p = (vcf.locus >= l1) & (vcf.locus < l2)
        q = (vcf.locus >= l3) & (vcf.locus < l4)
        self.assertTrue(vcf.filter_rows(p | q)._same(mt))

    def test_interval_filter_partitions(self):
        mt = hl.utils.range_matrix_table(100, 3, 3)
        path = new_temp_file()
        mt.write(path)
        intervals = [
            hl.Interval(hl.Struct(idx=5), hl.Struct(idx=10)),
            hl.Interval(hl.Struct(idx=12), hl.Struct(idx=13)),
            hl.Interval(hl.Struct(idx=15), hl.Struct(idx=17)),
            hl.Interval(hl.Struct(idx=19), hl.Struct(idx=20)),
        ]
        assert hl.read_matrix_table(path, _intervals=intervals, _filter_intervals=True).n_partitions() == 1

        intervals = [
            hl.Interval(hl.Struct(idx=5), hl.Struct(idx=10)),
            hl.Interval(hl.Struct(idx=12), hl.Struct(idx=13)),
            hl.Interval(hl.Struct(idx=15), hl.Struct(idx=17)),
            hl.Interval(hl.Struct(idx=45), hl.Struct(idx=50)),
            hl.Interval(hl.Struct(idx=52), hl.Struct(idx=53)),
            hl.Interval(hl.Struct(idx=55), hl.Struct(idx=57)),
            hl.Interval(hl.Struct(idx=75), hl.Struct(idx=80)),
            hl.Interval(hl.Struct(idx=82), hl.Struct(idx=83)),
            hl.Interval(hl.Struct(idx=85), hl.Struct(idx=87)),
        ]

        assert hl.read_matrix_table(path, _intervals=intervals, _filter_intervals=True).n_partitions() == 3

    @fails_service_backend()
    @test_timeout(3 * 60, local=6 * 60)
    def test_codecs_matrix(self):
        from hail.utils.java import scala_object

        supported_codecs = scala_object(Env.hail().io, 'BufferSpec').specs()
        ds = self.get_mt()
        temp = new_temp_file(extension='mt')
        for codec in supported_codecs:
            ds.write(temp, overwrite=True, _codec_spec=codec.toString())
            ds2 = hl.read_matrix_table(temp)
            self.assertTrue(ds._same(ds2))

    @fails_service_backend()
    @test_timeout(local=6 * 60)
    def test_codecs_table(self):
        from hail.utils.java import scala_object

        supported_codecs = scala_object(Env.hail().io, 'BufferSpec').specs()
        rt = self.get_mt().rows()
        temp = new_temp_file(extension='ht')
        for codec in supported_codecs:
            rt.write(temp, overwrite=True, _codec_spec=codec.toString())
            rt2 = hl.read_table(temp)
            self.assertTrue(rt._same(rt2))

    def test_fix3307_read_mt_wrong(self):
        mt = hl.import_vcf(resource('sample2.vcf'))
        mt = hl.split_multi_hts(mt)
        with hl.TemporaryDirectory(suffix='.mt', ensure_exists=False) as mt_path:
            mt.write(mt_path)
            mt2 = hl.read_matrix_table(mt_path)
            t = hl.read_table(mt_path + '/rows')
            self.assertTrue(mt.rows()._same(t))
            self.assertTrue(mt2.rows()._same(t))
            self.assertTrue(mt._same(mt2))

    def test_rename(self):
        dataset = self.get_mt()
        renamed1 = dataset.rename({'locus': 'locus2', 'info': 'info2', 's': 'info'})

        self.assertEqual(renamed1['locus2']._type, dataset['locus']._type)
        self.assertEqual(renamed1['info2']._type, dataset['info']._type)
        self.assertEqual(renamed1['info']._type, dataset['s']._type)

        self.assertEqual(renamed1['info']._indices, renamed1._col_indices)

        self.assertFalse('locus' in renamed1._fields)
        self.assertFalse('s' in renamed1._fields)

        with self.assertRaises(ValueError):
            dataset.rename({'locus': 'info'})

        with self.assertRaises(ValueError):
            dataset.rename({'locus': 'a', 's': 'a'})

        with self.assertRaises(LookupError):
            dataset.rename({'foo': 'a'})

    def test_range(self):
        ds = hl.utils.range_matrix_table(100, 10)
        self.assertEqual(ds.count_rows(), 100)
        self.assertEqual(ds.count_cols(), 10)
        et = ds.annotate_entries(entry_idx=10 * ds.row_idx + ds.col_idx).entries().add_index()
        self.assertTrue(et.all(et.idx == et.entry_idx))

    def test_filter_entries(self):
        ds = hl.utils.range_matrix_table(100, 10)
        ds = ds.annotate_rows(foo=5)  # triggered a RV bug
        ds = ds.annotate_cols(bar=5)
        ds = ds.filter_entries((ds.col_idx * ds.row_idx) % 4 == 0)

        entries = ds.entries()
        self.assertTrue(entries.all((entries.col_idx * entries.row_idx) % 4 == 0))

    def test_filter_na(self):
        mt = hl.utils.range_matrix_table(1, 1)

        self.assertEqual(mt.filter_rows(hl.missing(hl.tbool)).count_rows(), 0)
        self.assertEqual(mt.filter_cols(hl.missing(hl.tbool)).count_cols(), 0)
        self.assertEqual(mt.filter_entries(hl.missing(hl.tbool)).entries().count(), 0)

    def get_example_mt_for_to_table_on_various_fields(self):
        mt = hl.utils.range_matrix_table(3, 4)

        globe = 'the globe!'
        sample_ids = ['Bob', 'Alice', 'David', 'Carol']
        entries = [1, 0, 3, 2]
        rows = ['1:3:A:G', '1:2:A:G', '1:0:A:G']
        sorted_rows = sorted(rows)

        mt = mt.annotate_globals(globe=globe)
        mt = mt.annotate_cols(s=hl.array(sample_ids)[mt.col_idx]).key_cols_by('s')
        mt = mt.annotate_entries(e=hl.array(entries)[mt.col_idx])
        mt = mt.annotate_rows(r=hl.array(rows)[mt.row_idx]).key_rows_by('r')
        return mt, globe, sample_ids, entries, rows, sorted_rows

    def test_to_table_on_global_and_sample_fields(self):
        mt, globe, sample_ids, _, _, _ = self.get_example_mt_for_to_table_on_various_fields()

        self.assertEqual(mt.globe.collect(), [globe])

        self.assertEqual(mt.s.collect(), sample_ids)
        self.assertEqual((mt.s + '1').collect(), [s + '1' for s in sample_ids])
        self.assertEqual(('1' + mt.s).collect(), ['1' + s for s in sample_ids])
        self.assertEqual(mt.s.take(1), [sample_ids[0]])

    def test_to_table_on_entry_fields(self):
        mt, _, _, entries, _, _ = self.get_example_mt_for_to_table_on_various_fields()
        self.assertEqual(mt.e.collect(), entries * 3)
        self.assertEqual(mt.e.take(1), [entries[0]])

    def test_to_table_on_row_fields(self):
        mt, _, _, _, _, sorted_rows = self.get_example_mt_for_to_table_on_various_fields()
        self.assertEqual(mt.row_idx.collect(), [2, 1, 0])
        self.assertEqual(mt.r.collect(), sorted_rows)
        self.assertEqual(mt.r.take(1), [sorted_rows[0]])

    def test_to_table_on_col_and_col_key(self):
        mt, _, sample_ids, _, _, _ = self.get_example_mt_for_to_table_on_various_fields()
        self.assertEqual(mt.col_key.collect(), [hl.Struct(s=s) for s in sample_ids])
        self.assertEqual(mt.col.collect(), [hl.Struct(s=s, col_idx=i) for i, s in enumerate(sample_ids)])

    def test_to_table_on_row_and_row_key(self):
        mt, _, _, _, rows, sorted_rows = self.get_example_mt_for_to_table_on_various_fields()
        self.assertEqual(mt.row_key.collect(), [hl.Struct(r=r) for r in sorted_rows])
        self.assertEqual(
            mt.row.collect(), sorted([hl.Struct(r=r, row_idx=i) for i, r in enumerate(rows)], key=lambda x: x.r)
        )

    def test_to_table_on_entry(self):
        mt, _, _, entries, _, sorted_rows = self.get_example_mt_for_to_table_on_various_fields()
        self.assertEqual(mt.entry.collect(), [hl.Struct(e=e) for _ in sorted_rows for e in entries])

    def test_to_table_on_cols_method(self):
        mt, _, sample_ids, _, _, _ = self.get_example_mt_for_to_table_on_various_fields()
        self.assertEqual(mt.cols().s.collect(), sorted(sample_ids))
        self.assertEqual(mt.cols().s.take(1), [sorted(sample_ids)[0]])

    def test_to_table_on_entries_method(self):
        mt, _, _, entries, _, _ = self.get_example_mt_for_to_table_on_various_fields()
        self.assertEqual(mt.entries().e.collect(), sorted(entries) * 3)
        self.assertEqual(mt.entries().e.take(1), [sorted(entries)[0]])

    def test_to_table_on_rows_method(self):
        mt, _, _, _, _, sorted_rows = self.get_example_mt_for_to_table_on_various_fields()
        self.assertEqual(mt.rows().row_idx.collect(), [2, 1, 0])
        self.assertEqual(mt.rows().r.collect(), sorted_rows)
        self.assertEqual(mt.rows().r.take(1), [sorted_rows[0]])

    def test_order_by(self):
        ht = hl.utils.range_table(10)
        self.assertEqual(ht.order_by('idx').idx.collect(), list(range(10)))
        self.assertEqual(ht.order_by(hl.asc('idx')).idx.collect(), list(range(10)))
        self.assertEqual(ht.order_by(hl.desc('idx')).idx.collect(), list(range(10))[::-1])

    def test_order_by_complex_exprs(self):
        ht = hl.utils.range_table(10)
        assert ht.order_by(-ht.idx).idx.collect() == list(range(10))[::-1]

    def test_order_by_intervals(self):
        intervals = {
            0: hl.Interval(0, 3, includes_start=True, includes_end=False),
            1: hl.Interval(0, 4, includes_start=True, includes_end=True),
            2: hl.Interval(1, 4, includes_start=True, includes_end=False),
            3: hl.Interval(0, 4, includes_start=False, includes_end=False),
            4: hl.Interval(0, 4, includes_start=True, includes_end=False),
        }
        ht = hl.utils.range_table(5)

        ht = ht.annotate_globals(ilist=intervals)
        ht = ht.annotate(interval=ht['ilist'][ht['idx']])
        ht = ht.order_by(ht['interval'])

        ordered = ht['interval'].collect()
        expected = [intervals[i] for i in [0, 4, 1, 3, 2]]

        self.assertEqual(ordered, expected)

    def test_range_matrix_table(self):
        mt = hl.utils.range_matrix_table(13, 7, n_partitions=5)
        self.assertEqual(mt.globals.dtype, hl.tstruct())
        self.assertEqual(mt.row.dtype, hl.tstruct(row_idx=hl.tint32))
        self.assertEqual(mt.col.dtype, hl.tstruct(col_idx=hl.tint32))
        self.assertEqual(mt.entry.dtype, hl.tstruct())

        self.assertEqual(list(mt.row_key), ['row_idx'])
        self.assertEqual(list(mt.col_key), ['col_idx'])

        self.assertEqual([r.row_idx for r in mt.rows().collect()], list(range(13)))
        self.assertEqual([r.col_idx for r in mt.cols().collect()], list(range(7)))

    def test_range_matrix_table_0_rows_0_cols(self):
        mt = hl.utils.range_matrix_table(0, 0)
        self.assertEqual(mt.col_idx.collect(), [])
        self.assertEqual(mt.row_idx.collect(), [])
        mt = mt.annotate_entries(x=mt.row_idx * mt.col_idx)
        self.assertEqual(mt.x.collect(), [])

    def test_make_table(self):
        mt = hl.utils.range_matrix_table(3, 2)
        mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
        mt = mt.key_cols_by(col_idx=hl.str(mt.col_idx))

        t = hl.Table.parallelize(
            [
                {'row_idx': 0, '0.x': 0, '1.x': 0},
                {'row_idx': 1, '0.x': 0, '1.x': 1},
                {'row_idx': 2, '0.x': 0, '1.x': 2},
            ],
            hl.tstruct(**{'row_idx': hl.tint32, '0.x': hl.tint32, '1.x': hl.tint32}),
            key='row_idx',
        )

        self.assertTrue(mt.make_table()._same(t))

    def test_make_table_empty_entry_field(self):
        mt = hl.utils.range_matrix_table(3, 2)
        mt = mt.select_entries(**{'': mt.row_idx * mt.col_idx})
        mt = mt.key_cols_by(col_idx=hl.str(mt.col_idx))

        t = mt.make_table()
        self.assertEqual(t.row.dtype, hl.tstruct(**{'row_idx': hl.tint32, '0': hl.tint32, '1': hl.tint32}))

    def test_make_table_sep(self):
        mt = hl.utils.range_matrix_table(3, 2)
        mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
        mt = mt.key_cols_by(col_idx=hl.str(mt.col_idx))

        t = mt.make_table()
        assert list(t.row) == ['row_idx', '0.x', '1.x']

        t = mt.make_table(separator='__')
        assert list(t.row) == ['row_idx', '0__x', '1__x']

    def test_make_table_row_equivalence(self):
        mt = hl.utils.range_matrix_table(3, 3)
        mt = mt.annotate_rows(r1=hl.rand_norm(), r2=hl.rand_norm())
        mt = mt.annotate_entries(e1=hl.rand_norm(), e2=hl.rand_norm())
        mt = mt.key_cols_by(col_idx=hl.str(mt.col_idx))

        assert mt.make_table().select(*mt.row_value)._same(mt.rows())

    def test_make_table_na_error(self):
        mt = hl.utils.range_matrix_table(3, 3).key_cols_by(s=hl.missing('str'))
        mt = mt.annotate_entries(e1=1)
        with pytest.raises(ValueError):
            mt.make_table()

    def test_transmute(self):
        mt = (
            hl.utils.range_matrix_table(1, 1)
            .annotate_globals(g1=0, g2=0)
            .annotate_cols(c1=0, c2=0)
            .annotate_rows(r1=0, r2=0)
            .annotate_entries(e1=0, e2=0)
        )
        self.assertEqual(mt.transmute_globals(g3=mt.g2 + 1).globals.dtype, hl.tstruct(g1=hl.tint, g3=hl.tint))
        self.assertEqual(mt.transmute_rows(r3=mt.r2 + 1).row_value.dtype, hl.tstruct(r1=hl.tint, r3=hl.tint))
        self.assertEqual(mt.transmute_cols(c3=mt.c2 + 1).col_value.dtype, hl.tstruct(c1=hl.tint, c3=hl.tint))
        self.assertEqual(mt.transmute_entries(e3=mt.e2 + 1).entry.dtype, hl.tstruct(e1=hl.tint, e3=hl.tint))

    def test_transmute_agg(self):
        mt = hl.utils.range_matrix_table(1, 1).annotate_entries(x=5)
        mt = mt.transmute_rows(y=hl.agg.mean(mt.x))

    def test_agg_explode(self):
        t = hl.Table.parallelize(
            [
                hl.struct(a=[1, 2]),
                hl.struct(a=hl.empty_array(hl.tint32)),
                hl.struct(a=hl.missing(hl.tarray(hl.tint32))),
                hl.struct(a=[3]),
                hl.struct(a=[hl.missing(hl.tint32)]),
            ]
        )
        self.assertCountEqual(t.aggregate(hl.agg.explode(lambda elt: hl.agg.collect(elt), t.a)), [1, 2, None, 3])

    def test_agg_call_stats(self):
        t = hl.Table.parallelize(
            [
                hl.struct(c=hl.call(0, 0)),
                hl.struct(c=hl.call(0, 1)),
                hl.struct(c=hl.call(0, 2, phased=True)),
                hl.struct(c=hl.call(1)),
                hl.struct(c=hl.call(0)),
                hl.struct(c=hl.call()),
            ]
        )
        actual = t.aggregate(hl.agg.call_stats(t.c, ['A', 'T', 'G']))
        expected = hl.struct(AC=[5, 2, 1], AF=[5.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0], AN=8, homozygote_count=[1, 0, 0])

        self.assertTrue(hl.Table.parallelize([actual]), hl.Table.parallelize([expected]))

    def test_hardy_weinberg_test(self):
        mt = hl.import_vcf(resource('HWE_test.vcf'))
        mt_two_sided = mt.select_rows(**hl.agg.hardy_weinberg_test(mt.GT, one_sided=False))
        rt_two_sided = mt_two_sided.rows()
        expected_two_sided = hl.Table.parallelize(
            [
                hl.struct(locus=hl.locus('20', pos), alleles=alleles, het_freq_hwe=r, p_value=p)
                for (pos, alleles, r, p) in [
                    (1, ['A', 'G'], 0.0, 0.5),
                    (2, ['A', 'G'], 0.25, 0.5),
                    (3, ['T', 'C'], 0.5357142857142857, 0.21428571428571427),
                    (4, ['T', 'A'], 0.5714285714285714, 0.6571428571428573),
                    (5, ['G', 'A'], 0.3333333333333333, 0.5),
                ]
            ],
            key=['locus', 'alleles'],
        )

        self.assertTrue(rt_two_sided.filter(rt_two_sided.locus.position != 6)._same(expected_two_sided))
        rt6_two_sided = rt_two_sided.filter(rt_two_sided.locus.position == 6).collect()[0]
        self.assertEqual(rt6_two_sided['p_value'], 0.5)
        self.assertTrue(math.isnan(rt6_two_sided['het_freq_hwe']))

        mt_one_sided = mt.select_rows(**hl.agg.hardy_weinberg_test(mt.GT, one_sided=True))
        rt_one_sided = mt_one_sided.rows()
        expected_one_sided = hl.Table.parallelize(
            [
                hl.struct(locus=hl.locus('20', pos), alleles=alleles, het_freq_hwe=r, p_value=p)
                for (pos, alleles, r, p) in [
                    (1, ['A', 'G'], 0.0, 0.5),
                    (2, ['A', 'G'], 0.25, 0.5),
                    (3, ['T', 'C'], 0.5357142857142857, 0.7857142857142857),
                    (4, ['T', 'A'], 0.5714285714285714, 0.5714285714285715),
                    (5, ['G', 'A'], 0.3333333333333333, 0.5),
                ]
            ],
            key=['locus', 'alleles'],
        )

        self.assertTrue(rt_one_sided.filter(rt_one_sided.locus.position != 6)._same(expected_one_sided))
        rt6_one_sided = rt_one_sided.filter(rt_one_sided.locus.position == 6).collect()[0]
        self.assertEqual(rt6_one_sided['p_value'], 0.5)
        self.assertTrue(math.isnan(rt6_one_sided['het_freq_hwe']))

    def test_hw_func_and_agg_agree(self):
        mt = hl.import_vcf(resource('sample.vcf'))
        mt_two_sided = mt.annotate_rows(
            stats=hl.agg.call_stats(mt.GT, mt.alleles), hw=hl.agg.hardy_weinberg_test(mt.GT, one_sided=False)
        )
        mt_two_sided = mt_two_sided.annotate_rows(
            hw2=hl.hardy_weinberg_test(
                mt_two_sided.stats.homozygote_count[0],
                mt_two_sided.stats.AC[1] - 2 * mt_two_sided.stats.homozygote_count[1],
                mt_two_sided.stats.homozygote_count[1],
                one_sided=False,
            )
        )
        rt_two_sided = mt_two_sided.rows()
        self.assertTrue(rt_two_sided.all(rt_two_sided.hw == rt_two_sided.hw2))

        mt_one_sided = mt.annotate_rows(
            stats=hl.agg.call_stats(mt.GT, mt.alleles), hw=hl.agg.hardy_weinberg_test(mt.GT, one_sided=True)
        )
        mt_one_sided = mt_one_sided.annotate_rows(
            hw2=hl.hardy_weinberg_test(
                mt_one_sided.stats.homozygote_count[0],
                mt_one_sided.stats.AC[1] - 2 * mt_one_sided.stats.homozygote_count[1],
                mt_one_sided.stats.homozygote_count[1],
                one_sided=True,
            )
        )
        rt_one_sided = mt_one_sided.rows()
        self.assertTrue(rt_one_sided.all(rt_one_sided.hw == rt_one_sided.hw2))

    def test_write_stage_locally(self):
        mt = self.get_mt()
        f = new_temp_file(extension='mt')
        mt.write(f, stage_locally=True)

        mt2 = hl.read_matrix_table(f)
        self.assertTrue(mt._same(mt2))

    def test_write_no_parts(self):
        mt = hl.utils.range_matrix_table(10, 10, 2).filter_rows(False)
        path = new_temp_file(extension='mt')
        path2 = new_temp_file(extension='mt')
        assert mt.checkpoint(path)._same(mt)
        hl.read_matrix_table(path, _drop_rows=True).write(path2)

    def test_nulls_in_distinct_joins_1(self):
        # MatrixAnnotateRowsTable uses left distinct join
        mr = hl.utils.range_matrix_table(7, 3, 4)
        matrix1 = mr.key_rows_by(
            new_key=hl.if_else((mr.row_idx == 3) | (mr.row_idx == 5), hl.missing(hl.tint32), mr.row_idx)
        )
        matrix2 = mr.key_rows_by(
            new_key=hl.if_else((mr.row_idx == 4) | (mr.row_idx == 6), hl.missing(hl.tint32), mr.row_idx)
        )
        joined = matrix1.select_rows(idx1=matrix1.row_idx, idx2=matrix2.rows()[matrix1.new_key].row_idx)

        def row(new_key, idx1, idx2):
            return hl.Struct(new_key=new_key, idx1=idx1, idx2=idx2)

        expected = [
            row(0, 0, 0),
            row(1, 1, 1),
            row(2, 2, 2),
            row(4, 4, None),
            row(6, 6, None),
            row(None, 3, None),
            row(None, 5, None),
        ]
        self.assertEqual(joined.rows().collect(), expected)

    def test_nulls_in_distinct_joins_2(self):
        mr = hl.utils.range_matrix_table(7, 3, 4)
        matrix1 = mr.key_rows_by(
            new_key=hl.if_else((mr.row_idx == 3) | (mr.row_idx == 5), hl.missing(hl.tint32), mr.row_idx)
        )
        matrix2 = mr.key_rows_by(
            new_key=hl.if_else((mr.row_idx == 4) | (mr.row_idx == 6), hl.missing(hl.tint32), mr.row_idx)
        )
        # union_cols uses inner distinct join
        matrix1 = matrix1.annotate_entries(ridx=matrix1.row_idx, cidx=matrix1.col_idx)
        matrix2 = matrix2.annotate_entries(ridx=matrix2.row_idx, cidx=matrix2.col_idx)
        matrix2 = matrix2.key_cols_by(col_idx=matrix2.col_idx + 3)

        expected = hl.utils.range_matrix_table(3, 6, 1)
        expected = expected.key_rows_by(new_key=expected.row_idx)
        expected = expected.annotate_entries(ridx=expected.row_idx, cidx=expected.col_idx % 3)

        self.assertTrue(matrix1.union_cols(matrix2)._same(expected))

    @test_timeout(local=5 * 60, batch=10 * 60)
    def test_row_joins_into_table_1(self):
        rt = hl.utils.range_matrix_table(9, 13, 3)
        mt1 = rt.key_rows_by(idx=rt.row_idx)
        mt1 = mt1.select_rows(v=mt1.idx + 2)

        t1 = hl.utils.range_table(10, 3)
        t2 = t1.key_by(t1.idx, idx2=t1.idx + 1)
        t1 = t1.select(v=t1.idx + 2)
        t2 = t2.select(v=t2.idx + 2)

        tinterval1 = t1.key_by(k=hl.interval(t1.idx, t1.idx, True, True))
        tinterval1 = tinterval1.select(v=tinterval1.idx + 2)

        values = [hl.Struct(v=i + 2) for i in range(9)]
        # join on mt row key
        self.assertEqual(t1.index(mt1.row_key).collect(), values)
        self.assertEqual(t1.index(mt1.idx).collect(), values)
        with self.assertRaises(hl.expr.ExpressionException):
            t2.index(mt1.row_key).collect()

        # join on not mt row key
        self.assertEqual(t1.index(mt1.v).collect(), [hl.Struct(v=i + 2) for i in range(2, 10)] + [None])

        # join on interval of first field of mt row key
        self.assertEqual(tinterval1.index(mt1.idx).collect(), values)
        self.assertEqual(tinterval1.index(mt1.row_key).collect(), values)

    @test_timeout(local=5 * 60, batch=10 * 60)
    def test_row_joins_into_table_2(self):
        rt = hl.utils.range_matrix_table(9, 13, 3)
        mt2 = rt.key_rows_by(idx=rt.row_idx, idx2=rt.row_idx + 1)
        mt2 = mt2.select_rows(v=mt2.idx + 2)
        t1 = hl.utils.range_table(10, 3)
        t2 = t1.key_by(t1.idx, idx2=t1.idx + 1)
        t1 = t1.select(v=t1.idx + 2)
        t2 = t2.select(v=t2.idx + 2)

        tinterval1 = t1.key_by(k=hl.interval(t1.idx, t1.idx, True, True))
        tinterval1 = tinterval1.select(v=tinterval1.idx + 2)
        tinterval2 = t2.key_by(k=hl.interval(t2.key, t2.key, True, True))
        tinterval2 = tinterval2.select(v=tinterval2.idx + 2)

        values = [hl.Struct(v=i + 2) for i in range(9)]
        # join on mt row key
        self.assertEqual(t2.index(mt2.row_key).collect(), values)
        self.assertEqual(t2.index(mt2.idx, mt2.idx2).collect(), values)
        self.assertEqual(t1.index(mt2.idx).collect(), values)
        with self.assertRaises(hl.expr.ExpressionException):
            t2.index(mt2.idx).collect()

        # join on not mt row key
        self.assertEqual(t2.index(mt2.idx2, mt2.v).collect(), [hl.Struct(v=i + 2) for i in range(1, 10)])
        with self.assertRaises(hl.expr.ExpressionException):
            t2.index(mt2.v).collect()

        # join on interval of first field of mt row key
        self.assertEqual(tinterval1.index(mt2.idx).collect(), values)

        with self.assertRaises(hl.expr.ExpressionException):
            tinterval1.index(mt2.row_key).collect()
        with self.assertRaises(hl.expr.ExpressionException):
            tinterval2.index(mt2.idx).collect()
        with self.assertRaises(hl.expr.ExpressionException):
            tinterval2.index(mt2.row_key).collect()
        with self.assertRaises(hl.expr.ExpressionException):
            tinterval2.index(mt2.idx, mt2.idx2).collect()

    def test_refs_with_process_joins(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.annotate_entries(
            a_literal=hl.literal(['a']),
            a_col_join=hl.is_defined(mt.cols()[mt.col_key]),
            a_row_join=hl.is_defined(mt.rows()[mt.row_key]),
            an_entry_join=hl.is_defined(mt[mt.row_key, mt.col_key]),
            the_global_failure=hl.if_else(True, mt.globals, hl.missing(mt.globals.dtype)),
            the_row_failure=hl.if_else(True, mt.row, hl.missing(mt.row.dtype)),
            the_col_failure=hl.if_else(True, mt.col, hl.missing(mt.col.dtype)),
            the_entry_failure=hl.if_else(True, mt.entry, hl.missing(mt.entry.dtype)),
        )
        mt.count()

    def test_aggregate_localize_false(self):
        dim1, dim2 = 10, 10
        mt = hl.utils.range_matrix_table(dim1, dim2)
        mt = mt.annotate_entries(
            x=mt.aggregate_rows(hl.agg.max(mt.row_idx), _localize=False)
            + mt.aggregate_cols(hl.agg.max(mt.col_idx), _localize=False)
            + mt.aggregate_entries(hl.agg.max(mt.row_idx * mt.col_idx), _localize=False)
        )
        assert mt.x.take(1)[0] == (dim1 - 1) + (dim2 - 1) + (dim1 - 1) * (dim2 - 1)

    def test_agg_cols_filter(self):
        t = hl.utils.range_matrix_table(1, 10)
        tests = [
            (agg.filter(t.col_idx > 7, agg.collect(t.col_idx + 1).append(0)), [9, 10, 0]),
            (
                agg.filter(
                    t.col_idx > 7, agg.explode(lambda elt: agg.collect(elt + 1).append(0), [t.col_idx, t.col_idx + 1])
                ),
                [9, 10, 10, 11, 0],
            ),
            (
                agg.filter(
                    t.col_idx > 7, agg.group_by(t.col_idx % 3, hl.array(agg.collect_as_set(t.col_idx + 1)).append(0))
                ),
                {0: [10, 0], 2: [9, 0]},
            ),
        ]
        for aggregation, expected in tests:
            self.assertEqual(t.select_rows(result=aggregation).result.collect()[0], expected)

    def test_agg_cols_explode(self):
        t = hl.utils.range_matrix_table(1, 10)

        tests = [
            (
                agg.explode(
                    lambda elt: agg.collect(elt + 1).append(0),
                    hl.if_else(t.col_idx > 7, [t.col_idx, t.col_idx + 1], hl.empty_array(hl.tint32)),
                ),
                [9, 10, 10, 11, 0],
            ),
            (
                agg.explode(
                    lambda elt: agg.explode(lambda elt2: agg.collect(elt2 + 1).append(0), [elt, elt + 1]),
                    hl.if_else(t.col_idx > 7, [t.col_idx, t.col_idx + 1], hl.empty_array(hl.tint32)),
                ),
                [9, 10, 10, 11, 10, 11, 11, 12, 0],
            ),
            (
                agg.explode(
                    lambda elt: agg.filter(elt > 8, agg.collect(elt + 1).append(0)),
                    hl.if_else(t.col_idx > 7, [t.col_idx, t.col_idx + 1], hl.empty_array(hl.tint32)),
                ),
                [10, 10, 11, 0],
            ),
            (
                agg.explode(
                    lambda elt: agg.group_by(elt % 3, agg.collect(elt + 1).append(0)),
                    hl.if_else(t.col_idx > 7, [t.col_idx, t.col_idx + 1], hl.empty_array(hl.tint32)),
                ),
                {0: [10, 10, 0], 1: [11, 0], 2: [9, 0]},
            ),
        ]
        for aggregation, expected in tests:
            self.assertEqual(t.select_rows(result=aggregation).result.collect()[0], expected)

    def test_agg_cols_group_by(self):
        t = hl.utils.range_matrix_table(1, 10)
        tests = [
            (
                agg.group_by(t.col_idx % 2, hl.array(agg.collect_as_set(t.col_idx + 1)).append(0)),
                {0: [1, 3, 5, 7, 9, 0], 1: [2, 4, 6, 8, 10, 0]},
            ),
            (
                agg.group_by(
                    t.col_idx % 3, agg.filter(t.col_idx > 7, hl.array(agg.collect_as_set(t.col_idx + 1)).append(0))
                ),
                {0: [10, 0], 1: [0], 2: [9, 0]},
            ),
            (
                agg.group_by(
                    t.col_idx % 3,
                    agg.explode(
                        lambda elt: agg.collect(elt + 1).append(0),
                        hl.if_else(t.col_idx > 7, [t.col_idx, t.col_idx + 1], hl.empty_array(hl.tint32)),
                    ),
                ),
                {0: [10, 11, 0], 1: [0], 2: [9, 10, 0]},
            ),
        ]
        for aggregation, expected in tests:
            self.assertEqual(t.select_rows(result=aggregation).result.collect()[0], expected)

    def test_localize_entries_with_both_none_is_rows_table(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
        localized = mt.localize_entries(entries_array_field_name=None, columns_array_field_name=None)
        rows_table = mt.rows()
        assert rows_table._same(localized)

    def test_localize_entries_with_none_cols_adds_no_globals(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
        localized = mt.localize_entries(entries_array_field_name=Env.get_uid(), columns_array_field_name=None)
        assert hl.eval(mt.globals) == hl.eval(localized.globals)

    def test_localize_entries_with_none_entries_changes_no_rows(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
        localized = mt.localize_entries(entries_array_field_name=None, columns_array_field_name=Env.get_uid())
        rows_table = mt.rows()
        assert rows_table.select_globals()._same(localized.select_globals())

    def test_localize_entries_creates_arrays_of_entries_and_array_of_cols(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
        localized = mt.localize_entries(entries_array_field_name='entries', columns_array_field_name='cols')
        t = hl.utils.range_table(10)
        t = t.select(entries=hl.range(10).map(lambda y: hl.struct(x=t.idx * y)))
        t = t.select_globals(cols=hl.range(10).map(lambda y: hl.struct(col_idx=y)))
        t = t.rename({'idx': 'row_idx'})
        assert localized._same(t)

    def test_multi_write(self):
        mt = self.get_mt()
        f = new_temp_file()
        hl.experimental.write_matrix_tables([mt, mt], f)
        path1 = f + '0.mt'
        path2 = f + '1.mt'
        mt1 = hl.read_matrix_table(path1)
        mt2 = hl.read_matrix_table(path2)
        self.assertTrue(mt._same(mt1))
        self.assertTrue(mt._same(mt2))
        self.assertTrue(mt1._same(mt2))

    def test_matrix_type_equality(self):
        mt = hl.utils.range_matrix_table(1, 1)
        mt2 = mt.annotate_entries(foo=1)
        assert mt._type == mt._type
        assert mt._type != mt2._type

    def test_entry_filtering(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.filter_entries((mt.col_idx + mt.row_idx) % 2 == 0)

        assert mt.aggregate_entries(hl.agg.count()) == 50
        assert all(x == 5 for x in mt.annotate_cols(x=hl.agg.count()).x.collect())
        assert all(x == 5 for x in mt.annotate_rows(x=hl.agg.count()).x.collect())

        mt = mt.unfilter_entries()

        assert mt.aggregate_entries(hl.agg.count()) == 100
        assert all(x == 10 for x in mt.annotate_cols(x=hl.agg.count()).x.collect())
        assert all(x == 10 for x in mt.annotate_rows(x=hl.agg.count()).x.collect())

    def test_entry_filter_stats(self):
        mt = hl.utils.range_matrix_table(40, 20)
        mt = mt.filter_entries((mt.row_idx % 4 == 0) & (mt.col_idx % 4 == 0), keep=False)
        mt = mt.compute_entry_filter_stats()

        row_expected = hl.dict(
            {
                True: hl.struct(n_filtered=5, n_remaining=15, fraction_filtered=hl.float32(0.25)),
                False: hl.struct(n_filtered=0, n_remaining=20, fraction_filtered=hl.float32(0.0)),
            }
        )
        assert mt.aggregate_rows(hl.agg.all(mt.entry_stats_row == row_expected[mt.row_idx % 4 == 0]))

        col_expected = hl.dict(
            {
                True: hl.struct(n_filtered=10, n_remaining=30, fraction_filtered=hl.float32(0.25)),
                False: hl.struct(n_filtered=0, n_remaining=40, fraction_filtered=hl.float32(0.0)),
            }
        )
        assert mt.aggregate_cols(hl.agg.all(mt.entry_stats_col == col_expected[mt.col_idx % 4 == 0]))

    def test_annotate_col_agg_lowering(self):
        mt = hl.utils.range_matrix_table(10, 10, 2)
        mt = mt.annotate_cols(c1=[mt.col_idx, mt.col_idx * 2])
        mt = mt.annotate_entries(e1=mt.col_idx + mt.row_idx, e2=[mt.col_idx * mt.row_idx, mt.col_idx * mt.row_idx**2])
        common_ref = mt.c1[1]
        mt = mt.annotate_cols(
            exploded=hl.agg.explode(lambda e: common_ref + hl.agg.sum(e), mt.e2),
            array=hl.agg.array_agg(lambda e: common_ref + hl.agg.sum(e), mt.e2),
            filt=hl.agg.filter(mt.e1 < 5, hl.agg.sum(mt.e1) + common_ref),
            grouped=hl.agg.group_by(mt.e1 % 5, hl.agg.sum(mt.e1) + common_ref),
        )
        mt.cols()._force_count()

    def test_annotate_rows_scan_lowering(self):
        mt = hl.utils.range_matrix_table(10, 10, 2)
        mt = mt.annotate_rows(r1=[mt.row_idx, mt.row_idx * 2])
        common_ref = mt.r1[1]
        mt = mt.annotate_rows(
            exploded=hl.scan.explode(lambda e: common_ref + hl.scan.sum(e), mt.r1),
            array=hl.scan.array_agg(lambda e: common_ref + hl.scan.sum(e), mt.r1),
            filt=hl.scan.filter(mt.row_idx < 5, hl.scan.sum(mt.row_idx) + common_ref),
            grouped=hl.scan.group_by(mt.row_idx % 5, hl.scan.sum(mt.row_idx) + common_ref),
            an_agg=hl.agg.sum(mt.row_idx * mt.col_idx),
        )
        mt.cols()._force_count()

    def test_show_runs(self):
        mt = self.get_mt()
        mt.show()

    def test_show_header(self):
        mt = hl.utils.range_matrix_table(1, 1)
        mt = mt.annotate_entries(x=1)
        mt = mt.key_cols_by(col_idx=mt.col_idx + 10)

        expected = (
            '+---------+-------+\n'
            '| row_idx |  10.x |\n'
            '+---------+-------+\n'
            '|   int32 | int32 |\n'
            '+---------+-------+\n'
            '|       0 |     1 |\n'
            '+---------+-------+\n'
        )
        actual = mt.show(handler=str)
        assert actual == expected

    @test_timeout(batch=6 * 60)
    def test_partitioned_write(self):
        mt = hl.utils.range_matrix_table(40, 3, 5)

        def test_parts(parts, expected=mt):
            parts = [
                hl.Interval(start=hl.Struct(row_idx=s), end=hl.Struct(row_idx=e), includes_start=_is, includes_end=ie)
                for (s, e, _is, ie) in parts
            ]

            tmp = new_temp_file(extension='mt')
            mt.write(tmp, _partitions=parts)

            mt2 = hl.read_matrix_table(tmp)
            self.assertEqual(mt2.n_partitions(), len(parts))
            self.assertTrue(mt2._same(expected))

        test_parts([(0, 40, True, False)])

        test_parts([(-34, -31, True, True), (-30, 9, True, True), (10, 107, True, True), (108, 1000, True, True)])

        test_parts([(0, 5, True, False), (35, 40, True, True)], mt.filter_rows((mt.row_idx < 5) | (mt.row_idx >= 35)))

        test_parts([(5, 35, True, False)], mt.filter_rows((mt.row_idx >= 5) & (mt.row_idx < 35)))

    def test_partitioned_write_coerce(self):
        mt = hl.import_vcf(resource('sample.vcf'))
        parts = [hl.Interval(hl.Locus('20', 10277621), hl.Locus('20', 11898992))]
        tmp = new_temp_file(extension='mt')
        mt.write(tmp, _partitions=parts)

        mt2 = hl.read_matrix_table(tmp)
        assert mt2.aggregate_rows(
            hl.agg.all(hl.literal(hl.Interval(hl.Locus('20', 10277621), hl.Locus('20', 11898992))).contains(mt2.locus))
        )
        assert mt2.n_partitions() == len(parts)
        assert hl.filter_intervals(mt, parts)._same(mt2)

    def test_overwrite(self):
        mt = hl.utils.range_matrix_table(1, 1)
        f = new_temp_file(extension='mt')
        mt.write(f)

        with pytest.raises(hl.utils.FatalError, match="file already exists"):
            mt.write(f)

        mt.write(f, overwrite=True)

    def test_invalid_metadata(self):
        with pytest.raises(hl.utils.FatalError, match='metadata does not contain file version'):
            hl.read_matrix_table(resource('0.1-1fd5cc7.vds'))

    def test_legacy_files_with_required_globals(self):
        hl.read_table(resource('required_globals.ht'))._force_count()
        hl.read_matrix_table(resource('required_globals.mt'))._force_count_rows()

    def test_matrix_native_write_range(self):
        mt = hl.utils.range_matrix_table(11, 3, n_partitions=3)
        f = new_temp_file()
        mt.write(f)
        assert hl.read_matrix_table(f)._same(mt)

    def test_matrix_multi_write_range(self):
        mts = [
            hl.utils.range_matrix_table(11, 27, n_partitions=10),
            hl.utils.range_matrix_table(11, 3, n_partitions=10),
        ]
        f = new_temp_file()
        hl.experimental.write_matrix_tables(mts, f)
        assert hl.read_matrix_table(f + '0.mt')._same(mts[0])
        assert hl.read_matrix_table(f + '1.mt')._same(mts[1])

    def test_key_cols_by_extract_issue(self):
        mt = hl.utils.range_matrix_table(1000, 100)
        mt = mt.key_cols_by(col_id=hl.str(mt.col_idx))
        mt = mt.add_col_index()
        mt.show()

    def test_filtered_entries_group_rows_by(self):
        mt = hl.utils.range_matrix_table(1, 1)
        mt = mt.filter_entries(False)
        mt = mt.group_rows_by(x=mt.row_idx // 10).aggregate(c=hl.agg.count())
        assert mt.entries().collect() == [hl.Struct(x=0, col_idx=0, c=0)]

    def test_filtered_entries_group_cols_by(self):
        mt = hl.utils.range_matrix_table(1, 1)
        mt = mt.filter_entries(False)
        mt = mt.group_cols_by(x=mt.col_idx // 10).aggregate(c=hl.agg.count())
        assert mt.entries().collect() == [hl.Struct(row_idx=0, x=0, c=0)]

    def test_invalid_field_ref_error(self):
        mt = hl.balding_nichols_model(2, 5, 5)
        mt2 = hl.balding_nichols_model(2, 5, 5)
        with pytest.raises(hl.expr.ExpressionException, match='Found fields from 2 objects:'):
            mt.annotate_entries(x=mt.GT.n_alt_alleles() * mt2.af)

    def test_invalid_field_ref_annotate(self):
        mt = hl.balding_nichols_model(2, 5, 5)
        mt2 = hl.balding_nichols_model(2, 5, 5)
        with pytest.raises(hl.expr.ExpressionException, match='source mismatch'):
            mt.annotate_entries(x=mt2.af)

    def test_filter_locus_position_collect_returns_data(self):
        t = hl.utils.range_table(1)
        t = t.key_by(locus=hl.locus('2', t.idx + 1))
        assert t.filter(t.locus.position >= 1).collect() == [
            hl.utils.Struct(idx=0, locus=hl.genetics.Locus(contig='2', position=1, reference_genome='GRCh37'))
        ]

    @fails_local_backend()
    def test_lower_row_agg_init_arg(self):
        mt = hl.balding_nichols_model(5, 200, 200)
        mt2 = hl.variant_qc(mt)
        mt2 = mt2.filter_rows((mt2.variant_qc.AF[0] > 0.05) & (mt2.variant_qc.AF[0] < 0.95))
        mt2 = mt2.sample_rows(0.99)
        rows = mt2.rows()
        mt = mt.semi_join_rows(rows)
        hl.hwe_normalized_pca(mt.GT)


def test_keys_before_scans():
    mt = hl.utils.range_matrix_table(6, 6)
    mt = mt.annotate_rows(rev_idx=-mt.row_idx)
    mt = mt.key_rows_by(mt.rev_idx)

    mt = mt.annotate_rows(idx_scan=hl.scan.collect(mt.row_idx))

    mt = mt.key_rows_by(mt.row_idx)
    assert mt.rows().idx_scan.collect() == [[5, 4, 3, 2, 1], [5, 4, 3, 2], [5, 4, 3], [5, 4], [5], []]


def test_read_write_all_types():
    mt = create_all_values_matrix_table()
    tmp_file = new_temp_file()
    mt.write(tmp_file)
    assert hl.read_matrix_table(tmp_file)._same(mt)


def test_read_write_balding_nichols_model():
    mt = hl.balding_nichols_model(3, 10, 10)
    tmp_file = new_temp_file()
    mt.write(tmp_file)
    assert hl.read_matrix_table(tmp_file)._same(mt)


def test_read_partitions():
    ht = hl.utils.range_matrix_table(n_rows=100, n_cols=10, n_partitions=3)
    path = new_temp_file()
    ht.write(path)
    assert hl.read_matrix_table(path, _n_partitions=10).n_partitions() == 10


def test_filter_against_invalid_contig():
    mt = hl.balding_nichols_model(3, 5, 20)
    fmt = mt.filter_rows(mt.locus.contig == "chr1")
    assert fmt.rows()._force_count() == 0


def assert_unique_uids(mt):
    x = mt.aggregate_rows(hl.struct(r=hl.agg.collect_as_set(hl.rand_int64()), n=hl.agg.count()))
    assert len(x.r) == x.n
    x = mt.aggregate_cols(hl.struct(r=hl.agg.collect_as_set(hl.rand_int64()), n=hl.agg.count()))
    assert len(x.r) == x.n
    x = mt.aggregate_entries(hl.struct(r=hl.agg.collect_as_set(hl.rand_int64()), n=hl.agg.count()))
    assert len(x.r) == x.n


def assert_contains_node(t, node):
    assert t._mir.base_search(lambda x: isinstance(x, node))


def test_matrix_randomness_read():
    mt = hl.utils.range_matrix_table(10, 10, 3)
    assert_contains_node(mt, ir.MatrixRead)
    assert_unique_uids(mt)


@test_timeout(batch=8 * 60)
def test_matrix_randomness_aggregate_rows_by_key_with_body_randomness():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = (
        rmt.group_rows_by(k=rmt.row_idx % 5)
        .aggregate_rows(r=hl.rand_int64())
        .aggregate_entries(e=hl.rand_int64())
        .result()
    )
    assert_contains_node(mt, ir.MatrixAggregateRowsByKey)
    x = mt.aggregate_rows(hl.struct(r=hl.agg.collect_as_set(mt.r), n=hl.agg.count()))
    assert len(x.r) == x.n
    x = mt.aggregate_entries(hl.struct(r=hl.agg.collect_as_set(mt.e), n=hl.agg.count()))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


@test_timeout(batch=8 * 60)
def test_matrix_randomness_aggregate_rows_by_key_then_aggregate_entries_with_agg_randomness():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = (
        rmt.group_rows_by(k=rmt.row_idx % 5)
        .aggregate_rows(r=hl.agg.collect(hl.rand_int64()))
        .aggregate_entries(e=hl.agg.collect(hl.rand_int64()))
        .result()
    )
    assert_contains_node(mt, ir.MatrixAggregateRowsByKey)
    x = mt.aggregate_rows(hl.agg.explode(lambda r: hl.struct(r=hl.agg.collect_as_set(r), n=hl.agg.count()), mt.r))
    assert len(x.r) == x.n
    x = mt.aggregate_entries(hl.agg.explode(lambda r: hl.struct(r=hl.agg.collect_as_set(r), n=hl.agg.count()), mt.e))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


@test_timeout(batch=8 * 60)
def test_matrix_randomness_aggregate_rows_by_key_without_body_randomness():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = (
        rmt.group_rows_by(k=rmt.row_idx % 5)
        .aggregate_rows(row_agg=hl.agg.sum(rmt.row_idx))
        .aggregate_entries(entry_agg=hl.agg.sum(rmt.row_idx + rmt.col_idx))
        .result()
    )
    assert_contains_node(mt, ir.MatrixAggregateRowsByKey)
    assert_unique_uids(mt)


def test_matrix_randomness_filter_rows_with_cond_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.filter_rows(hl.rand_int64() % 2 == 0)
    assert_contains_node(mt, ir.MatrixFilterRows)
    mt.entries()._force_count()  # test with no consumer randomness
    assert_unique_uids(mt)


def test_matrix_randomness_filter_rows_without_cond_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.filter_rows(rmt.row_idx < 5)
    assert_contains_node(mt, ir.MatrixFilterRows)
    assert_unique_uids(mt)


def test_matrix_randomness_choose_cols():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.choose_cols([2, 3, 7])
    assert_contains_node(mt, ir.MatrixChooseCols)
    assert_unique_uids(mt)


def test_matrix_randomness_map_cols_with_body_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_cols(r=hl.rand_int64())
    assert_contains_node(mt, ir.MatrixMapCols)
    x = mt.aggregate_cols(hl.struct(r=hl.agg.collect_as_set(mt.r), n=hl.agg.count()))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


def test_matrix_randomness_map_cols_with_agg_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_cols(r=hl.agg.collect(hl.rand_int64()))
    assert_contains_node(mt, ir.MatrixMapCols)
    x = mt.aggregate_cols(hl.agg.explode(lambda r: hl.struct(r=hl.agg.collect_as_set(r), n=hl.agg.count()), mt.r))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


def test_matrix_randomness_map_cols_with_scan_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_cols(r=hl.scan.collect(hl.rand_int64()))
    assert_contains_node(mt, ir.MatrixMapCols)
    x = mt.aggregate_cols(hl.struct(r=hl.agg.explode(lambda r: hl.agg.collect_as_set(r), mt.r), n=hl.agg.count()))
    assert len(x.r) == x.n - 1
    assert_unique_uids(mt)


def test_matrix_randomness_map_cols_without_body_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_cols(x=2 * rmt.col_idx)
    assert_contains_node(mt, ir.MatrixMapCols)
    assert_unique_uids(mt)


def test_matrix_randomness_union_cols():
    r, c = 5, 5
    mt = hl.utils.range_matrix_table(2 * r, c)
    mt2 = hl.utils.range_matrix_table(2 * r, c)
    mt2 = mt2.key_rows_by(row_idx=mt2.row_idx + r)
    mt2 = mt2.key_cols_by(col_idx=mt2.col_idx + c)
    mt = mt.union_cols(mt2)
    assert_contains_node(mt, ir.MatrixUnionCols)
    assert_unique_uids(mt)


def test_matrix_randomness_map_entries_with_body_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_entries(r=hl.rand_int64())
    assert_contains_node(mt, ir.MatrixMapEntries)
    x = mt.aggregate_entries(hl.struct(r=hl.agg.collect_as_set(mt.r), n=hl.agg.count()))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


def test_matrix_randomness_map_entries_without_body_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_entries(x=rmt.row_idx + rmt.col_idx)
    assert_contains_node(mt, ir.MatrixMapEntries)
    assert_unique_uids(mt)


def test_matrix_randomness_filter_entries_with_cond_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.filter_entries(hl.rand_int64() % 2 == 0)
    assert_contains_node(mt, ir.MatrixFilterEntries)
    mt.entries()._force_count()  # test with no consumer randomness
    assert_unique_uids(mt)


def test_matrix_randomness_filter_entries_without_cond_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.filter_entries(rmt.row_idx + rmt.col_idx < 10)
    assert_contains_node(mt, ir.MatrixFilterEntries)
    assert_unique_uids(mt)


def test_matrix_randomness_key_rows_by():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.key_rows_by(k=rmt.row_idx // 4)
    assert_contains_node(mt, ir.MatrixKeyRowsBy)
    assert_unique_uids(mt)


def test_matrix_randomness_map_rows():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_rows(r=hl.rand_int64())
    assert_contains_node(mt, ir.MatrixMapRows)
    x = mt.aggregate_rows(hl.struct(r=hl.agg.collect_as_set(mt.r), n=hl.agg.count()))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


def test_matrix_randomness_map_rows_with_agg_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_rows(r=hl.agg.collect(hl.rand_int64()))
    assert_contains_node(mt, ir.MatrixMapRows)
    x = mt.aggregate_rows(hl.agg.explode(lambda r: hl.struct(r=hl.agg.collect_as_set(r), n=hl.agg.count()), mt.r))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


def test_matrix_randomness_map_rows_with_scan_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_rows(r=hl.scan.collect(hl.rand_int64()))
    assert_contains_node(mt, ir.MatrixMapRows)
    x = mt.aggregate_rows(hl.struct(r=hl.agg.explode(lambda r: hl.agg.collect_as_set(r), mt.r), n=hl.agg.count()))
    assert len(x.r) == x.n - 1
    assert_unique_uids(mt)


def test_matrix_randomness_map_rows_without_body_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_rows(x=2 * rmt.row_idx)
    assert_contains_node(mt, ir.MatrixMapRows)
    assert_unique_uids(mt)


def test_matrix_randomness_map_globals_with_body_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_globals(x=hl.rand_int64())
    assert_contains_node(mt, ir.MatrixMapGlobals)
    mt.entries()._force_count()  # test with no consumer randomness
    assert_unique_uids(mt)


def test_matrix_randomness_map_globals_without_body_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.annotate_globals(x=1)
    assert_contains_node(mt, ir.MatrixMapGlobals)
    assert_unique_uids(mt)


def test_matrix_randomness_filter_cols_with_cond_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.filter_cols(hl.rand_int64() % 2 == 0)
    assert_contains_node(mt, ir.MatrixFilterCols)
    mt.entries()._force_count()  # test with no consumer randomness
    assert_unique_uids(mt)


def test_matrix_randomness_filter_cols_without_cond_randomness():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.filter_cols(rmt.col_idx < 5)
    assert_contains_node(mt, ir.MatrixFilterCols)
    assert_unique_uids(mt)


def test_matrix_randomness_collect_cols_by_key():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.key_cols_by(k=rmt.col_idx % 5)
    mt = mt.collect_cols_by_key()
    assert_contains_node(mt, ir.MatrixCollectColsByKey)
    assert_unique_uids(mt)


@test_timeout(batch=5 * 60)
def test_matrix_randomness_aggregate_cols_by_key_with_body_randomness():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = (
        rmt.group_cols_by(k=rmt.col_idx % 5)
        .aggregate_cols(r=hl.rand_int64())
        .aggregate_entries(e=hl.rand_int64())
        .result()
    )
    assert_contains_node(mt, ir.MatrixAggregateColsByKey)
    x = mt.aggregate_cols(hl.struct(r=hl.agg.collect_as_set(mt.r), n=hl.agg.count()))
    assert len(x.r) == x.n
    x = mt.aggregate_entries(hl.struct(r=hl.agg.collect_as_set(mt.e), n=hl.agg.count()))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


@test_timeout(batch=5 * 60)
def test_matrix_randomness_aggregate_cols_by_key_with_agg_randomness():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = (
        rmt.group_cols_by(k=rmt.col_idx % 5)
        .aggregate_cols(r=hl.agg.collect(hl.rand_int64()))
        .aggregate_entries(e=hl.agg.collect(hl.rand_int64()))
        .result()
    )
    assert_contains_node(mt, ir.MatrixAggregateColsByKey)
    x = mt.aggregate_cols(hl.agg.explode(lambda r: hl.struct(r=hl.agg.collect_as_set(r), n=hl.agg.count()), mt.r))
    assert len(x.r) == x.n
    x = mt.aggregate_entries(hl.agg.explode(lambda r: hl.struct(r=hl.agg.collect_as_set(r), n=hl.agg.count()), mt.e))
    assert len(x.r) == x.n
    assert_unique_uids(mt)


@test_timeout(batch=5 * 60)
def test_matrix_randomness_aggregate_cols_by_key_without_body_randomness():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = (
        rmt.group_cols_by(k=rmt.col_idx % 5)
        .aggregate_cols(row_agg=hl.agg.sum(rmt.col_idx))
        .aggregate_entries(entry_agg=hl.agg.sum(rmt.row_idx + rmt.col_idx))
        .result()
    )
    assert_contains_node(mt, ir.MatrixAggregateColsByKey)
    assert_unique_uids(mt)


def test_matrix_randomness_explode_rows():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = rmt.annotate_rows(s=hl.struct(a=hl.range(rmt.row_idx)))
    mt = mt.explode_rows(mt.s.a)
    assert_contains_node(mt, ir.MatrixExplodeRows)
    assert_unique_uids(mt)


def test_matrix_randomness_repartition():
    if not hl.current_backend().requires_lowering:
        rmt = hl.utils.range_matrix_table(20, 10, 3)
        mt = rmt.repartition(5)
        assert_contains_node(mt, ir.MatrixRepartition)
        assert_unique_uids(mt)


def test_matrix_randomness_union_rows():
    r, c = 5, 5
    mt = hl.utils.range_matrix_table(2 * r, c)
    mt2 = hl.utils.range_matrix_table(2 * r, c)
    mt2 = mt2.key_rows_by(row_idx=mt2.row_idx + r)
    mt = mt.union_rows(mt2)
    assert_contains_node(mt, ir.MatrixUnionRows)
    assert_unique_uids(mt)


def test_matrix_randomness_distinct_by_row():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = rmt.key_rows_by(k=rmt.row_idx % 5)
    mt = mt.distinct_by_row()
    assert_contains_node(mt, ir.MatrixDistinctByRow)
    assert_unique_uids(mt)


def test_matrix_randomness_rows_head():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = rmt.head(10)
    assert_contains_node(mt, ir.MatrixRowsHead)
    assert_unique_uids(mt)


def test_matrix_randomness_cols_head():
    rmt = hl.utils.range_matrix_table(10, 20, 3)
    mt = rmt.head(None, 10)
    assert_contains_node(mt, ir.MatrixColsHead)
    assert_unique_uids(mt)


def test_matrix_randomness_rows_tail():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    mt = rmt.tail(10)
    assert_contains_node(mt, ir.MatrixRowsTail)
    assert_unique_uids(mt)


def test_matrix_randomness_cols_tail():
    rmt = hl.utils.range_matrix_table(10, 20, 3)
    mt = rmt.tail(None, 10)
    assert_contains_node(mt, ir.MatrixColsTail)
    assert_unique_uids(mt)


def test_matrix_randomness_explode_cols():
    rmt = hl.utils.range_matrix_table(10, 20, 3)
    mt = rmt.annotate_cols(s=hl.struct(a=hl.range(rmt.col_idx)))
    mt = mt.explode_cols(mt.s.a)
    assert_contains_node(mt, ir.MatrixExplodeCols)
    assert_unique_uids(mt)


def test_matrix_randomness_cast_table_to_matrix():
    rt = hl.utils.range_table(10, 3)
    t = rt.annotate(e=hl.range(10).map(lambda i: hl.struct(x=i)))
    t = t.annotate_globals(c=hl.range(10).map(lambda i: hl.struct(y=i)))
    mt = t._unlocalize_entries('e', 'c', [])
    assert_contains_node(mt, ir.CastTableToMatrix)
    assert_unique_uids(mt)


def test_matrix_randomness_annotate_rows_table():
    t = hl.utils.range_table(12, 3)
    t = t.key_by(k=(t.idx // 2) * 2)
    mt = hl.utils.range_matrix_table(8, 10, 3)
    mt = mt.key_rows_by(k=(mt.row_idx // 2) * 3)
    joined = mt.annotate_rows(x=t[mt.k].idx)
    assert_contains_node(joined, ir.MatrixAnnotateRowsTable)
    assert_unique_uids(joined)


def test_matrix_randomness_annotate_cols_table():
    t = hl.utils.range_table(12, 3)
    t = t.key_by(k=(t.idx // 2) * 2)
    mt = hl.utils.range_matrix_table(10, 8, 3)
    mt = mt.key_cols_by(k=(mt.col_idx // 2) * 3)
    joined = mt.annotate_cols(x=t[mt.k].idx)
    assert_contains_node(joined, ir.MatrixAnnotateColsTable)
    assert_unique_uids(joined)


def test_matrix_randomness_to_matrix_apply():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt._filter_partitions([0, 2])
    assert_contains_node(mt, ir.MatrixToMatrixApply)
    assert_unique_uids(mt)


def test_matrix_randomness_rename():
    rmt = hl.utils.range_matrix_table(10, 10, 3)
    mt = rmt.rename({'row_idx': 'row_index'})
    assert_contains_node(mt, ir.MatrixRename)
    assert_unique_uids(mt)


def test_matrix_randomness_filter_intervals():
    rmt = hl.utils.range_matrix_table(20, 10, 3)
    intervals = [hl.interval(0, 5), hl.interval(10, 15)]
    mt = hl.filter_intervals(rmt, intervals)
    assert_contains_node(mt, ir.MatrixFilterIntervals)
    assert_unique_uids(mt)


def test_upcast_tuples():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_cols(foo=[('0', 1)])
    t = t.explode_cols(t.foo)
    t = t.annotate_cols(x=t.foo[1])
    t = t.drop('foo')
    t.cols().collect()
