import math
import operator
import random
import unittest

import hail as hl
import hail.expr.aggregators as agg
from hail.utils.misc import new_temp_file
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def get_vds(self, min_partitions=None) -> hl.MatrixTable:
        return hl.import_vcf(resource("sample.vcf"), min_partitions=min_partitions)

    def test_range_count(self):
        self.assertEqual(hl.utils.range_matrix_table(7, 13).count(), (7, 13))

    def test_row_key_field_show_runs(self):
        ds = self.get_vds()
        ds.locus.show()

    def test_update(self):
        vds = self.get_vds()
        vds = vds.select_entries(dp=vds.DP, gq=vds.GQ)
        self.assertTrue(schema_eq(vds.entry.dtype, hl.tstruct(dp=hl.tint32, gq=hl.tint32)))

    def test_annotate(self):
        vds = self.get_vds()
        vds = vds.annotate_globals(foo=5)

        self.assertEqual(vds.globals.dtype, hl.tstruct(foo=hl.tint32))

        vds = vds.annotate_rows(x1=agg.count(),
                                x2=agg.fraction(False),
                                x3=agg.count_where(True),
                                x4=vds.info.AC + vds.foo)

        vds = vds.annotate_cols(apple=6)
        vds = vds.annotate_cols(y1=agg.count(),
                                y2=agg.fraction(False),
                                y3=agg.count_where(True),
                                y4=vds.foo + vds.apple)

        expected_schema = hl.tstruct(s=hl.tstr, apple=hl.tint32, y1=hl.tint64, y2=hl.tfloat64, y3=hl.tint64,
                                     y4=hl.tint32)

        self.assertTrue(schema_eq(vds.col.dtype, expected_schema),
                        "expected: " + str(vds.col.dtype) + "\nactual: " + str(expected_schema))

        vds = vds.select_entries(z1=vds.x1 + vds.foo,
                                 z2=vds.x1 + vds.y1 + vds.foo)
        self.assertTrue(schema_eq(vds.entry.dtype, hl.tstruct(z1=hl.tint64, z2=hl.tint64)))

    def test_annotate_globals(self):
        mt = hl.utils.range_matrix_table(1, 1)
        ht = hl.utils.range_table(1, 1)
        data = [
            (5, hl.tint, operator.eq),
            (float('nan'), hl.tfloat32, lambda x, y: str(x) == str(y)),
            (float('inf'), hl.tfloat64, lambda x, y: str(x) == str(y)),
            (float('-inf'), hl.tfloat64, lambda x, y: str(x) == str(y)),
            (1.111, hl.tfloat64, operator.eq),
            ([hl.Struct(**{'a': None, 'b': 5}),
              hl.Struct(**{'a': 'hello', 'b': 10})], hl.tarray(hl.tstruct(a=hl.tstr, b=hl.tint)), operator.eq)
        ]

        for x, t, f in data:
            self.assertTrue(f(hl.eval(mt.annotate_globals(foo=hl.literal(x, t)).foo), x), f"{x}, {t}")
            self.assertTrue(f(hl.eval(ht.annotate_globals(foo=hl.literal(x, t)).foo), x), f"{x}, {t}")

    def test_filter(self):
        vds = self.get_vds()
        vds = vds.annotate_globals(foo=5)
        vds = vds.annotate_rows(x1=agg.count())
        vds = vds.annotate_cols(y1=agg.count())
        vds = vds.annotate_entries(z1=vds.DP)

        vds = vds.filter_rows((vds.x1 == 5) & (agg.count() == 3) & (vds.foo == 2))
        vds = vds.filter_cols((vds.y1 == 5) & (agg.count() == 3) & (vds.foo == 2))
        vds = vds.filter_entries((vds.z1 < 5) & (vds.y1 == 3) & (vds.x1 == 5) & (vds.foo == 2))
        vds.count_rows()

    def test_aggregate(self):
        vds = self.get_vds()

        vds = vds.annotate_globals(foo=5)
        vds = vds.annotate_rows(x1=agg.count())
        vds = vds.annotate_cols(y1=agg.count())
        vds = vds.annotate_entries(z1=vds.DP)

        qv = vds.aggregate_rows(agg.count())
        qs = vds.aggregate_cols(agg.count())
        qg = vds.aggregate_entries(agg.count())

        self.assertIsNotNone(vds.aggregate_entries(hl.agg.take(vds.s, 1)[0]))

        self.assertEqual(qv, 346)
        self.assertEqual(qs, 100)
        self.assertEqual(qg, qv * qs)

        qvs = vds.aggregate_rows(hl.Struct(x=agg.collect(vds.locus.contig),
                                           y=agg.collect(vds.x1)))

        qss = vds.aggregate_cols(hl.Struct(x=agg.collect(vds.s),
                                           y=agg.collect(vds.y1)))

        qgs = vds.aggregate_entries(hl.Struct(x=agg.filter(False, agg.collect(vds.y1)),
                                              y=agg.filter(hl.rand_bool(0.1), agg.collect(vds.GT))))

    def test_aggregate_ir(self):
        ds = (hl.utils.range_matrix_table(5, 5)
              .annotate_globals(g1=5)
              .annotate_entries(e1=3))

        x = [("col_idx", lambda e: ds.aggregate_cols(e)),
             ("row_idx", lambda e: ds.aggregate_rows(e))]

        for name, f in x:
            r = f(hl.struct(x=agg.sum(ds[name]) + ds.g1,
                            y=agg.filter(ds[name] % 2 != 0, agg.sum(ds[name] + 2)) + ds.g1,
                            z=agg.sum(ds.g1 + ds[name]) + ds.g1,
                            mean=agg.mean(ds[name])))
            self.assertEqual(convert_struct_to_dict(r), {u'x': 15, u'y': 13, u'z': 40, u'mean': 2.0})

            r = f(5)
            self.assertEqual(r, 5)

            r = f(hl.null(hl.tint32))
            self.assertEqual(r, None)

            r = f(agg.filter(ds[name] % 2 != 0, agg.sum(ds[name] + 2)) + ds.g1)
            self.assertEqual(r, 13)

        r = ds.aggregate_entries(agg.filter((ds.row_idx % 2 != 0) & (ds.col_idx % 2 != 0),
                                            agg.sum(ds.e1 + ds.g1 + ds.row_idx + ds.col_idx)) + ds.g1)
        self.assertTrue(r, 48)

    def test_select_entries(self):
        mt = hl.utils.range_matrix_table(10, 10, n_partitions=4)
        mt = mt.annotate_entries(a=hl.struct(b=mt.row_idx, c=mt.col_idx), foo=mt.row_idx * 10 + mt.col_idx)
        mt = mt.select_entries(mt.a.b, mt.a.c, mt.foo)
        mt = mt.annotate_entries(bc=mt.b * 10 + mt.c)
        mt_entries = mt.entries()

        assert (mt_entries.all(mt_entries.bc == mt_entries.foo))

    def test_select_cols(self):
        mt = hl.utils.range_matrix_table(3, 5, n_partitions=4)
        mt = mt.annotate_entries(e=mt.col_idx * mt.row_idx)
        mt = mt.annotate_globals(g=1)
        mt = mt.annotate_cols(sum=agg.sum(mt.e + mt.col_idx + mt.row_idx + mt.g) + mt.col_idx + mt.g,
                              count=agg.count_where(mt.e % 2 == 0),
                              foo=agg.count())

        result = convert_struct_to_dict(mt.cols().collect()[-2])
        self.assertEqual(result, {'col_idx': 3, 'sum': 28, 'count': 2, 'foo': 3})

    def test_drop(self):
        vds = self.get_vds()
        vds = vds.annotate_globals(foo=5)
        vds = vds.annotate_cols(bar=5)
        vds1 = vds.drop('GT', 'info', 'foo', 'bar')
        self.assertTrue('foo' not in vds1.globals)
        self.assertTrue('info' not in vds1.row)
        self.assertTrue('bar' not in vds1.col)
        self.assertTrue('GT' not in vds1.entry)
        vds1._force_count_rows()

        vds2 = vds.drop(vds.GT, vds.info, vds.foo, vds.bar)
        self.assertTrue('foo' not in vds2.globals)
        self.assertTrue('info' not in vds2.row)
        self.assertTrue('bar' not in vds2.col)
        self.assertTrue('GT' not in vds2.entry)
        vds2._force_count_rows()

    def test_explode_rows(self):
        mt = hl.utils.range_matrix_table(4, 4)
        mt = mt.annotate_entries(e=mt.row_idx * 10 + mt.col_idx)

        self.assertTrue(mt.annotate_rows(x=[1]).explode_rows('x').drop('x')._same(mt))

        self.assertEqual(mt.annotate_rows(x=hl.empty_array('int')).explode_rows('x').count_rows(), 0)
        self.assertEqual(mt.annotate_rows(x=hl.null('array<int>')).explode_rows('x').count_rows(), 0)
        self.assertEqual(mt.annotate_rows(x=hl.range(0, mt.row_idx)).explode_rows('x').count_rows(), 6)
        mt = mt.annotate_rows(x=hl.struct(y=hl.range(0, mt.row_idx)))
        self.assertEqual(mt.explode_rows(mt.x.y).count_rows(), 6)

    def test_explode_cols(self):
        mt = hl.utils.range_matrix_table(4, 4)
        mt = mt.annotate_entries(e=mt.row_idx * 10 + mt.col_idx)

        self.assertTrue(mt.annotate_cols(x=[1]).explode_cols('x').drop('x')._same(mt))

        self.assertEqual(mt.annotate_cols(x=hl.empty_array('int')).explode_cols('x').count_cols(), 0)
        self.assertEqual(mt.annotate_cols(x=hl.null('array<int>')).explode_cols('x').count_cols(), 0)
        self.assertEqual(mt.annotate_cols(x=hl.range(0, mt.col_idx)).explode_cols('x').count_cols(), 6)

    def test_explode_key_errors(self):
        mt = hl.utils.range_matrix_table(1, 1).key_cols_by(a=[1]).key_rows_by(b=[1])
        with self.assertRaises(ValueError):
            mt.explode_cols('a')
        with self.assertRaises(ValueError):
            mt.explode_rows('b')

    def test_aggregate_cols_by(self):
        mt = hl.utils.range_matrix_table(2, 4)
        mt = (mt.annotate_cols(group=mt.col_idx < 2)
              .annotate_globals(glob=5))
        grouped = mt.group_cols_by(mt.group)
        result = grouped.aggregate(sum=hl.agg.sum(mt.row_idx * 2 + mt.col_idx + mt.glob) + 3)

        expected = (hl.Table.parallelize([
            {'row_idx': 0, 'group': True, 'sum': 14},
            {'row_idx': 0, 'group': False, 'sum': 18},
            {'row_idx': 1, 'group': True, 'sum': 18},
            {'row_idx': 1, 'group': False, 'sum': 22}
        ], hl.tstruct(row_idx=hl.tint, group=hl.tbool, sum=hl.tint64))
                    .annotate_globals(glob=5)
                    .key_by('row_idx', 'group'))

        self.assertTrue(result.entries()._same(expected))

    def test_aggregate_rows_by(self):
        mt = hl.utils.range_matrix_table(4, 2)
        mt = (mt.annotate_rows(group=mt.row_idx < 2)
              .annotate_globals(glob=5))
        grouped = mt.group_rows_by(mt.group)
        result = grouped.aggregate(sum=hl.agg.sum(mt.col_idx * 2 + mt.row_idx + mt.glob) + 3)

        expected = (hl.Table.parallelize([
            {'col_idx': 0, 'group': True, 'sum': 14},
            {'col_idx': 1, 'group': True, 'sum': 18},
            {'col_idx': 0, 'group': False, 'sum': 18},
            {'col_idx': 1, 'group': False, 'sum': 22}
        ], hl.tstruct(group=hl.tbool, col_idx=hl.tint, sum=hl.tint64))
                    .annotate_globals(glob=5)
                    .key_by('group', 'col_idx'))

        self.assertTrue(result.entries()._same(expected))

    def test_collect_cols_by_key(self):
        mt = hl.utils.range_matrix_table(3, 3)
        col_dict = hl.literal({0: [1], 1: [2, 3], 2: [4, 5, 6]})
        mt = mt.annotate_cols(foo=col_dict.get(mt.col_idx)) \
            .explode_cols('foo')
        mt = mt.annotate_entries(bar=mt.row_idx * mt.foo)

        grouped = mt.collect_cols_by_key()

        self.assertListEqual(grouped.cols().order_by('col_idx').collect(),
                             [hl.Struct(col_idx=0, foo=[1]),
                              hl.Struct(col_idx=1, foo=[2, 3]),
                              hl.Struct(col_idx=2, foo=[4, 5, 6])])
        self.assertListEqual(
            grouped.entries().select('bar')
                .order_by('row_idx', 'col_idx').collect(),
            [hl.Struct(row_idx=0, col_idx=0, bar=[0]),
             hl.Struct(row_idx=0, col_idx=1, bar=[0, 0]),
             hl.Struct(row_idx=0, col_idx=2, bar=[0, 0, 0]),
             hl.Struct(row_idx=1, col_idx=0, bar=[1]),
             hl.Struct(row_idx=1, col_idx=1, bar=[2, 3]),
             hl.Struct(row_idx=1, col_idx=2, bar=[4, 5, 6]),
             hl.Struct(row_idx=2, col_idx=0, bar=[2]),
             hl.Struct(row_idx=2, col_idx=1, bar=[4, 6]),
             hl.Struct(row_idx=2, col_idx=2, bar=[8, 10, 12])])

    def test_weird_names(self):
        ds = self.get_vds()
        exprs = {'a': 5, '   a    ': 5, r'\%!^!@#&#&$%#$%': [5]}

        ds.annotate_globals(**exprs)
        ds.select_globals(**exprs)

        ds.annotate_cols(**exprs)
        ds1 = ds.select_cols(**exprs)

        ds.annotate_rows(**exprs)
        ds2 = ds.select_rows(**exprs)

        ds.annotate_entries(**exprs)
        ds.select_entries(**exprs)

        ds1.explode_cols('\%!^!@#&#&$%#$%')
        ds1.explode_cols(ds1['\%!^!@#&#&$%#$%'])
        ds1.group_cols_by(ds1.a).aggregate(**{'*``81': agg.count()})

        ds1.drop('\%!^!@#&#&$%#$%')
        ds1.drop(ds1['\%!^!@#&#&$%#$%'])

        ds2.explode_rows('\%!^!@#&#&$%#$%')
        ds2.explode_rows(ds2['\%!^!@#&#&$%#$%'])
        ds2.group_rows_by(ds2.a).aggregate(**{'*``81': agg.count()})

    def test_joins(self):
        vds = self.get_vds().select_rows(x1=1, y1=1)
        vds2 = vds.select_rows(x2=1, y2=2)
        vds2 = vds2.select_cols(c1=1, c2=2)

        vds = vds.annotate_rows(y2=vds2.index_rows(vds.row_key).y2)
        vds = vds.annotate_cols(c2=vds2.index_cols(vds.s).c2)

        vds = vds.annotate_cols(c2=vds2.index_cols(hl.str(vds.s)).c2)

        rt = vds.rows()
        ct = vds.cols()

        vds.annotate_rows(**rt[vds.locus, vds.alleles])

        self.assertTrue(rt.all(rt.y2 == 2))
        self.assertTrue(ct.all(ct.c2 == 2))

    def test_joins_with_key_structs(self):
        mt = self.get_vds()

        rows = mt.rows()
        cols = mt.cols()

        self.assertEqual(rows[mt.locus, mt.alleles].take(1), rows[mt.row_key].take(1))
        self.assertEqual(cols[mt.s].take(1), cols[mt.col_key].take(1))

        self.assertEqual(mt.index_rows(mt.row_key).take(1), mt.index_rows(mt.locus, mt.alleles).take(1))
        self.assertEqual(mt.index_cols(mt.col_key).take(1), mt.index_cols(mt.s).take(1))
        self.assertEqual(mt[mt.row_key, mt.col_key].take(1), mt[(mt.locus, mt.alleles), mt.s].take(1))

    def test_table_join(self):
        ds = self.get_vds()
        # test different row schemas
        self.assertTrue(ds.union_cols(ds.drop(ds.info))
                        .count_rows(), 346)

    def test_naive_coalesce(self):
        vds = self.get_vds(min_partitions=8)
        self.assertEqual(vds.n_partitions(), 8)
        repart = vds.naive_coalesce(2)
        self.assertTrue(vds._same(repart))

    def test_coalesce_with_no_rows(self):
        mt = self.get_vds().filter_rows(False)
        self.assertEqual(mt.repartition(1).count_rows(), 0)

    def test_literals_rebuild(self):
        mt = hl.utils.range_matrix_table(1, 1)
        mt = mt.annotate_rows(x = hl.cond(hl.len(hl.literal([1,2,3])) < hl.rand_unif(10, 11), mt.globals, hl.struct()))
        mt._force_count_rows()

    def test_unions(self):
        dataset = hl.import_vcf(resource('sample2.vcf'))

        # test union_rows
        ds1 = dataset.filter_rows(dataset.locus.position % 2 == 1)
        ds2 = dataset.filter_rows(dataset.locus.position % 2 == 0)

        datasets = [ds1, ds2]
        r1 = ds1.union_rows(ds2)
        r2 = hl.MatrixTable.union_rows(*datasets)

        self.assertTrue(r1._same(r2))

        # test union_cols
        ds = dataset.union_cols(dataset).union_cols(dataset)
        for s, count in ds.aggregate_cols(agg.counter(ds.s)).items():
            self.assertEqual(count, 3)

    def test_union_cols_example(self):
        joined = hl.import_vcf(resource('joined.vcf'))

        left = hl.import_vcf(resource('joinleft.vcf'))
        right = hl.import_vcf(resource('joinright.vcf'))

        self.assertTrue(left.union_cols(right)._same(joined))

    def test_index(self):
        ds = self.get_vds(min_partitions=8)
        self.assertEqual(ds.n_partitions(), 8)
        ds = ds.add_row_index('rowidx').add_col_index('colidx')

        for i, struct in enumerate(ds.cols().select('colidx').collect()):
            self.assertEqual(i, struct.colidx)
        for i, struct in enumerate(ds.rows().select('rowidx').collect()):
            self.assertEqual(i, struct.rowidx)

    def test_choose_cols(self):
        ds = self.get_vds()
        indices = list(range(ds.count_cols()))
        random.shuffle(indices)

        old_order = ds.key_cols_by()['s'].collect()
        self.assertEqual(ds.choose_cols(indices).key_cols_by()['s'].collect(),
                         [old_order[i] for i in indices])

        self.assertEqual(ds.choose_cols(list(range(10))).s.collect(),
                         old_order[:10])

    def test_choose_cols_vs_explode(self):
        ds = self.get_vds()

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
        ds = self.get_vds()
        kt = hl.Table.parallelize(
            [{'key': 0, 'value': True},
             {'key': 1, 'value': False}],
            hl.tstruct(key=hl.tint32, value=hl.tbool),
            key=['key'])
        ds = ds.annotate_rows(key=ds.locus.position % 2)
        ds = ds.annotate_rows(value=kt[ds['key']]['value'])
        rt = ds.rows()
        self.assertTrue(
            rt.all(((rt.locus.position % 2) == 0) == rt['value']))

    def test_computed_key_join_2(self):
        # multiple keys
        ds = self.get_vds()
        kt = hl.Table.parallelize(
            [{'key1': 0, 'key2': 0, 'value': 0},
             {'key1': 1, 'key2': 0, 'value': 1},
             {'key1': 0, 'key2': 1, 'value': -2},
             {'key1': 1, 'key2': 1, 'value': -1}],
            hl.tstruct(key1=hl.tint32, key2=hl.tint32, value=hl.tint32),
            key=['key1', 'key2'])
        ds = ds.annotate_rows(key1=ds.locus.position % 2, key2=ds.info.DP % 2)
        ds = ds.annotate_rows(value=kt[ds.key1, ds.key2]['value'])
        rt = ds.rows()
        self.assertTrue(
            rt.all((rt.locus.position % 2) - 2 * (rt.info.DP % 2) == rt['value']))

    def test_computed_key_join_3(self):
        # duplicate row keys
        ds = self.get_vds()
        kt = hl.Table.parallelize(
            [{'culprit': 'InbreedingCoeff', 'foo': 'bar', 'value': 'IB'}],
            hl.tstruct(culprit=hl.tstr, foo=hl.tstr, value=hl.tstr),
            key=['culprit', 'foo'])
        ds = ds.annotate_rows(
            dsfoo='bar',
            info=ds.info.annotate(culprit=[ds.info.culprit, "foo"]))
        ds = ds.explode_rows(ds.info.culprit)
        ds = ds.annotate_rows(value=kt[ds.info.culprit, ds.dsfoo]['value'])
        rt = ds.rows()
        self.assertTrue(
            rt.all(hl.cond(
                rt.info.culprit == "InbreedingCoeff",
                rt['value'] == "IB",
                hl.is_missing(rt['value']))))

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

    def test_filter_cols_required_entries(self):
        mt1 = hl.utils.range_matrix_table(10, 10, n_partitions=4)
        mt1 = mt1.filter_cols(mt1.col_idx < 3)
        self.assertEqual(len(mt1.entries().collect()), 30)

    def test_filter_cols_with_global_references(self):
        mt = hl.utils.range_matrix_table(10, 10)
        s = hl.literal({1, 3, 5, 7})
        self.assertEqual(mt.filter_cols(s.contains(mt.col_idx)).count_cols(), 4)

    def test_vcf_regression(self):
        ds = hl.import_vcf(resource('33alleles.vcf'))
        self.assertEqual(
            ds.filter_rows(ds.alleles.length() == 2).count_rows(), 0)

    def test_field_groups(self):
        ds = self.get_vds()

        df = ds.annotate_rows(row_struct=ds.row).rows()
        self.assertTrue(df.all((df.info == df.row_struct.info) & (df.qual == df.row_struct.qual)))

        ds2 = ds.add_col_index()
        df = ds2.annotate_cols(col_struct=ds2.col).cols()
        self.assertTrue(df.all((df.col_idx == df.col_struct.col_idx)))

        df = ds.annotate_entries(entry_struct=ds.entry).entries()
        self.assertTrue(df.all(
            ((hl.is_missing(df.GT) |
              (df.GT == df.entry_struct.GT)) &
             (df.AD == df.entry_struct.AD))))

    def test_filter_partitions(self):
        ds = self.get_vds(min_partitions=8)
        self.assertEqual(ds.n_partitions(), 8)
        self.assertEqual(ds._filter_partitions([0, 1, 4]).n_partitions(), 3)
        self.assertEqual(ds._filter_partitions(range(3)).n_partitions(), 3)
        self.assertEqual(ds._filter_partitions([4, 5, 7], keep=False).n_partitions(), 5)
        self.assertTrue(
            ds._same(hl.MatrixTable.union_rows(
                ds._filter_partitions([0, 3, 7]),
                ds._filter_partitions([0, 3, 7], keep=False))))

    def test_from_rows_table(self):
        mt = hl.import_vcf(resource('sample.vcf'))
        mt = mt.annotate_globals(foo='bar')
        rt = mt.rows()
        rm = hl.MatrixTable.from_rows_table(rt)
        self.assertTrue(rm._same(mt.filter_cols(False).select_entries().key_cols_by().select_cols()))

    def test_sample_rows(self):
        ds = self.get_vds()
        ds_small = ds.sample_rows(0.01)
        self.assertTrue(ds_small.count_rows() < ds.count_rows())

    def test_read_stored_cols(self):
        ds = self.get_vds()
        ds = ds.annotate_globals(x='foo')
        f = new_temp_file(suffix='vds')
        ds.write(f)
        t = hl.read_table(f + '/cols')
        self.assertTrue(ds.cols()._same(t))

    def test_read_stored_rows(self):
        ds = self.get_vds()
        ds = ds.annotate_globals(x='foo')
        f = new_temp_file(suffix='vds')
        ds.write(f)
        t = hl.read_table(f + '/rows')
        self.assertTrue(ds.rows()._same(t))

    def test_read_stored_globals(self):
        ds = self.get_vds()
        ds = ds.annotate_globals(x=5, baz='foo')
        f = new_temp_file(suffix='vds')
        ds.write(f)
        t = hl.read_table(f + '/globals')
        self.assertTrue(ds.globals_table()._same(t))

    def test_codecs_matrix(self):
        from hail.utils.java import Env, scala_object
        codecs = scala_object(Env.hail().io, 'CodecSpec').codecSpecs()
        ds = self.get_vds()
        temp = new_temp_file(suffix='hmt')
        for codec in codecs:
            ds.write(temp, overwrite=True, _codec_spec=codec.toString())
            ds2 = hl.read_matrix_table(temp)
            self.assertTrue(ds._same(ds2))

    def test_codecs_table(self):
        from hail.utils.java import Env, scala_object
        codecs = scala_object(Env.hail().io, 'CodecSpec').codecSpecs()
        rt = self.get_vds().rows()
        temp = new_temp_file(suffix='ht')
        for codec in codecs:
            rt.write(temp, overwrite=True, _codec_spec=codec.toString())
            rt2 = hl.read_table(temp)
            self.assertTrue(rt._same(rt2))

    def test_fix3307_read_mt_wrong(self):
        mt = hl.import_vcf(resource('sample2.vcf'))
        mt = hl.split_multi_hts(mt)
        mt.write('/tmp/foo.mt', overwrite=True)
        mt2 = hl.read_matrix_table('/tmp/foo.mt')
        t = hl.read_table('/tmp/foo.mt/rows')
        self.assertTrue(mt.rows()._same(t))
        self.assertTrue(mt2.rows()._same(t))
        self.assertTrue(mt._same(mt2))

    def test_rename(self):
        dataset = self.get_vds()
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

        self.assertEqual(mt.filter_rows(hl.null(hl.tbool)).count_rows(), 0)
        self.assertEqual(mt.filter_cols(hl.null(hl.tbool)).count_cols(), 0)
        self.assertEqual(mt.filter_entries(hl.null(hl.tbool)).entries().count(), 0)

    def test_to_table_on_various_fields(self):
        mt = hl.utils.range_matrix_table(3, 4)

        sample_ids = ['Bob', 'Alice', 'David', 'Carol']
        entries = [1, 0, 3, 2]
        rows = ['1:3:A:G', '1:2:A:G', '1:0:A:G']

        mt = mt.annotate_cols(s=hl.array(sample_ids)[mt.col_idx]).key_cols_by('s')
        mt = mt.annotate_entries(e=hl.array(entries)[mt.col_idx])
        mt = mt.annotate_rows(r=hl.array(rows)[mt.row_idx]).key_rows_by('r')

        self.assertEqual(mt.s.collect(), sample_ids)
        self.assertEqual(mt.s.take(1), [sample_ids[0]])
        self.assertEqual(mt.e.collect(), entries * 3)
        self.assertEqual(mt.e.take(1), [entries[0]])
        self.assertEqual(mt.row_idx.collect(), [2, 1, 0])
        self.assertEqual(mt.r.collect(), sorted(rows))
        self.assertEqual(mt.r.take(1), [sorted(rows)[0]])

        self.assertEqual(mt.cols().s.collect(), sorted(sample_ids))
        self.assertEqual(mt.cols().s.take(1), [sorted(sample_ids)[0]])
        self.assertEqual(mt.entries().e.collect(), sorted(entries) * 3)
        self.assertEqual(mt.entries().e.take(1), [sorted(entries)[0]])
        self.assertEqual(mt.rows().row_idx.collect(), [2, 1, 0])
        self.assertEqual(mt.rows().r.collect(), sorted(rows))
        self.assertEqual(mt.rows().r.take(1), [sorted(rows)[0]])

    def test_order_by(self):
        ht = hl.utils.range_table(10)
        self.assertEqual(ht.order_by('idx').idx.collect(), list(range(10)))
        self.assertEqual(ht.order_by(hl.asc('idx')).idx.collect(), list(range(10)))
        self.assertEqual(ht.order_by(hl.desc('idx')).idx.collect(), list(range(10))[::-1])

    def test_order_by_intervals(self):
        intervals = {0: hl.Interval(0, 3, includes_start=True, includes_end=False),
                     1: hl.Interval(0, 4, includes_start=True, includes_end=True),
                     2: hl.Interval(1, 4, includes_start=True, includes_end=False),
                     3: hl.Interval(0, 4, includes_start=False, includes_end=False),
                     4: hl.Interval(0, 4, includes_start=True, includes_end=False)}
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

    def test_make_table(self):
        mt = hl.utils.range_matrix_table(3, 2)
        mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
        mt = mt.key_cols_by(col_idx=hl.str(mt.col_idx))

        t = hl.Table.parallelize(
            [{'row_idx': 0, '0.x': 0, '1.x': 0},
             {'row_idx': 1, '0.x': 0, '1.x': 1},
             {'row_idx': 2, '0.x': 0, '1.x': 2}],
            hl.tstruct(**{'row_idx': hl.tint32, '0.x': hl.tint32, '1.x': hl.tint32}),
            key='row_idx')

        self.assertTrue(mt.make_table()._same(t))

    def test_make_table_empty_entry_field(self):
        mt = hl.utils.range_matrix_table(3, 2)
        mt = mt.select_entries(**{'': mt.row_idx * mt.col_idx})
        mt = mt.key_cols_by(col_idx=hl.str(mt.col_idx))

        t = mt.make_table()
        self.assertEqual(
            t.row.dtype,
            hl.tstruct(**{'row_idx': hl.tint32, '0': hl.tint32, '1': hl.tint32}))

    def test_transmute(self):
        mt = (
            hl.utils.range_matrix_table(1, 1)
                .annotate_globals(g1=0, g2=0)
                .annotate_cols(c1=0, c2=0)
                .annotate_rows(r1=0, r2=0)
                .annotate_entries(e1=0, e2=0))
        self.assertEqual(mt.transmute_globals(g3=mt.g2 + 1).globals.dtype, hl.tstruct(g1=hl.tint, g3=hl.tint))
        self.assertEqual(mt.transmute_rows(r3=mt.r2 + 1).row_value.dtype, hl.tstruct(r1=hl.tint, r3=hl.tint))
        self.assertEqual(mt.transmute_cols(c3=mt.c2 + 1).col_value.dtype, hl.tstruct(c1=hl.tint, c3=hl.tint))
        self.assertEqual(mt.transmute_entries(e3=mt.e2 + 1).entry.dtype, hl.tstruct(e1=hl.tint, e3=hl.tint))

    def test_transmute_agg(self):
        mt = hl.utils.range_matrix_table(1, 1).annotate_entries(x=5)
        mt = mt.transmute_rows(y = hl.agg.mean(mt.x))

    def test_agg_explode(self):
        t = hl.Table.parallelize([
            hl.struct(a=[1, 2]),
            hl.struct(a=hl.empty_array(hl.tint32)),
            hl.struct(a=hl.null(hl.tarray(hl.tint32))),
            hl.struct(a=[3]),
            hl.struct(a=[hl.null(hl.tint32)])
        ])
        self.assertCountEqual(t.aggregate(hl.agg.explode(lambda elt: hl.agg.collect(elt), t.a)),
                              [1, 2, None, 3])

    def test_agg_call_stats(self):
        t = hl.Table.parallelize([
            hl.struct(c=hl.call(0, 0)),
            hl.struct(c=hl.call(0, 1)),
            hl.struct(c=hl.call(0, 2, phased=True)),
            hl.struct(c=hl.call(1)),
            hl.struct(c=hl.call(0)),
            hl.struct(c=hl.call())
        ])
        actual = t.aggregate(hl.agg.call_stats(t.c, ['A', 'T', 'G']))
        expected = hl.struct(AC=[5, 2, 1],
                             AF=[5.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0],
                             AN=8,
                             homozygote_count=[1, 0, 0])

        self.assertTrue(hl.Table.parallelize([actual]),
                        hl.Table.parallelize([expected]))

    def test_hardy_weinberg_test(self):
        mt = hl.import_vcf(resource('HWE_test.vcf'))
        mt = mt.select_rows(**hl.agg.hardy_weinberg_test(mt.GT))
        rt = mt.rows()
        expected = hl.Table.parallelize([
            hl.struct(
                locus=hl.locus('20', pos),
                alleles=alleles,
                het_freq_hwe=r,
                p_value=p)
            for (pos, alleles, r, p) in [
                (1, ['A', 'G'], 0.0, 0.5),
                (2, ['A', 'G'], 0.25, 0.5),
                (3, ['T', 'C'], 0.5357142857142857, 0.21428571428571427),
                (4, ['T', 'A'], 0.5714285714285714, 0.6571428571428573),
                (5, ['G', 'A'], 0.3333333333333333, 0.5)]],
            key=['locus', 'alleles'])
        self.assertTrue(rt.filter(rt.locus.position != 6)._same(expected))

        rt6 = rt.filter(rt.locus.position == 6).collect()[0]
        self.assertEqual(rt6['p_value'], 0.5)
        self.assertTrue(math.isnan(rt6['het_freq_hwe']))

    def test_hw_func_and_agg_agree(self):
        mt = hl.import_vcf(resource('sample.vcf'))
        mt = mt.annotate_rows(
            stats=hl.agg.call_stats(mt.GT, mt.alleles),
            hw=hl.agg.hardy_weinberg_test(mt.GT))
        mt = mt.annotate_rows(
            hw2=hl.hardy_weinberg_test(mt.stats.homozygote_count[0],
                                       mt.stats.AC[1] - 2 * mt.stats.homozygote_count[1],
                                       mt.stats.homozygote_count[1]))
        rt = mt.rows()
        self.assertTrue(rt.all(rt.hw == rt.hw2))

    def test_write_stage_locally(self):
        mt = self.get_vds()
        f = new_temp_file(suffix='mt')
        mt.write(f, stage_locally=True)

        mt2 = hl.read_matrix_table(f)
        self.assertTrue(mt._same(mt2))

    def test_nulls_in_distinct_joins(self):

        # MatrixAnnotateRowsTable uses left distinct join
        mr = hl.utils.range_matrix_table(7, 3, 4)
        matrix1 = mr.key_rows_by(new_key=hl.cond((mr.row_idx == 3) | (mr.row_idx == 5),
                                                hl.null(hl.tint32), mr.row_idx))
        matrix2 = mr.key_rows_by(new_key=hl.cond((mr.row_idx == 4) | (mr.row_idx == 6),
                                                hl.null(hl.tint32), mr.row_idx))

        joined = matrix1.select_rows(idx1=matrix1.row_idx,
                                     idx2=matrix2.rows()[matrix1.new_key].row_idx)

        def row(new_key, idx1, idx2):
            return hl.Struct(new_key=new_key, idx1=idx1, idx2=idx2)

        expected = [row(0, 0, 0),
                    row(1, 1, 1),
                    row(2, 2, 2),
                    row(4, 4, None),
                    row(6, 6, None),
                    row(None, 3, None),
                    row(None, 5, None)]
        self.assertEqual(joined.rows().collect(), expected)

        # union_cols uses inner distinct join
        matrix1 = matrix1.annotate_entries(ridx=matrix1.row_idx,
                                           cidx=matrix1.col_idx)
        matrix2 = matrix2.annotate_entries(ridx=matrix2.row_idx,
                                           cidx=matrix2.col_idx)
        matrix2 = matrix2.key_cols_by(col_idx=matrix2.col_idx + 3)

        expected = hl.utils.range_matrix_table(3, 6, 1)
        expected = expected.key_rows_by(new_key=expected.row_idx)
        expected = expected.annotate_entries(ridx=expected.row_idx,
                                             cidx=expected.col_idx % 3)

        self.assertTrue(matrix1.union_cols(matrix2)._same(expected))

    def test_row_joins_into_table(self):
        rt = hl.utils.range_matrix_table(9, 13, 3)
        mt1 = rt.key_rows_by(idx=rt.row_idx)
        mt1 = mt1.select_rows(v=mt1.idx + 2)
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
        self.assertEqual(t1.index(mt1.row_key).collect(), values)
        self.assertEqual(t2.index(mt2.row_key).collect(), values)
        self.assertEqual(t1.index(mt1.idx).collect(), values)
        self.assertEqual(t2.index(mt2.idx, mt2.idx2).collect(), values)
        self.assertEqual(t1.index(mt2.idx).collect(), values)
        with self.assertRaises(hl.expr.ExpressionException):
            t2.index(mt2.idx).collect()
        with self.assertRaises(hl.expr.ExpressionException):
            t2.index(mt1.row_key).collect()

        # join on not mt row key
        self.assertEqual(t1.index(mt1.v).collect(), [hl.Struct(v=i + 2) for i in range(2, 10)] + [None])
        self.assertEqual(t2.index(mt2.idx2, mt2.v).collect(), [hl.Struct(v=i + 2) for i in range(1, 10)])
        with self.assertRaises(hl.expr.ExpressionException):
            t2.index(mt2.v).collect()

        # join on interval of first field of mt row key
        self.assertEqual(tinterval1.index(mt1.idx).collect(), values)
        self.assertEqual(tinterval1.index(mt1.row_key).collect(), values)
        self.assertEqual(tinterval1.index(mt2.idx).collect(), values)

        with self.assertRaises(hl.expr.ExpressionException):
            tinterval1.index(mt2.row_key).collect()
        with self.assertRaises(hl.expr.ExpressionException):
            tinterval2.index(mt2.idx).collect()
        with self.assertRaises(hl.expr.ExpressionException):
            tinterval2.index(mt2.row_key).collect()
        with self.assertRaises(hl.expr.ExpressionException):
            tinterval2.index(mt2.idx, mt2.idx2).collect()
