import unittest
import hail as hl
import hail.ir as ir
from hail.utils.java import Env
from .helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class ValueIRTests(unittest.TestCase):
    def value_irs(self):
        b = ir.TrueIR()
        c = ir.Ref('c', hl.tbool)
        i = ir.I32(5)
        j = ir.I32(7)
        st = ir.Str('Hail')
        a = ir.Ref('a', hl.tarray(hl.tint32))
        aa = ir.Ref('aa', hl.tarray(hl.tarray(hl.tint32)))
        da = ir.Ref('da', hl.tarray(hl.ttuple(hl.tint32, hl.tstr)))
        v = ir.Ref('v', hl.tint32)
        s = ir.Ref('s', hl.tstruct(x = hl.tint32, y = hl.tint64, z = hl.tfloat64))
        t = ir.Ref('t', hl.ttuple(hl.tint32, hl.tint64, hl.tfloat64))
        call = ir.Ref('call', hl.tcall)

        collect_sig = ir.AggSignature('Collect', [], None, [hl.tint32])

        call_stats_sig = ir.AggSignature('CallStats', [], [hl.tint32], [hl.tcall])
        call_stats_type = hl.tstruct(AC=hl.tarray(hl.tint32),
                                     AF=hl.tarray(hl.tfloat64),
                                     AN=hl.tint32,
                                     homozygote_count=hl.tarray(hl.tint32))

        hist_sig = ir.AggSignature(
            'Histogram', [hl.tfloat64, hl.tfloat64, hl.tint32], None, [hl.tfloat64])
        hist_type = hl.tstruct(bin_edges=hl.tarray(hl.tfloat64),
                               bin_freq=hl.tarray(hl.tint64),
                               n_smaller=hl.tint64,
                               n_larger=hl.tint64)

        take_by_sig = ir.AggSignature('TakeBy', [hl.tint32], None, [hl.tfloat64, hl.tfloat64])
        take_by_type = hl.tarray(hl.tfloat64)

        value_irs = [
            i, ir.I64(5), ir.F32(3.14), ir.F64(3.14), s, ir.TrueIR(), ir.FalseIR(), ir.Void(),
            ir.Cast(i, hl.tfloat64),
            ir.NA(hl.tint32),
            ir.IsNA(i),
            ir.If(b, i, j),
            ir.Let('v', i, v),
            ir.Ref('x', hl.tint32),
            ir.ApplyBinaryOp('+', i, j),
            ir.ApplyUnaryOp('-', i),
            ir.ApplyComparisonOp('EQ', i, j),
            ir.MakeArray([i, ir.NA(hl.tint32), ir.I32(-3)], hl.tarray(hl.tint32)),
            ir.ArrayRef(a, i),
            ir.ArrayLen(a),
            ir.ArrayRange(ir.I32(0), ir.I32(5), ir.I32(1)),
            ir.ArraySort(a, b, False),
            ir.ToSet(a),
            ir.ToDict(da),
            ir.ToArray(a),
            ir.LowerBoundOnOrderedCollection(a, i, True),
            ir.GroupByKey(da),
            ir.ArrayMap(a, 'v', v),
            ir.ArrayFilter(a, 'v', v),
            ir.ArrayFlatMap(aa, 'v', v),
            ir.ArrayFold(a, ir.I32(0), 'x', 'v', v),
            ir.ArrayScan(a, ir.I32(0), 'x', 'v', v),
            ir.ArrayFor(a, 'v', ir.Void()),
            ir.AggFilter(ir.TrueIR(), ir.I32(0)),
            ir.AggExplode(ir.ArrayRange(ir.I32(0), ir.I32(2), ir.I32(1)), 'x', ir.I32(0)),
            ir.AggGroupBy(ir.TrueIR(), ir.I32(0)),
            ir.ApplyAggOp([ir.I32(0)], [], None, collect_sig, hl.tarray(hl.tint32)),
            ir.ApplyScanOp([ir.I32(0)], [], None, collect_sig, hl.tarray(hl.tint32)),
            ir.ApplyAggOp(
                [ir.F64(-2.11)], [ir.F64(-5.0), ir.F64(5.0), ir.I32(100)], None, hist_sig, hist_type),
            ir.ApplyAggOp([call], [], [ir.I32(2)], call_stats_sig, call_stats_type),
            ir.ApplyAggOp([ir.F64(-2.11), ir.F64(-2.11)], [ir.I32(10)], None, take_by_sig, take_by_type),
            ir.InitOp(ir.I32(0), [ir.I32(2)], call_stats_sig),
            ir.SeqOp(ir.I32(0), [i], collect_sig),
            ir.SeqOp(ir.I32(0), [ir.F64(-2.11), ir.I32(17)], take_by_sig),
            ir.Begin([ir.Void()]),
            ir.MakeStruct([('x', i)]),
            ir.SelectFields(s, ['x', 'z']),
            ir.InsertFields(s, [('x', i)]),
            ir.GetField(s, 'x'),
            ir.MakeTuple([i, b]),
            ir.GetTupleElement(t, 1),
            ir.StringSlice(st, ir.I32(1), ir.I32(2)),
            ir.StringLength(st),
            ir.In(2, hl.tfloat64),
            ir.Die('mumblefoo', hl.tfloat64),
            ir.Apply('&&', b, c),
            ir.Apply('toFloat64', i),
            ir.Uniroot('x', ir.F64(3.14), ir.F64(-5.0), ir.F64(5.0)),
            ir.Literal(hl.tarray(hl.tint32), [1, 2, None]),
        ]

        return value_irs

    def test_parses(self):
        for x in self.value_irs():
            Env.hail().expr.Parser.parse_value_ir(str(x))

    def test_copies(self):
        for x in self.value_irs():
            self.assertEqual(x, x.copy(*x.children))


class TableIRTests(unittest.TestCase):

    def table_irs(self):
        b = ir.TrueIR()
        table_read = ir.TableRead(
            resource('backward_compatability/1.0.0/table/0.ht'), False, None)
        table_read_row_type = hl.dtype('struct{idx: int32, f32: float32, i64: int64, m: float64, astruct: struct{a: int32, b: float64}, mstruct: struct{x: int32, y: str}, aset: set<str>, mset: set<float64>, d: dict<array<str>, float64>, md: dict<int32, str>, h38: locus<GRCh38>, ml: locus<GRCh37>, i: interval<locus<GRCh37>>, c: call, mc: call, t: tuple(call, str, str), mt: tuple(locus<GRCh37>, bool)}')

        matrix_read = ir.MatrixRead(
            resource('backward_compatability/1.0.0/matrix_table/0.hmt'), False, False)

        range = ir.TableRange(10, 4)
        table_irs = [
            ir.TableKeyBy(table_read, ['m', 'd'], False),
            ir.TableFilter(table_read, b),
            table_read,
            ir.MatrixColsTable(matrix_read),
            ir.TableAggregateByKey(
                table_read,
                ir.MakeStruct([('a', ir.I32(5))])),
            ir.TableKeyByAndAggregate(
                table_read,
                ir.MakeStruct([('a', ir.I32(5))]),
                ir.MakeStruct([('b', ir.I32(5))]),
                1, 2),
            ir.TableJoin(
                table_read,
                ir.TableRange(100, 10), 'inner', 1),
            ir.MatrixEntriesTable(matrix_read),
            ir.MatrixRowsTable(matrix_read),
            ir.TableParallelize(ir.Literal(hl.tarray(hl.tstruct(a=hl.tint32)), [{'a':None}, {'a':5}, {'a':-3}]), None),
            ir.TableMapRows(
                ir.TableKeyBy(table_read, []),
                ir.MakeStruct([
                    ('a', ir.GetField(ir.Ref('row', table_read_row_type), 'f32')),
                    ('b', ir.F64(-2.11))])),
            ir.TableMapGlobals(
                table_read,
                ir.MakeStruct([
                    ('foo', ir.NA(hl.tarray(hl.tint32)))])),
            ir.TableRange(100, 10),
            ir.TableRepartition(table_read, 10, False),
            ir.TableUnion(
                [ir.TableRange(100, 10), ir.TableRange(50, 10)]),
            ir.TableExplode(table_read, 'mset'),
            ir.TableHead(table_read, 10),
            ir.TableOrderBy(ir.TableKeyBy(table_read, []), [('m', 'A'), ('m', 'D')]),
            ir.TableDistinct(table_read),
            ir.LocalizeEntries(matrix_read, '__entries'),
            ir.TableRename(table_read, {'idx': 'idx_foo'}, {'global_f32': 'global_foo'})
        ]

        return table_irs

    def test_parses(self):
        for x in self.table_irs():
            Env.hail().expr.Parser.parse_table_ir(str(x))

    def test_matrix_ir_parses(self):
        hl.index_bgen(resource('example.8bits.bgen'),
                      reference_genome=hail.get_reference('GRCh37'),
                      contig_recoding={'01': '1'})

        collect_sig = ir.AggSignature('Collect', [], None, [hl.tint32])
        collect = ir.MakeStruct([('x', ir.ApplyAggOp([ir.I32(0)], [], None, collect_sig, hl.tarray(hl.tint32)))])

        matrix_read = ir.MatrixRead(
            resource('backward_compatability/1.0.0/matrix_table/0.hmt'), False, False)
        table_read = ir.TableRead(resource('backward_compatability/1.0.0/table/0.ht'), False, None)

        matrix_irs = [
            ir.MatrixUnionRows(ir.MatrixRange(5, 5, 1), ir.MatrixRange(5, 5, 1)),
            ir.UnlocalizeEntries(
                ir.LocalizeEntries(matrix_read, '__entries'),
                ir.MatrixColsTable(matrix_read),
                '__entries'),
            ir.MatrixAggregateRowsByKey(matrix_read, collect, collect),
            ir.MatrixAggregateColsByKey(matrix_read, collect, collect),
            ir.MatrixRange(1, 1, 10),
            ir.MatrixImportVCF([resource('sample.vcf')], False, False, None, None, False, ['GT'],
                               hail.get_reference('GRCh37'), {}, True, False),
            ir.MatrixImportBGEN([resource('example.8bits.bgen')], ['GP'], resource('example.sample'), {}, 10, 1,
                                ['varid'], None),
            ir.MatrixFilterRows(matrix_read, ir.FalseIR()),
            ir.MatrixFilterCols(matrix_read, ir.FalseIR()),
            ir.MatrixFilterEntries(matrix_read, ir.FalseIR()),
            ir.MatrixChooseCols(matrix_read, [1, 0]),
            ir.MatrixMapCols(matrix_read, ir.MakeStruct([('x', ir.I64(20))]), ['x']),
            ir.MatrixKeyRowsBy(matrix_read, ['row_i64'], False),
            ir.MatrixMapRows(ir.MatrixKeyRowsBy(matrix_read, []), ir.MakeStruct([('x', ir.I64(20))])),
            ir.MatrixMapEntries(matrix_read, ir.MakeStruct([('x', ir.I64(20))])),
            ir.MatrixMapGlobals(matrix_read, ir.MakeStruct([('x', ir.I64(20))])),
            ir.TableToMatrixTable(table_read, ['f32', 'i64'], ['m', 'astruct'], ['aset'], ['mset'], 100),
            ir.MatrixCollectColsByKey(matrix_read),
            ir.MatrixExplodeRows(matrix_read, ['row_aset']),
            ir.MatrixExplodeCols(matrix_read, ['col_aset']),
            ir.MatrixAnnotateRowsTable(matrix_read, table_read, '__foo', None),
            ir.MatrixAnnotateColsTable(matrix_read, table_read, '__foo'),
        ]

        for x in matrix_irs:
            try:
                Env.hail().expr.Parser.parse_matrix_ir(str(x))
            except Exception as e:
                raise ValueError(str(x)) from e


class ValueTests(unittest.TestCase):

    def values(self):
        values = [
            (hl.tbool, True),
            (hl.tint32, 0),
            (hl.tint64, 0),
            (hl.tfloat32, 0.5),
            (hl.tfloat64, 0.5),
            (hl.tstr, "foo"),
            (hl.tstruct(x=hl.tint32), hl.Struct(x=0)),
            (hl.tarray(hl.tint32), [0, 1, 4]),
            (hl.tset(hl.tint32), {0, 1, 4}),
            (hl.tdict(hl.tstr, hl.tint32), {"a": 0, "b": 1, "c": 4}),
            (hl.tinterval(hl.tint32), hl.Interval(0, 1, True, False)),
            (hl.tlocus(hl.default_reference()), hl.Locus("1", 1)),
            (hl.tcall, hl.Call([0, 1]))
        ]
        return values

    def test_value_same_after_parsing(self):
        for t, v in self.values():
            row_v = ir.Literal(t, v)
            map_globals_ir = ir.TableMapGlobals(
                ir.TableRange(1, 1),
                ir.InsertFields(
                    ir.Ref("global", hl.tstruct()),
                    [("foo", row_v)]))
            new_globals = hl.eval(hl.Table._from_ir(map_globals_ir).globals)
            self.assertEquals(new_globals, hl.Struct(foo=v))
