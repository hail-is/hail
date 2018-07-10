import unittest
import hail as hl
import hail.ir as ir
from hail.utils.java import Env
from .helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

class IRTests(unittest.TestCase):
    def test_value_ir_parses(self):
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
        
        hist_sig = ir.AggSignature(
            'Histogram', [hl.tfloat64, hl.tfloat64, hl.tint32], None, [hl.tfloat64])
        
        take_by_sig = ir.AggSignature('TakeBy', [hl.tint32], None, [hl.tfloat64, hl.tfloat64])
        
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
            ir.ApplyComparisonOp(ir.ComparisonOp('EQ', hl.tint32), i, j),
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
            ir.ArrayFor(a, 'v', ir.Void()),
            ir.ApplyAggOp(ir.I32(0), [], None, collect_sig),
            ir.ApplyAggOp(
                ir.F64(-2.11), [ir.F64(-5.0), ir.F64(5.0), ir.I32(100)], None, hist_sig),
            ir.ApplyAggOp(call, [], [ir.I32(2)], call_stats_sig),
            ir.ApplyAggOp(ir.F64(-2.11), [ir.I32(10)], None, take_by_sig),
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
            ir.Apply('&&', [b, c]),
            ir.Apply('toFloat64', [i]),
            ir.Apply('isDefined', [s]),
            ir.Uniroot('x', ir.F64(3.14), ir.F64(-5.0), ir.F64(5.0))
        ]
        
        for x in value_irs:
            Env.hail().expr.Parser.parse_value_ir(str(x))

    def test_table_ir_parses(self):
        b = ir.TrueIR()
        table_read = ir.TableRead(
            'src/test/resources/backward_compatability/1.0.0/table/0.ht', False, None)
        table_read_row_type = hl.dtype('struct{idx: int32, f32: float32, i64: int64, m: float64, astruct: struct{a: int32, b: float64}, mstruct: struct{x: int32, y: str}, aset: set<str>, mset: set<float64>, d: dict<array<str>, float64>, md: dict<int32, str>, h38: locus<GRCh38>, ml: locus<GRCh37>, i: interval<locus<GRCh37>>, c: call, mc: call, t: tuple(call, str, str), mt: tuple(locus<GRCh37>, bool)}')

        matrix_read = ir.MatrixRead(
            'src/test/resources/backward_compatability/1.0.0/matrix_table/0.hmt', False, False)

        range = ir.TableRange(10, 4)
        table_irs = [
            ir.TableUnkey(table_read),
            ir.TableKeyBy(table_read, ['m', 'd'], 1, True),
            ir.TableFilter(table_read, b),
            table_read,
            ir.MatrixColsTable(matrix_read),
            ir.TableAggregateByKey(
                table_read,
                ir.MakeStruct([('a', ir.I32(5))])),
            ir.TableJoin(
                table_read,
                ir.TableRange(100, 10), 'inner'),
            ir.MatrixEntriesTable(matrix_read),
            ir.MatrixRowsTable(matrix_read),
            ir.TableParallelize(
                'Table{global:Struct{},key:None,row:Struct{a:Int32}}',
                ir.Value(hl.tarray(hl.tstruct(a=hl.tint32)), [{'a':None}, {'a':5}, {'a':-3}]),
                None),
            ir.TableMapRows(
                table_read,
                ir.MakeStruct([
                    ('a', ir.GetField(ir.Ref('row', table_read_row_type), 'f32')),
                    ('b', ir.F64(-2.11))]),
                None, None),
            ir.TableMapGlobals(
                table_read,
                ir.MakeStruct([
                    ('foo', ir.NA(hl.tarray(hl.tint32)))]),
                ir.Value(hl.tstruct(), {})),
            ir.TableRange(100, 10),
            ir.TableUnion(
                [ir.TableRange(100, 10), ir.TableRange(50, 10)]),
            ir.TableExplode(table_read, 'mset'),
            ir.TableOrderBy(ir.TableUnkey(table_read), [('m', 'A'), ('m', 'D')]),
            ir.TableDistinct(table_read),
        ]

        for x in table_irs:
            Env.hail().expr.Parser.parse_table_ir(str(x))
