import unittest
import hail as hl
import hail.ir as ir
from hail.utils.java import Env

class IRTests(unittest.TestCase):
    def test_ir_str(self):
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
        
        ir_examples = [
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
            # ApplyAggOp, InitOp, SeqOp
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
        
        for x in ir_examples:
            s = str(x)
            print(s)
            Env.hail().expr.Parser.parse_value_ir(s)
