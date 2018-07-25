import unittest

import hail as hl
from hail.ir import *
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

a = MakeArray([I32(1), I32(3)], hl.tint32)
t = MakeTuple([I32(0), I32(1)])
struct = MakeStruct([("a", I32(0)), ("b", Str("hello"))])
tuplea = MakeArray([t, t], hl.ttuple(hl.tint64, hl.tint64))
agg_sig = AggSignature("Count", [], [], [hl.tint32])
tir = TableIR()
mir = MatrixIR()
agg_query = MakeStruct([("a", ApplyAggOp(SeqOp(I32(0), [I32(1)], agg_sig), [], None, agg_sig))])

irs = [
    I32(5),
    I64(4),
    F32(3.0),
    F64(5.0),
    Str("hello"),
    FalseIR(),
    TrueIR(),
    Void(),
    Cast(I32(0), hl.tint64),
    NA(hl.tint32),
    IsNA(I32(0)),
    If(TrueIR(), I32(4), I32(0)),
    Let("foo", I32(3), Str("hello")),
    Ref("foo", hl.tint64),
    ApplyBinaryOp('+', I32(5), I32(3)),
    ApplyUnaryOp('-', I32(3)),
    ApplyComparisonOp('>', I32(1), I32(3)),
    a,
    ArrayRef(a, I32(0)),
    ArrayLen(a),
    ArrayRange(I32(0), I32(5), I32(1)),
    ArraySort(a, TrueIR(), True),
    ToSet(a),
    ToDict(tuplea),
    ToArray(a),
    LowerBoundOnOrderedCollection(a, I32(2), True),
    GroupByKey(tuplea),
    ArrayMap(a, "x", I32(5)),
    ArrayFilter(a, "x", TrueIR()),
    ArrayFlatMap(a, "x", a),
    ArrayFold(a, I32(0), "accum", "value", I32(1)),
    ArrayFor(a, "value", I32(1)),
    ApplyAggOp(SeqOp(I32(0), [I32(1)], agg_sig), [], None, agg_sig),
    InitOp(I32(0), [I32(1)], agg_sig),
    SeqOp(I32(0), [I32(3)], agg_sig),
    Begin([a, t, tuplea]),
    struct,
    SelectFields(struct, ["a"]),
    InsertFields(struct, [("c", I32(0)), ("d", Str("hello"))]),
    GetField(struct, "a"),
    t,
    GetTupleElement(t, 0),
    StringSlice(Str("hello"), I32(0), I32(2)),
    StringLength(Str("hello")),
    In(5, hl.tint32),
    Die("goodbye", hl.tint32),
    Apply("fet", I32(0), I32(5), I32(10), I32(1)),
    Uniroot("x", ApplyBinaryOp('+', Ref("a", hl.tint32), Ref("b", hl.tint32)), I32(0), I32(4)),
    TableCount(tir),
    TableAggregate(tir, agg_query),
    MatrixAggregate(mir, agg_query),
    TableWrite(tir, "/home/foo", True),
    TableExport(tir, "/home/foo.tsv", "foo.tsv", True, hl.tstruct(a=hl.tint32)),
    MatrixWrite(mir, "matrix_writer_string")
]

class Tests(unittest.TestCase):
    def test_copy(self):
        for ir in irs:
            self.assertEqual(ir, ir.copy(*ir.children))
