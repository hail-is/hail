import functools
import random
import re
import unittest

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import hail as hl
from hail import ir
from hail.expr import construct_expr
from hail.expr.types import tint32
from hail.ir.renderer import CSERenderer
from hail.utils import new_temp_file
from hail.utils.java import Env
from test.hail.helpers import resource, skip_unless_spark_backend


def value_irs_env():
    return {
        'c': hl.tbool,
        'a': hl.tarray(hl.tint32),
        'st': hl.tstream(hl.tint32),
        'whitenStream': hl.tstream(
            hl.tstruct(prevWindow=hl.tndarray(hl.tfloat64, 2), newChunk=hl.tndarray(hl.tfloat64, 2))
        ),
        'mat': hl.tndarray(hl.tfloat64, 2),
        'aa': hl.tarray(hl.tarray(hl.tint32)),
        'sta': hl.tstream(hl.tarray(hl.tint32)),
        'sts': hl.tstream(hl.tstruct(x=hl.tint32, y=hl.tint64, z=hl.tfloat64)),
        'da': hl.tstream(hl.ttuple(hl.tint32, hl.tstr)),
        'nd': hl.tndarray(hl.tfloat64, 2),
        'v': hl.tint32,
        's': hl.tstruct(x=hl.tint32, y=hl.tint64, z=hl.tfloat64),
        't': hl.ttuple(hl.tint32, hl.tint64, hl.tfloat64),
        'call': hl.tcall,
        'x': hl.tint32,
    }


@pytest.fixture(scope='module')
def value_irs():
    env = value_irs_env()
    b = ir.TrueIR()
    c = ir.Ref('c', env['c'])
    i = ir.I32(5)
    j = ir.I32(7)
    a = ir.Ref('a', env['a'])
    st = ir.Ref('st', env['st'])
    whitenStream = ir.Ref('whitenStream', env['whitenStream'])
    sta = ir.Ref('sta', env['sta'])
    sts = ir.Ref('sts', env['sts'])
    da = ir.Ref('da', env['da'])
    nd = ir.Ref('nd', env['nd'])
    v = ir.Ref('v', env['v'])
    s = ir.Ref('s', env['s'])
    t = ir.Ref('t', env['t'])
    ir.Ref('call', env['call'])
    rngState = ir.RNGStateLiteral()

    table = ir.TableRange(5, 3)

    matrix_read = ir.MatrixRead(
        ir.MatrixNativeReader(resource('backward_compatability/1.0.0/matrix_table/0.hmt'), None, False),
        False,
        False,
    )

    block_matrix_read = ir.BlockMatrixRead(ir.BlockMatrixNativeReader(resource('blockmatrix_example/0')))

    def aggregate(x):
        return ir.TableAggregate(table, x)

    return [
        i,
        ir.I64(5),
        ir.F32(3.14),
        ir.F64(3.14),
        s,
        ir.TrueIR(),
        ir.FalseIR(),
        ir.Void(),
        ir.Cast(i, hl.tfloat64),
        ir.NA(hl.tint32),
        ir.IsNA(i),
        ir.If(b, i, j),
        ir.Coalesce(i, j),
        ir.Let('v', i, v),
        ir.Ref('x', env['x']),
        ir.ApplyBinaryPrimOp('+', i, j),
        ir.ApplyUnaryPrimOp('-', i),
        ir.ApplyComparisonOp('EQ', i, j),
        ir.MakeArray([i, ir.NA(hl.tint32), ir.I32(-3)], hl.tarray(hl.tint32)),
        ir.ArrayRef(a, i),
        ir.ArrayLen(a),
        ir.ArraySort(
            ir.ToStream(a), 'l', 'r', ir.ApplyComparisonOp("LT", ir.Ref('l', hl.tint32), ir.Ref('r', hl.tint32))
        ),
        ir.ToSet(st),
        ir.ToDict(da),
        ir.ToArray(st),
        ir.CastToArray(ir.NA(hl.tset(hl.tint32))),
        ir.MakeNDArray(
            ir.MakeArray([ir.F64(-1.0), ir.F64(1.0)], hl.tarray(hl.tfloat64)),
            ir.MakeTuple([ir.I64(1), ir.I64(2)]),
            ir.TrueIR(),
        ),
        ir.NDArrayShape(nd),
        ir.NDArrayReshape(nd, ir.MakeTuple([ir.I64(5)])),
        ir.NDArrayRef(nd, [ir.I64(1), ir.I64(2)]),
        ir.NDArrayMap(nd, 'unused', v),
        ir.NDArrayMatMul(nd, nd),
        ir.LowerBoundOnOrderedCollection(a, i, False),
        ir.GroupByKey(da),
        ir.RNGSplit(rngState, ir.MakeTuple([ir.I64(1), ir.I64(2), ir.I64(3)])),
        ir.StreamMap(st, 'unused', v),
        ir.StreamZip([st, st], ['a', 'b'], ir.TrueIR(), 'ExtendNA'),
        ir.StreamFilter(st, 'v', c),
        ir.StreamFlatMap(sta, 'unused', ir.ToStream(a)),
        ir.StreamFold(st, ir.I32(0), 'x', 'v', v),
        ir.StreamScan(st, ir.I32(0), 'x', 'v', v),
        ir.StreamWhiten(whitenStream, "newChunk", "prevWindow", 1, 1, 1, 1, False),
        ir.StreamJoinRightDistinct(sts, sts, ['x'], ['x'], 'l', 'r', ir.I32(1), "left"),
        ir.StreamFor(st, 'v', ir.Void()),
        aggregate(ir.AggFilter(ir.TrueIR(), ir.I32(0), False)),
        aggregate(ir.AggExplode(ir.StreamRange(ir.I32(0), ir.I32(2), ir.I32(1)), 'x', ir.I32(0), False)),
        aggregate(ir.AggGroupBy(ir.TrueIR(), ir.I32(0), False)),
        aggregate(
            ir.AggArrayPerElement(
                ir.ToArray(ir.StreamRange(ir.I32(0), ir.I32(2), ir.I32(1))), 'x', 'y', ir.I32(0), False
            )
        ),
        aggregate(ir.ApplyAggOp('Collect', [], [ir.I32(0)])),
        aggregate(ir.ApplyAggOp('CallStats', [ir.I32(2)], [ir.NA(hl.tcall)])),
        aggregate(ir.ApplyAggOp('TakeBy', [ir.I32(10)], [ir.F64(-2.11), ir.F64(-2.11)])),
        ir.Begin([ir.Void()]),
        ir.MakeStruct([('x', i)]),
        ir.SelectFields(s, ['x', 'z']),
        ir.InsertFields(s, [('x', i)], None),
        ir.GetField(s, 'x'),
        ir.MakeTuple([i, b]),
        ir.GetTupleElement(t, 1),
        ir.Die(ir.Str('mumblefoo'), hl.tfloat64),
        ir.Apply('land', hl.tbool, b, c),
        ir.Apply('toFloat64', hl.tfloat64, i),
        ir.Literal(hl.tarray(hl.tint32), [1, 2, None]),
        ir.EncodedLiteral(hl.tarray(hl.tint32), [1, 2, None]),
        ir.TableCount(table),
        ir.TableGetGlobals(table),
        ir.TableCollect(ir.TableKeyBy(table, [], False)),
        ir.TableToValueApply(table, {'name': 'ForceCountTable'}),
        ir.MatrixToValueApply(matrix_read, {'name': 'ForceCountMatrixTable'}),
        ir.TableAggregate(table, ir.MakeStruct([('foo', ir.ApplyAggOp('Collect', [], [ir.I32(0)]))])),
        ir.TableWrite(table, ir.TableNativeWriter(new_temp_file(), False, True, "fake_codec_spec$$")),
        ir.TableWrite(table, ir.TableTextWriter(new_temp_file(), None, True, "concatenated", ",")),
        ir.MatrixAggregate(matrix_read, ir.MakeStruct([('foo', ir.ApplyAggOp('Collect', [], [ir.I32(0)]))])),
        ir.MatrixWrite(matrix_read, ir.MatrixNativeWriter(new_temp_file(), False, False, "", None, None)),
        ir.MatrixWrite(
            matrix_read,
            ir.MatrixNativeWriter(
                new_temp_file(),
                False,
                False,
                "",
                '[{"start":{"row_idx":0},"end":{"row_idx": 10},"includeStart":true,"includeEnd":false}]',
                hl.dtype('array<interval<struct{row_idx:int32}>>'),
            ),
        ),
        ir.MatrixWrite(matrix_read, ir.MatrixVCFWriter(new_temp_file(), None, ir.ExportType.CONCATENATED, None, False)),
        ir.MatrixWrite(matrix_read, ir.MatrixGENWriter(new_temp_file(), 4)),
        ir.MatrixWrite(matrix_read, ir.MatrixPLINKWriter(new_temp_file())),
        ir.MatrixMultiWrite(
            [matrix_read, matrix_read],
            ir.MatrixNativeMultiWriter([new_temp_file(), new_temp_file()], False, False, None),
        ),
        ir.BlockMatrixWrite(block_matrix_read, ir.BlockMatrixNativeWriter('fake.bm', False, False, False)),
        ir.LiftMeOut(ir.I32(1)),
        ir.BlockMatrixWrite(block_matrix_read, ir.BlockMatrixPersistWriter('x', 'MEMORY_ONLY')),
    ]


static_len_value_irs = 81


@pytest.fixture(params=range(static_len_value_irs))
def value_ir(value_irs, request):
    assert len(value_irs) == static_len_value_irs
    return value_irs[request.param]


def test_ir_value_type(value_ir):
    env = value_irs_env()
    typ = Env.backend().value_type(
        functools.reduce(
            lambda res, binding: ir.Let(binding[0], ir.NA(binding[1]), res),
            reversed([(k, env[k]) for k in value_ir.free_vars]),
            value_ir,
        )
    )
    assert value_ir.typ == typ


def test_ir_copies(value_ir):
    cp = value_ir.copy(*value_ir.children)
    assert value_ir == cp
    assert hash(value_ir) == hash(cp)


@pytest.fixture(scope='module')
def table_irs():
    b = ir.TrueIR()
    table_read = ir.TableRead(
        ir.TableNativeReader(resource('backward_compatability/1.1.0/table/0.ht'), None, False), False
    )
    table_read_row_type = hl.dtype(
        'struct{idx: int32, f32: float32, i64: int64, m: float64, astruct: struct{a: int32, b: float64}, mstruct: struct{x: int32, y: str}, aset: set<str>, mset: set<float64>, d: dict<array<str>, float64>, md: dict<int32, str>, h38: locus<GRCh38>, ml: locus<GRCh37>, i: interval<locus<GRCh37>>, c: call, mc: call, t: tuple(call, str, str), mt: tuple(locus<GRCh37>, bool)}'
    )

    matrix_read = ir.MatrixRead(
        ir.MatrixNativeReader(resource('backward_compatability/1.0.0/matrix_table/0.hmt'), None, False),
        False,
        False,
    )

    block_matrix_read = ir.BlockMatrixRead(ir.BlockMatrixNativeReader(resource('blockmatrix_example/0')))

    aa = hl.literal([[0.00], [0.01], [0.02]])._ir

    return [
        ir.TableKeyBy(table_read, ['m', 'd'], False),
        ir.TableFilter(table_read, b),
        table_read,
        ir.MatrixColsTable(matrix_read),
        ir.TableAggregateByKey(table_read, ir.MakeStruct([('a', ir.I32(5))])),
        ir.TableKeyByAndAggregate(
            table_read, ir.MakeStruct([('a', ir.I32(5))]), ir.MakeStruct([('b', ir.I32(5))]), 1, 2
        ),
        ir.TableJoin(table_read, ir.TableRange(100, 10), 'inner', 1),
        ir.MatrixEntriesTable(matrix_read),
        ir.MatrixRowsTable(matrix_read),
        ir.TableParallelize(
            ir.MakeStruct([
                ('rows', ir.Literal(hl.tarray(hl.tstruct(a=hl.tint32)), [{'a': None}, {'a': 5}, {'a': -3}])),
                ('global', ir.MakeStruct([])),
            ]),
            None,
        ),
        ir.TableMapRows(
            ir.TableKeyBy(table_read, []),
            ir.MakeStruct([
                ('a', ir.GetField(ir.Ref('row', table_read_row_type), 'f32')),
                ('b', ir.F64(-2.11)),
                ('c', ir.ApplyScanOp('Collect', [], [ir.I32(0)])),
            ]),
        ),
        ir.TableMapGlobals(table_read, ir.MakeStruct([('foo', ir.NA(hl.tarray(hl.tint32)))])),
        ir.TableRange(100, 10),
        ir.TableRepartition(table_read, 10, ir.RepartitionStrategy.COALESCE),
        ir.TableUnion([ir.TableRange(100, 10), ir.TableRange(50, 10)]),
        ir.TableExplode(table_read, ['mset']),
        ir.TableHead(table_read, 10),
        ir.TableOrderBy(ir.TableKeyBy(table_read, []), [('m', 'A'), ('m', 'D')]),
        ir.TableDistinct(table_read),
        ir.CastMatrixToTable(matrix_read, '__entries', '__cols'),
        ir.TableRename(table_read, {'idx': 'idx_foo'}, {'global_f32': 'global_foo'}),
        ir.TableMultiWayZipJoin([table_read, table_read], '__data', '__globals'),
        ir.MatrixToTableApply(
            matrix_read,
            {
                'name': 'LinearRegressionRowsSingle',
                'yFields': ['col_m'],
                'xField': 'entry_m',
                'covFields': [],
                'rowBlockSize': 10,
                'passThrough': [],
            },
        ),
        ir.TableToTableApply(table_read, {'name': 'TableFilterPartitions', 'parts': [0], 'keep': True}),
        ir.BlockMatrixToTableApply(block_matrix_read, aa, {'name': 'PCRelate', 'maf': 0.01, 'blockSize': 4096}),
        ir.TableFilterIntervals(
            table_read,
            [hl.utils.Interval(hl.utils.Struct(row_idx=0), hl.utils.Struct(row_idx=10))],
            hl.tstruct(row_idx=hl.tint32),
            keep=False,
        ),
        ir.TableMapPartitions(table_read, 'glob', 'rows', ir.Ref('rows', hl.tstream(table_read_row_type)), 0, 1),
        ir.TableGen(
            contexts=ir.StreamRange(ir.I32(0), ir.I32(10), ir.I32(1)),
            globals=ir.MakeStruct([]),
            cname="contexts",
            gname="globals",
            body=ir.ToStream(ir.MakeArray([ir.MakeStruct([('a', ir.I32(1))])], type=None)),
            partitioner=ir.Partitioner(
                hl.tstruct(a=hl.tint), [hl.Interval(hl.Struct(a=1), hl.Struct(a=2), True, True)]
            ),
        ),
    ]


static_len_table_irs = 28


@pytest.fixture(params=range(static_len_table_irs))
def table_ir(table_irs, request):
    assert len(table_irs) == static_len_table_irs
    return table_irs[request.param]


def test_table_ir_table_type(table_ir):
    typ = Env.backend().table_type(table_ir)
    assert table_ir.typ == typ


def test_table_ir_highly_nested():
    N = 10
    M = 250
    ht = hl.utils.range_table(N)
    for i in range(M):
        ht = ht.annotate(**{f'x{i}': i})
    str(ht._tir)

    # TODO: Scala Pretty errors out with a StackOverflowError here
    # ht._force_count()


@pytest.fixture(scope='module')
def matrix_irs():
    collect = ir.MakeStruct([('x', ir.ApplyAggOp('Collect', [], [ir.I32(0)]))])

    matrix_read = ir.MatrixRead(
        ir.MatrixNativeReader(resource('backward_compatability/1.0.0/matrix_table/0.hmt'), None, False),
        False,
        False,
    )
    table_read = ir.TableRead(
        ir.TableNativeReader(resource('backward_compatability/1.1.0/table/0.ht'), None, False), False
    )

    matrix_range = ir.MatrixRead(ir.MatrixRangeReader(1, 1, 10))
    return [
        ir.MatrixRepartition(matrix_range, 100, ir.RepartitionStrategy.SHUFFLE),
        ir.MatrixUnionRows(matrix_range, matrix_range),
        ir.MatrixDistinctByRow(matrix_range),
        ir.MatrixRowsHead(matrix_read, 5),
        ir.MatrixColsHead(matrix_read, 5),
        ir.CastTableToMatrix(ir.CastMatrixToTable(matrix_read, '__entries', '__cols'), '__entries', '__cols', []),
        ir.MatrixAggregateRowsByKey(matrix_read, collect, collect),
        ir.MatrixAggregateColsByKey(matrix_read, collect, collect),
        matrix_read,
        matrix_range,
        ir.MatrixRead(
            ir.MatrixVCFReader(
                resource('sample.vcf'),
                ['GT'],
                hl.tfloat64,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                True,
                False,
                True,
                None,
                None,
            )
        ),
        ir.MatrixRead(ir.MatrixBGENReader(resource('example.8bits.bgen'), None, {}, 10, 1, None)),
        ir.MatrixFilterRows(matrix_read, ir.FalseIR()),
        ir.MatrixFilterCols(matrix_read, ir.FalseIR()),
        ir.MatrixFilterEntries(matrix_read, ir.FalseIR()),
        ir.MatrixChooseCols(matrix_read, [1, 0]),
        ir.MatrixMapCols(matrix_read, ir.MakeStruct([('x', ir.I64(20))]), ['x']),
        ir.MatrixKeyRowsBy(matrix_read, ['row_i64'], False),
        ir.MatrixMapRows(ir.MatrixKeyRowsBy(matrix_read, []), ir.MakeStruct([('x', ir.I64(20))])),
        ir.MatrixMapEntries(matrix_read, ir.MakeStruct([('x', ir.I64(20))])),
        ir.MatrixMapGlobals(matrix_read, ir.MakeStruct([('x', ir.I64(20))])),
        ir.MatrixCollectColsByKey(matrix_read),
        ir.MatrixExplodeRows(matrix_read, ['row_aset']),
        ir.MatrixExplodeCols(matrix_read, ['col_aset']),
        ir.MatrixAnnotateRowsTable(matrix_read, table_read, '__foo'),
        ir.MatrixAnnotateColsTable(matrix_read, table_read, '__foo'),
        ir.MatrixToMatrixApply(matrix_read, {'name': 'MatrixFilterPartitions', 'parts': [0], 'keep': True}),
        ir.MatrixRename(
            matrix_read,
            {'global_f32': 'global_foo'},
            {'col_f32': 'col_foo'},
            {'row_aset': 'row_aset2'},
            {'entry_f32': 'entry_foo'},
        ),
        ir.MatrixFilterIntervals(
            matrix_read,
            [hl.utils.Interval(hl.utils.Struct(row_idx=0), hl.utils.Struct(row_idx=10))],
            hl.tstruct(row_idx=hl.tint32),
            keep=False,
        ),
    ]


static_len_matrix_irs = 29


@pytest.fixture(params=range(static_len_matrix_irs))
def matrix_ir(matrix_irs, request):
    assert len(matrix_irs) == static_len_matrix_irs
    return matrix_irs[request.param]


def test_matrix_ir_matrix_type(matrix_ir):
    typ = Env.backend().matrix_type(matrix_ir)
    assert typ == matrix_ir.typ


@pytest.fixture(scope='module')
def blockmatrix_irs():
    scalar_ir = ir.F64(2)
    vector_ir = ir.MakeArray([ir.F64(3), ir.F64(2)], hl.tarray(hl.tfloat64))

    read = ir.BlockMatrixRead(ir.BlockMatrixNativeReader(resource('blockmatrix_example/0')))
    add_two_bms = ir.BlockMatrixMap2(
        read, read, 'l', 'r', ir.ApplyBinaryPrimOp('+', ir.Ref('l', hl.tfloat64), ir.Ref('r', hl.tfloat64)), "Union"
    )
    negate_bm = ir.BlockMatrixMap(read, 'element', ir.ApplyUnaryPrimOp('-', ir.Ref('element', hl.tfloat64)), False)
    sqrt_bm = ir.BlockMatrixMap(
        read, 'element', hl.sqrt(construct_expr(ir.Ref('element', hl.tfloat64), hl.tfloat64))._ir, False
    )

    scalar_to_bm = ir.ValueToBlockMatrix(scalar_ir, [1, 1], 1)
    col_vector_to_bm = ir.ValueToBlockMatrix(vector_ir, [2, 1], 1)
    row_vector_to_bm = ir.ValueToBlockMatrix(vector_ir, [1, 2], 1)
    broadcast_scalar = ir.BlockMatrixBroadcast(scalar_to_bm, [], [2, 2], 256)
    broadcast_col = ir.BlockMatrixBroadcast(col_vector_to_bm, [0], [2, 2], 256)
    broadcast_row = ir.BlockMatrixBroadcast(row_vector_to_bm, [1], [2, 2], 256)
    transpose = ir.BlockMatrixBroadcast(broadcast_scalar, [1, 0], [2, 2], 256)
    matmul = ir.BlockMatrixDot(broadcast_scalar, transpose)

    rectangle = ir.Literal(hl.tarray(hl.tint64), [0, 1, 5, 6])
    band = ir.Literal(hl.ttuple(hl.tint64, hl.tint64), (-1, 1))
    intervals = ir.Literal(hl.ttuple(hl.tarray(hl.tint64), hl.tarray(hl.tint64)), ([0, 1, 5, 6], [5, 6, 8, 9]))

    sparsify1 = ir.BlockMatrixSparsify(read, rectangle, ir.RectangleSparsifier)
    sparsify2 = ir.BlockMatrixSparsify(read, band, ir.BandSparsifier(True))
    sparsify3 = ir.BlockMatrixSparsify(read, intervals, ir.RowIntervalSparsifier(True))

    densify = ir.BlockMatrixDensify(read)

    pow_ir = (
        construct_expr(ir.Ref('l', hl.tfloat64), hl.tfloat64) ** construct_expr(ir.Ref('r', hl.tfloat64), hl.tfloat64)
    )._ir
    squared_bm = ir.BlockMatrixMap2(scalar_to_bm, scalar_to_bm, 'l', 'r', pow_ir, "NeedsDense")
    slice_bm = ir.BlockMatrixSlice(matmul, [slice(0, 2, 1), slice(0, 1, 1)])

    return [
        read,
        add_two_bms,
        negate_bm,
        sqrt_bm,
        scalar_to_bm,
        col_vector_to_bm,
        row_vector_to_bm,
        broadcast_scalar,
        broadcast_col,
        broadcast_row,
        squared_bm,
        transpose,
        sparsify1,
        sparsify2,
        sparsify3,
        densify,
        matmul,
        slice_bm,
    ]


static_len_blockmatrix_irs = 18


@pytest.fixture(params=range(static_len_blockmatrix_irs))
def blockmatrix_ir(blockmatrix_irs, request):
    assert len(blockmatrix_irs) == static_len_blockmatrix_irs
    return blockmatrix_irs[request.param]


@skip_unless_spark_backend()
def test_blockmatrix_ir_parses(blockmatrix_ir):
    Env.backend()._parse_blockmatrix_ir(str(blockmatrix_ir))


@skip_unless_spark_backend()
def test_blockmatrix_ir_persist():
    backend = Env.backend()
    bmir = hl.linalg.BlockMatrix.fill(1, 1, 0.0)._bmir
    backend.execute(ir.BlockMatrixWrite(bmir, ir.BlockMatrixPersistWriter('x', 'MEMORY_ONLY')))
    persist = ir.BlockMatrixRead(ir.BlockMatrixPersistReader('x', bmir))
    backend._parse_blockmatrix_ir(str(persist))


def test_blockmatrix_ir_blockmatrix_type(blockmatrix_ir):
    typ = Env.backend().blockmatrix_type(blockmatrix_ir)
    assert blockmatrix_ir.typ == typ


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
            (hl.tcall, hl.Call([0, 1])),
        ]
        return values

    def test_value_same_after_parsing(self):
        test_exprs = []
        expecteds = []
        for t, v in self.values():
            row_v = ir.Literal(t, v)
            range = ir.TableRange(1, 1)
            map_globals_ir = ir.TableMapGlobals(
                range, ir.InsertFields(ir.Ref("global", range.typ.global_type), [("foo", row_v)], None)
            )

            test_exprs.append(hl.Table(map_globals_ir).index_globals())
            expecteds.append(hl.Struct(foo=v))

        actuals = hl.eval(hl.tuple(test_exprs))
        for expr, actual, expected in zip(test_exprs, actuals, expecteds):
            assert actual == expected, str(expr)


class CSETests(unittest.TestCase):
    def test_cse(self):
        x = ir.I32(5)
        x = ir.ApplyBinaryPrimOp('+', x, x)
        expected = '(Let eval __cse_1 (I32 5) (ApplyBinaryPrimOp `+` (Ref __cse_1) (Ref __cse_1)))'
        assert expected == CSERenderer()(x)

    def test_cse_debug(self):
        x = hl.nd.array([0, 1])
        y = hl.tuple((x, x))
        dlen = y[0]
        hl.eval(hl.tuple([hl.if_else(dlen[0] > 1, 1, 1), hl.if_else(dlen[0] > 1, hl.nd.array([0]), dlen)]))

    def test_cse_complex_lifting(self):
        x = ir.I32(5)
        sum = ir.ApplyBinaryPrimOp('+', x, x)
        prod = ir.ApplyBinaryPrimOp('*', sum, sum)
        cond = ir.If(ir.ApplyComparisonOp('EQ', prod, x), sum, x)
        expected = (
            '(Let eval __cse_1 (I32 5)'
            ' (Let eval __cse_2 (ApplyBinaryPrimOp `+` (Ref __cse_1) (Ref __cse_1))'
            ' (If (ApplyComparisonOp EQ (ApplyBinaryPrimOp `*` (Ref __cse_2) (Ref __cse_2)) (Ref __cse_1))'
            ' (Let eval __cse_3 (I32 5)'
            ' (ApplyBinaryPrimOp `+` (Ref __cse_3) (Ref __cse_3)))'
            ' (I32 5))))'
        )
        assert expected == CSERenderer()(cond)

    def test_stream_cse(self):
        x = ir.StreamRange(ir.I32(0), ir.I32(10), ir.I32(1))
        a1 = ir.ToArray(x)
        a2 = ir.ToArray(x)
        t = ir.MakeTuple([a1, a2])
        expected_re = (
            '(Let eval __cse_1 (I32 0)'
            ' (Let eval __cse_2 (I32 10)'
            ' (Let eval __cse_3 (I32 1)'
            ' (MakeTuple (0 1)'
            ' (ToArray (StreamRange [0-9]+ False (Ref __cse_1) (Ref __cse_2) (Ref __cse_3)))'
            ' (ToArray (StreamRange [0-9]+ False (Ref __cse_1) (Ref __cse_2) (Ref __cse_3)))))))'
        )
        expected_re = expected_re.replace('(', '\\(').replace(')', '\\)')
        assert re.match(expected_re, CSERenderer()(t))

    def test_cse2(self):
        x = ir.I32(5)
        y = ir.I32(4)
        sum = ir.ApplyBinaryPrimOp('+', x, x)
        prod = ir.ApplyBinaryPrimOp('*', sum, y)
        div = ir.ApplyBinaryPrimOp('/', prod, sum)
        expected = (
            '(Let eval __cse_1 (I32 5)'
            ' (Let eval __cse_2 (ApplyBinaryPrimOp `+` (Ref __cse_1) (Ref __cse_1))'
            ' (ApplyBinaryPrimOp `/`'
            ' (ApplyBinaryPrimOp `*`'
            ' (Ref __cse_2)'
            ' (I32 4))'
            ' (Ref __cse_2))))'
        )
        assert expected == CSERenderer()(div)

    def test_cse_ifs(self):
        outer_repeated = ir.I32(5)
        inner_repeated = ir.I32(1)
        sum = ir.ApplyBinaryPrimOp('+', inner_repeated, inner_repeated)
        prod = ir.ApplyBinaryPrimOp('*', sum, outer_repeated)
        cond = ir.If(ir.TrueIR(), prod, outer_repeated)
        expected = (
            '(If (True)'
            ' (Let eval __cse_1 (I32 1)'
            ' (ApplyBinaryPrimOp `*`'
            ' (ApplyBinaryPrimOp `+` (Ref __cse_1) (Ref __cse_1))'
            ' (I32 5)))'
            ' (I32 5))'
        )
        assert expected == CSERenderer()(cond)

    def test_shadowing(self):
        x = ir.ApplyBinaryPrimOp('*', ir.Ref('row', tint32), ir.I32(2))
        sum = ir.ApplyBinaryPrimOp('+', x, x)
        inner = ir.Let('row', sum, sum)
        outer = ir.Let('row', ir.I32(5), inner)
        expected = (
            '(Let eval __cse_2 (I32 2)'
            ' (Let eval row (I32 5)'
            ' (Let eval __cse_1 (ApplyBinaryPrimOp `*` (Ref row) (Ref __cse_2))'
            ' (Let eval row (ApplyBinaryPrimOp `+` (Ref __cse_1) (Ref __cse_1))'
            ' (Let eval __cse_3 (ApplyBinaryPrimOp `*` (Ref row) (Ref __cse_2))'
            ' (ApplyBinaryPrimOp `+` (Ref __cse_3) (Ref __cse_3)))))))'
        )
        assert expected == CSERenderer()(outer)

    def test_agg_cse(self):
        table = ir.TableRange(5, 1)
        x = ir.GetField(ir.Ref('row', table.typ.row_type), 'idx')
        inner_sum = ir.ApplyBinaryPrimOp('+', x, x)
        agg = ir.ApplyAggOp('AggOp', [], [inner_sum])
        outer_sum = ir.ApplyBinaryPrimOp('+', agg, agg)
        filter = ir.AggFilter(ir.TrueIR(), outer_sum, False)
        table_agg = ir.TableAggregate(table, ir.MakeTuple([outer_sum, filter]))
        expected = (
            '(TableAggregate (TableRange 5 1)'
            ' (AggLet __cse_1 False (GetField idx (Ref row))'
            ' (AggLet __cse_3 False (ApplyBinaryPrimOp `+` (Ref __cse_1) (Ref __cse_1))'
            ' (Let eval __cse_2 (ApplyAggOp AggOp () ((Ref __cse_3)))'
            ' (MakeTuple (0 1)'
            ' (ApplyBinaryPrimOp `+` (Ref __cse_2) (Ref __cse_2))'
            ' (AggFilter False (True)'
            ' (Let eval __cse_4 (ApplyAggOp AggOp () ((Ref __cse_3)))'
            ' (ApplyBinaryPrimOp `+` (Ref __cse_4) (Ref __cse_4)))))))))'
        )
        assert expected == CSERenderer()(table_agg)

    def test_init_op(self):
        x = ir.I32(5)
        sum = ir.ApplyBinaryPrimOp('+', x, x)
        agg = ir.ApplyAggOp('CallStats', [sum], [sum])
        top = ir.ApplyBinaryPrimOp('+', sum, agg)
        expected = (
            '(Let eval __cse_1 (I32 5)'
            ' (AggLet __cse_3 False (I32 5)'
            ' (ApplyBinaryPrimOp `+`'
            ' (ApplyBinaryPrimOp `+` (Ref __cse_1) (Ref __cse_1))'
            ' (ApplyAggOp CallStats'
            ' ((Let eval __cse_2 (I32 5)'
            ' (ApplyBinaryPrimOp `+` (Ref __cse_2) (Ref __cse_2))))'
            ' ((ApplyBinaryPrimOp `+` (Ref __cse_3) (Ref __cse_3)))))))'
        )
        assert expected == CSERenderer()(top)

    def test_agg_let(self):
        agg = ir.ApplyAggOp('AggOp', [], [ir.Ref('foo', tint32)])
        sum = ir.ApplyBinaryPrimOp('+', agg, agg)
        agglet = ir.AggLet('foo', ir.I32(2), sum, False)
        expected = (
            '(AggLet foo False (I32 2)'
            ' (Let eval __cse_1 (ApplyAggOp AggOp () ((Ref foo)))'
            ' (ApplyBinaryPrimOp `+` (Ref __cse_1) (Ref __cse_1))))'
        )
        assert expected == CSERenderer()(agglet)

    def test_refs(self):
        table = ir.TableRange(10, 1)
        ref = ir.Ref('row', table.typ.row_type)
        x = ir.TableMapRows(table, ir.MakeStruct([('foo', ir.GetField(ref, 'idx')), ('bar', ir.GetField(ref, 'idx'))]))
        expected = (
            '(TableMapRows (TableRange 10 1)'
            ' (MakeStruct'
            ' (foo (GetField idx (Ref row)))'
            ' (bar (GetField idx (Ref row)))))'
        )
        assert expected == CSERenderer()(x)


def _assert_encoding_roundtrip(value):
    lit = hl.literal(value)
    round_trip_just_python_encoding = lit.dtype._from_encoding(lit.dtype._to_encoding(value))
    round_trip_through_scala = hl.eval(lit)
    for round_trip in (round_trip_just_python_encoding, round_trip_through_scala):
        if isinstance(value, np.ndarray):
            assert_array_equal(round_trip, value)
        else:
            assert round_trip == value


@pytest.mark.parametrize(
    'value',
    [
        1,
        5.0,
        "foo",
        [1, 2, 3, 4],
        (5, 6, 7, 8),
        {"foo", "bar", "baz"},
        {"a": {"b": 1}},
        {"a": 1, "b": 2},
        {1: "a", 2: "b"},  # Check values that are pointers to other memory
        {1: [], 2: [1, 2, 3, 4]},
        hl.Call([0, 1]),
        hl.Call([1, 0], phased=True),
        hl.Call([2], phased=True),
        hl.Call([]),
        hl.Call([1, 1]),
        hl.Call([17495, 17495]),
    ],
)
def test_literal_encodings(value):
    _assert_encoding_roundtrip(value)


@pytest.mark.parametrize(
    'value',
    [
        np.array([]),
        np.array([1]),
        np.array([1, 2, 3, 4]),
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([[[[1]], [[2]]], [[[3]], [[4]]], [[[5]], [[6]]]]),
    ],
)
def test_literal_ndarray_encodings(value):
    _assert_encoding_roundtrip(value)
    _assert_encoding_roundtrip(value.T)


def test_decoding_multiple_dicts():
    dict = {0: 'a', 1: 'b', 2: 'c'}
    dict2 = {0: 'x', 1: 'y', 2: 'z'}
    ht = hl.utils.range_table(1).annotate(indices=hl.array([0, 1, 2]))
    ht.select(a=ht.indices.map(lambda i: hl.struct(x=hl.dict(dict).get(i), y=hl.dict(dict2).get(i)))).collect()


def test_locus_interval_encoding():
    start = hl.Locus(contig='chr1', position=10001, reference_genome='GRCh38')
    end = hl.Locus(contig='chr1', position=11001, reference_genome='GRCh38')
    interval = hl.Interval(start=start, end=end, includes_start=True, includes_end=False)
    _assert_encoding_roundtrip(start)
    _assert_encoding_roundtrip(end)
    _assert_encoding_roundtrip(interval)


def test_very_large_ir_deserializes():
    # see: https://github.com/hail-is/hail/issues/14580
    #      https://github.com/hail-is/hail/issues/14650
    large_list = [random.getrandbits(63) for _ in range(5_000_000)]
    large_lit = hl.literal(large_list, hl.tarray(hl.tint64))
    round_trip = hl.eval(large_lit)
    assert large_list == round_trip
