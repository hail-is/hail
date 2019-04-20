import copy

import hail
from hail.ir.blockmatrix_writer import BlockMatrixWriter, BlockMatrixMultiWriter
from hail.utils.java import escape_str, escape_id, dump_json, parsable_strings
from hail.expr.types import *
from hail.typecheck import *
from .base_ir import *
from .matrix_writer import MatrixWriter, MatrixNativeMultiWriter
from .table_writer import TableWriter
from .renderer import Renderer, Renderable, RenderableStr, ParensRenderer

from collections import defaultdict

def _env_bind(env, k, v):
    env = env.copy()
    env[k] = v
    return env


class I32(IR):
    @typecheck_method(x=int)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def _eq(self, other):
        return self.x == other.x

    def copy(self):
        return I32(self.x)

    def head_str(self):
        return self.x

    def _compute_type(self, env, agg_env):
        self._type = tint32


class I64(IR):
    @typecheck_method(x=int)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def _eq(self, other):
        return self.x == other.x

    def copy(self):
        return I64(self.x)

    def head_str(self):
        return self.x

    def _compute_type(self, env, agg_env):
        self._type = tint64


class F32(IR):
    @typecheck_method(x=numeric)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def _eq(self, other):
        return self.x == other.x

    def copy(self):
        return F32(self.x)

    def head_str(self):
        return self.x

    def _compute_type(self, env, agg_env):
        self._type = tfloat32


class F64(IR):
    @typecheck_method(x=numeric)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def _eq(self, other):
        return self.x == other.x

    def copy(self):
        return F64(self.x)

    def head_str(self):
        return self.x

    def _compute_type(self, env, agg_env):
        self._type = tfloat64


class Str(IR):
    @typecheck_method(x=str)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def _eq(self, other):
        return self.x == other.x

    def copy(self):
        return Str(self.x)

    def head_str(self):
        return f'"{escape_str(self.x)}"'

    def _compute_type(self, env, agg_env):
        self._type = tstr


class FalseIR(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        return FalseIR()

    def _ir_name(self):
        return 'False'

    def _compute_type(self, env, agg_env):
        self._type = tbool


class TrueIR(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        return TrueIR()

    def _ir_name(self):
        return 'True'

    def _compute_type(self, env, agg_env):
        self._type = tbool


class Void(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        return Void()

    def _compute_type(self, env, agg_env):
        self._type = tvoid


class Cast(IR):
    @typecheck_method(v=IR, typ=hail_type)
    def __init__(self, v, typ):
        super().__init__(v)
        self.v = v
        self._typ = typ

    @property
    def typ(self):
        return self._typ

    def _eq(self, other):
        return self._typ == other._typ

    @typecheck_method(v=IR)
    def copy(self, v):
        return Cast(v, self.typ)

    def head_str(self):
        return self._typ._parsable_string()

    def _compute_type(self, env, agg_env):
        self.v._compute_type(env, agg_env)
        self._type = self._typ


class NA(IR):
    @typecheck_method(typ=hail_type)
    def __init__(self, typ):
        super().__init__()
        self._typ = typ

    @property
    def typ(self):
        return self._typ

    def _eq(self, other):
        return self._typ == other._typ

    def copy(self):
        return NA(self._typ)

    def head_str(self):
        return self._typ._parsable_string()

    def _compute_type(self, env, agg_env):
        self._type = self._typ


class IsNA(IR):
    @typecheck_method(value=IR)
    def __init__(self, value):
        super().__init__(value)
        self.value = value

    @typecheck_method(value=IR)
    def copy(self, value):
        return IsNA(value)

    def _compute_type(self, env, agg_env):
        self.value._compute_type(env, agg_env)
        self._type = tbool


class If(IR):
    @typecheck_method(cond=IR, cnsq=IR, altr=IR)
    def __init__(self, cond, cnsq, altr):
        super().__init__(cond, cnsq, altr)
        self.cond = cond
        self.cnsq = cnsq
        self.altr = altr

    @typecheck_method(cond=IR, cnsq=IR, altr=IR)
    def copy(self, cond, cnsq, altr):
        return If(cond, cnsq, altr)

    def _compute_type(self, env, agg_env):
        self.cond._compute_type(env, agg_env)
        self.cnsq._compute_type(env, agg_env)
        self.altr._compute_type(env, agg_env)
        assert (self.cnsq.typ == self.altr.typ)
        self._type = self.cnsq.typ


class Let(IR):
    @typecheck_method(name=str, value=IR, body=IR)
    def __init__(self, name, value, body):
        super().__init__(value, body)
        self.name = name
        self.value = value
        self.body = body

    @typecheck_method(value=IR, body=IR)
    def copy(self, value, body):
        return Let(self.name, value, body)

    def head_str(self):
        return escape_id(self.name)

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _eq(self, other):
        return other.name == self.name

    def _compute_type(self, env, agg_env):
        self.value._compute_type(env, agg_env)
        self.body._compute_type(_env_bind(env, self.name, self.value._type), agg_env)
        self._type = self.body._type


class Ref(IR):
    @typecheck_method(name=str)
    def __init__(self, name):
        super().__init__()
        self.name = name

    def copy(self):
        return Ref(self.name)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return other.name == self.name

    def _compute_type(self, env, agg_env):
        self._type = env[self.name]


class TopLevelReference(Ref):
    @typecheck_method(name=str)
    def __init__(self, name):
        super().__init__(name)

    @property
    def is_nested_field(self):
        return True

    def copy(self):
        return TopLevelReference(self.name)

    def _ir_name(self):
        return 'Ref'

    def _compute_type(self, env, agg_env):
        assert self.name in env, f'{self.name} not found in {env}'
        self._type = env[self.name]


class ApplyBinaryPrimOp(IR):
    @typecheck_method(op=str, l=IR, r=IR)
    def __init__(self, op, l, r):
        super().__init__(l, r)
        self.op = op
        self.l = l
        self.r = r

    @typecheck_method(l=IR, r=IR)
    def copy(self, l, r):
        return ApplyBinaryPrimOp(self.op, l, r)

    def head_str(self):
        return escape_id(self.op)

    def _eq(self, other):
        return other.op == self.op

    def _compute_type(self, env, agg_env):
        self.l._compute_type(env, agg_env)
        self.r._compute_type(env, agg_env)
        if self.op == '/':
            if self.l.typ == tfloat64:
                self._type = tfloat64
            else:
                self._type = tfloat32
        else:
            self._type = self.l.typ


class ApplyUnaryPrimOp(IR):
    @typecheck_method(op=str, x=IR)
    def __init__(self, op, x):
        super().__init__(x)
        self.op = op
        self.x = x

    @typecheck_method(x=IR)
    def copy(self, x):
        return ApplyUnaryPrimOp(self.op, x)

    def head_str(self):
        return escape_id(self.op)

    def _eq(self, other):
        return other.op == self.op

    def _compute_type(self, env, agg_env):
        self.x._compute_type(env, agg_env)
        self._type = self.x.typ


class ApplyComparisonOp(IR):
    @typecheck_method(op=str, l=IR, r=IR)
    def __init__(self, op, l, r):
        super().__init__(l, r)
        self.op = op
        self.l = l
        self.r = r

    @typecheck_method(l=IR, r=IR)
    def copy(self, l, r):
        return ApplyComparisonOp(self.op, l, r)

    def head_str(self):
        return escape_id(self.op)

    def _eq(self, other):
        return other.op == self.op

    def _compute_type(self, env, agg_env):
        self.l._compute_type(env, agg_env)
        self.r._compute_type(env, agg_env)
        self._type = tbool


class MakeArray(IR):
    @typecheck_method(args=sequenceof(IR), type=nullable(hail_type))
    def __init__(self, args, type):
        super().__init__(*args)
        self.args = args
        self._type = type

    def copy(self, *args):
        return MakeArray(args, self._type)

    def head_str(self):
        return self._type._parsable_string() if self._type is not None else 'None'

    def _eq(self, other):
        return other._type == self._type

    def _compute_type(self, env, agg_env):
        for a in self.args:
            a._compute_type(env, agg_env)
        if self._type is None:
            self._type = tarray(self.args[0].typ)


class ArrayRef(IR):
    @typecheck_method(a=IR, i=IR)
    def __init__(self, a, i):
        super().__init__(a, i)
        self.a = a
        self.i = i

    @typecheck_method(a=IR, i=IR)
    def copy(self, a, i):
        return ArrayRef(a, i)

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.i._compute_type(env, agg_env)
        self._type = self.a.typ.element_type


class ArrayLen(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return ArrayLen(a)

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self._type = tint32


class ArrayRange(IR):
    @typecheck_method(start=IR, stop=IR, step=IR)
    def __init__(self, start, stop, step):
        super().__init__(start, stop, step)
        self.start = start
        self.stop = stop
        self.step = step

    @typecheck_method(start=IR, stop=IR, step=IR)
    def copy(self, start, stop, step):
        return ArrayRange(start, stop, step)

    def _compute_type(self, env, agg_env):
        self.start._compute_type(env, agg_env)
        self.stop._compute_type(env, agg_env)
        self.step._compute_type(env, agg_env)
        self._type = tarray(tint32)


class MakeNDArray(IR):
    @typecheck_method(ndim=int, data=IR, shape=IR, row_major=IR)
    def __init__(self, ndim, data, shape, row_major):
        super().__init__(data, shape, row_major)
        self.ndim = ndim
        self.data = data
        self.shape = shape
        self.row_major = row_major

    @typecheck_method(data=IR, shape=IR, row_major=IR)
    def copy(self, data, shape, row_major):
        return MakeNDArray(self.ndim, data, shape, row_major)

    def head_str(self):
        return f'{self.ndim}'

    def _compute_type(self, env, agg_env):
        self.data._compute_type(env, agg_env)
        self.shape._compute_type(env, agg_env)
        self.row_major._compute_type(env, agg_env)
        self._type = tndarray(self.data.typ.element_type, self.ndim)


class NDArrayMap(IR):
    @typecheck_method(nd=IR, name=str, body=IR)
    def __init__(self, nd, name, body):
        super().__init__(nd, body)
        self.nd = nd
        self.name = name
        self.body = body

    @typecheck_method(nd=IR, body=IR)
    def copy(self, nd, body):
        return NDArrayMap(nd, self.name, body)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return self.name == other.name

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env):
        self.nd._compute_type(env, agg_env)
        self.body._compute_type(_env_bind(env, self.name, self.nd.typ.element_type), agg_env)
        self._type = tndarray(self.body.typ, self.nd.typ.ndim)


class NDArrayRef(IR):
    @typecheck_method(nd=IR, idxs=sequenceof(IR))
    def __init__(self, nd, idxs):
        super().__init__(nd, *idxs)
        self.nd = nd
        self.idxs = idxs

    def copy(self, *args):
        return NDArrayRef(args[0], args[1:])

    def _compute_type(self, env, agg_env):
        self.nd._compute_type(env, agg_env)
        [idx._compute_type(env, agg_env) for idx in self.idxs]
        self._type = self.nd.typ.element_type


class ArraySort(IR):
    @typecheck_method(a=IR, l_name=str, r_name=str, compare=IR)
    def __init__(self, a, l_name, r_name, compare):
        super().__init__(a, compare)
        self.a = a
        self.l_name = l_name
        self.r_name = r_name
        self.compare = compare

    @typecheck_method(a=IR, compare=IR)
    def copy(self, a, compare):
        return ArraySort(a, self.l_name, self.r_name, compare)

    def head_str(self):
        return f'{escape_id(self.l_name)} {escape_id(self.r_name)}'

    @property
    def bound_variables(self):
        return {self.l_name, self.r_name} | super().bound_variables

    def __eq__(self, other):
        return other.l_name == self.l_name and other.r_name == self.r_name

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self._type = self.a.typ


class ToSet(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return ToSet(a)

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self._type = tset(self.a.typ.element_type)


class ToDict(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return ToDict(a)

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self._type = tdict(self.a.typ['key'], self.a.typ['value'])


class ToArray(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return ToArray(a)

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self._type = tarray(self.a.typ.element_type)


class LowerBoundOnOrderedCollection(IR):
    @typecheck_method(ordered_collection=IR, elem=IR, on_key=bool)
    def __init__(self, ordered_collection, elem, on_key):
        super().__init__(ordered_collection, elem)
        self.ordered_collection = ordered_collection
        self.elem = elem
        self.on_key = on_key

    @typecheck_method(ordered_collection=IR, elem=IR)
    def copy(self, ordered_collection, elem):
        return LowerBoundOnOrderedCollection(ordered_collection, elem, self.on_key)

    def head_str(self):
        return self.on_key

    def _compute_type(self, env, agg_env):
        self.ordered_collection._compute_type(env, agg_env)
        self.elem._compute_type(env, agg_env)
        self._type = tint32


class GroupByKey(IR):
    @typecheck_method(collection=IR)
    def __init__(self, collection):
        super().__init__(collection)
        self.collection = collection

    @typecheck_method(collection=IR)
    def copy(self, collection):
        return GroupByKey(collection)

    def _compute_type(self, env, agg_env):
        self.collection._compute_type(env, agg_env)
        self._type = tdict(self.collection.typ.element_type.types[0],
                           tarray(self.collection.typ.element_type.types[1]))


class ArrayMap(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        return ArrayMap(a, self.name, body)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return self.name == other.name

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.body._compute_type(_env_bind(env, self.name, self.a.typ.element_type), agg_env)
        self._type = tarray(self.body.typ)


class ArrayFilter(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        return ArrayFilter(a, self.name, body)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return self.name == other.name

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.body._compute_type(_env_bind(env, self.name, self.a.typ.element_type), agg_env)
        self._type = self.a.typ


class ArrayFlatMap(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        return ArrayFlatMap(a, self.name, body)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return self.name == other.name

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.body._compute_type(_env_bind(env, self.name, self.a.typ.element_type), agg_env)
        self._type = tarray(self.body.typ.element_type)


class ArrayFold(IR):
    @typecheck_method(a=IR, zero=IR, accum_name=str, value_name=str, body=IR)
    def __init__(self, a, zero, accum_name, value_name, body):
        super().__init__(a, zero, body)
        self.a = a
        self.zero = zero
        self.accum_name = accum_name
        self.value_name = value_name
        self.body = body

    @typecheck_method(a=IR, zero=IR, body=IR)
    def copy(self, a, zero, body):
        return ArrayFold(a, zero, self.accum_name, self.value_name, body)

    def head_str(self):
        return f'{escape_id(self.accum_name)} {escape_id(self.value_name)}'

    def _eq(self, other):
        return other.accum_name == self.accum_name and \
               other.value_name == self.value_name

    @property
    def bound_variables(self):
        return {self.accum_name, self.value_name} | super().bound_variables

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.zero._compute_type(env, agg_env)
        self.body._compute_type(
            _env_bind(
                _env_bind(env, self.value_name, self.a.typ.element_type),
                self.accum_name, self.zero.typ),
            agg_env)
        self._type = self.zero.typ


class ArrayScan(IR):
    @typecheck_method(a=IR, zero=IR, accum_name=str, value_name=str, body=IR)
    def __init__(self, a, zero, accum_name, value_name, body):
        super().__init__(a, zero, body)
        self.a = a
        self.zero = zero
        self.accum_name = accum_name
        self.value_name = value_name
        self.body = body

    @typecheck_method(a=IR, zero=IR, body=IR)
    def copy(self, a, zero, body):
        return ArrayScan(a, zero, self.accum_name, self.value_name, body)

    def head_str(self):
        return f'{escape_id(self.accum_name)} {escape_id(self.value_name)}'

    def _eq(self, other):
        return other.accum_name == self.accum_name and \
               other.value_name == self.value_name

    @property
    def bound_variables(self):
        return {self.accum_name, self.value_name} | super().bound_variables

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.zero._compute_type(env, agg_env)
        self.body._compute_type(
            _env_bind(
                _env_bind(env, self.value_name, self.a.typ.element_type),
                self.accum_name, self.zero.typ),
            agg_env)
        self._type = tarray(self.body.typ)


class ArrayLeftJoinDistinct(IR):
    @typecheck_method(left=IR, right=IR, l_name=str, r_name=str, compare=IR, join=IR)
    def __init__(self, left, right, l_name, r_name, compare, join):
        super().__init__(left, right, compare, join)
        self.left = left
        self.right = right
        self.l_name = l_name
        self.r_name = r_name
        self.compare = compare
        self.join = join

    @typecheck_method(left=IR, right=IR, compare=IR, join=IR)
    def copy(self, left, right, compare, join):
        return ArrayLeftJoinDistinct(left, right, self.l_name, self.r_name, compare, join)

    def head_str(self):
        return f'{escape_id(self.l_name)} {escape_id(self.r_name)}'

    def _eq(self, other):
        return other.l_name == self.l_name and \
               other.r_name == self.r_name

    @property
    def bound_variables(self):
        return {self.l_name, self.r_name} | super().bound_variables


class ArrayFor(IR):
    @typecheck_method(a=IR, value_name=str, body=IR)
    def __init__(self, a, value_name, body):
        super().__init__(a, body)
        self.a = a
        self.value_name = value_name
        self.body = body

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        return ArrayFor(a, self.value_name, body)

    def head_str(self):
        return escape_id(self.value_name)

    def _eq(self, other):
        return self.value_name == other.value_name

    @property
    def bound_variables(self):
        return {self.value_name} | super().bound_variables

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.body._compute_type(_env_bind(env, self.value_name, self.a.typ.element_type), agg_env)
        self._type = tvoid


class AggFilter(IR):
    @typecheck_method(cond=IR, agg_ir=IR, is_scan=bool)
    def __init__(self, cond, agg_ir, is_scan):
        super().__init__(cond, agg_ir)
        self.cond = cond
        self.agg_ir = agg_ir
        self.is_scan = is_scan

    @typecheck_method(cond=IR, agg_ir=IR)
    def copy(self, cond, agg_ir):
        return AggFilter(cond, agg_ir, self.is_scan)

    def head_str(self):
        return str(self.is_scan)

    def _eq(self, other):
        return self.is_scan == other.is_scan

    def _compute_type(self, env, agg_env):
        self.cond._compute_type(agg_env, None)
        self.agg_ir._compute_type(env, agg_env)
        self._type = self.agg_ir.typ


class AggExplode(IR):
    @typecheck_method(array=IR, name=str, agg_body=IR, is_scan=bool)
    def __init__(self, array, name, agg_body, is_scan):
        super().__init__(array, agg_body)
        self.name = name
        self.array = array
        self.agg_body = agg_body
        self.is_scan = is_scan

    @typecheck_method(array=IR, agg_body=IR)
    def copy(self, array, agg_body):
        return AggExplode(array, self.name, agg_body, self.is_scan)

    def head_str(self):
        return f'{escape_id(self.name)} {self.is_scan}'

    def _eq(self, other):
        return self.name == other.name and self.is_scan == other.is_scan

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env):
        self.array._compute_type(agg_env, None)
        self.agg_body._compute_type(env, _env_bind(agg_env, self.name, self.array.typ.element_type))
        self._type = self.agg_body.typ


class AggGroupBy(IR):
    @typecheck_method(key=IR, agg_ir=IR, is_scan=bool)
    def __init__(self, key, agg_ir, is_scan):
        super().__init__(key, agg_ir)
        self.key = key
        self.agg_ir = agg_ir
        self.is_scan = is_scan

    @typecheck_method(key=IR, agg_ir=IR)
    def copy(self, key, agg_ir):
        return AggGroupBy(key, agg_ir, self.is_scan)

    def head_str(self):
        return str(self.is_scan)

    def _eq(self, other):
        return self.is_scan == other.is_scan

    def _compute_type(self, env, agg_env):
        self.key._compute_type(agg_env, None)
        self.agg_ir._compute_type(env, agg_env)
        self._type = tdict(self.key.typ, self.agg_ir.typ)


class AggArrayPerElement(IR):
    @typecheck_method(array=IR, element_name=str, index_name=str, agg_ir=IR, is_scan=bool)
    def __init__(self, array, element_name, index_name, agg_ir, is_scan):
        super().__init__(array, agg_ir)
        self.array = array
        self.element_name = element_name
        self.index_name = index_name
        self.agg_ir = agg_ir
        self.is_scan = is_scan

    @typecheck_method(array=IR, agg_ir=IR)
    def copy(self, array, agg_ir):
        return AggArrayPerElement(array, self.element_name, self.index_name, agg_ir, self.is_scan)

    def head_str(self):
        return f'{escape_id(self.element_name)} {escape_id(self.index_name)} {self.is_scan}'

    def _eq(self, other):
        return self.element_name == other.element_name and self.index_name == other.index_name and  self.is_scan == other.is_scan

    def _compute_type(self, env, agg_env):
        self.array._compute_type(agg_env, None)
        self.agg_ir._compute_type(_env_bind(env, self.index_name, tint32),
                                  _env_bind(agg_env, self.element_name, self.array.typ.element_type))
        self._type = tarray(self.agg_ir.typ)

    @property
    def bound_variables(self):
        return {self.element_name, self.index_name} | super().bound_variables


def _register(registry, name, f):
    registry[name].append(f)

_aggregator_registry = defaultdict(list)


def register_aggregator(name, ctor_params, init_params, seq_params, ret_type):
    _register(_aggregator_registry, name, (ctor_params, init_params, seq_params, ret_type))


def lookup_aggregator_return_type(name, ctor_args, init_args, seq_args):
    if name in _aggregator_registry:
        fns = _aggregator_registry[name]
        for f in fns:
            (ctor_params, init_params, seq_params, ret_type) = f
            for p in ctor_params:
                p.clear()
            if init_params:
                for p in init_params:
                    p.clear()
            for p in seq_params:
                p.clear()
            if init_params:
                init_match = all(p.unify(a) for p, a in zip(init_params, init_args))
            else:
                init_match = init_args is None
            if (init_match
                    and all(p.unify(a) for p, a in zip(ctor_params, ctor_args))
                    and all(p.unify(a) for p, a in zip(seq_params, seq_args))):
                return ret_type.subst()
    raise KeyError(f'aggregator {name}({ ",".join([str(t) for t in seq_args]) }) not found')


class BaseApplyAggOp(IR):
    @typecheck_method(agg_op=str,
                      constructor_args=sequenceof(IR),
                      init_op_args=nullable(sequenceof(IR)),
                      seq_op_args=sequenceof(IR))
    def __init__(self, agg_op, constructor_args, init_op_args, seq_op_args):
        init_op_children = [] if init_op_args is None else init_op_args
        super().__init__(*constructor_args, *init_op_children, *seq_op_args)
        self.agg_op = agg_op
        self.constructor_args = constructor_args
        self.init_op_args = init_op_args
        self.seq_op_args = seq_op_args

    def copy(self, *args):
        new_instance = self.__class__
        n_seq_op_args = len(self.seq_op_args)
        n_constructor_args = len(self.constructor_args)
        constr_args = args[:n_constructor_args]
        init_op_args = args[n_constructor_args:-n_seq_op_args]
        seq_op_args = args[-n_seq_op_args:]
        return new_instance(self.agg_op, constr_args, init_op_args if len(init_op_args) != 0 else None, seq_op_args)

    def head_str(self):
        return f' {self.agg_op} '

    def render_children(self, r):
        return [
            ParensRenderer(self.constructor_args),
            RenderableStr('None') if not self.init_op_args else ParensRenderer(self.init_op_args),
            ParensRenderer(self.seq_op_args)
        ]

    @property
    def aggregations(self):
        assert all(map(lambda c: len(c.aggregations) == 0, self.children))
        return [self]

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               other.agg_op == self.agg_op and \
               other.constructor_args == self.constructor_args and \
               other.init_op_args == self.init_op_args and \
               other.seq_op_args == self.seq_op_args

    def _compute_type(self, env, agg_env):
        for a in self.constructor_args:
            a._compute_type(env, agg_env)
        if self.init_op_args:
            for a in self.init_op_args:
                a._compute_type(env, agg_env)
        for a in self.seq_op_args:
            a._compute_type(agg_env, None)

        self._type = lookup_aggregator_return_type(
            self.agg_op,
            [a.typ for a in self.constructor_args],
            [a.typ for a in self.init_op_args] if self.init_op_args else None,
            [a.typ for a in self.seq_op_args])


class ApplyAggOp(BaseApplyAggOp):
    @typecheck_method(agg_op=str,
                      constructor_args=sequenceof(IR),
                      init_op_args=nullable(sequenceof(IR)),
                      seq_op_args=sequenceof(IR))
    def __init__(self, agg_op, constructor_args, init_op_args, seq_op_args):
        super().__init__(agg_op, constructor_args, init_op_args, seq_op_args)


class ApplyScanOp(BaseApplyAggOp):
    @typecheck_method(agg_op=str,
                      constructor_args=sequenceof(IR),
                      init_op_args=nullable(sequenceof(IR)),
                      seq_op_args=sequenceof(IR))
    def __init__(self, agg_op, constructor_args, init_op_args, seq_op_args):
        super().__init__(agg_op, constructor_args, init_op_args, seq_op_args)


class Begin(IR):
    @typecheck_method(xs=sequenceof(IR))
    def __init__(self, xs):
        super().__init__(*xs)
        self.xs = xs

    def copy(self, *xs):
        return Begin(xs)

    def _compute_type(self, env, agg_env):
        for x in self.xs:
            x._compute_type(env, agg_env)
        self._type = tvoid


class MakeStruct(IR):
    @typecheck_method(fields=sequenceof(sized_tupleof(str, IR)))
    def __init__(self, fields):
        super().__init__(*[ir for (n, ir) in fields])
        self.fields = fields

    def copy(self, *irs):
        assert len(irs) == len(self.fields)
        return MakeStruct([(n, ir) for (n, _), ir in zip(self.fields, irs)])

    def render_children(self, r):
        return [InsertFields.IFRenderField(escape_id(f), x) for f, x in self.fields]

    def __eq__(self, other):
        return isinstance(other, MakeStruct) \
               and other.fields == self.fields

    def _compute_type(self, env, agg_env):
        for f, x in self.fields:
            x._compute_type(env, agg_env)
        self._type = tstruct(**{f: x.typ for f, x in self.fields})


class SelectFields(IR):
    @typecheck_method(old=IR, fields=sequenceof(str))
    def __init__(self, old, fields):
        super().__init__(old)
        self.old = old
        self.fields = fields

    @typecheck_method(old=IR)
    def copy(self, old):
        return SelectFields(old, self.fields)

    def head_str(self):
        return '({})'.format(' '.join(map(escape_id, self.fields)))

    def _eq(self, other):
        return self.fields == other.fields

    def _compute_type(self, env, agg_env):
        self.old._compute_type(env, agg_env)
        self._type = self.old.typ._select_fields(self.fields)


class InsertFields(IR):
    class IFRenderField(Renderable):
        def __init__(self, field, child):
            self.field = field
            self.child = child

        def render_head(self, r: 'Renderer'):
            return f'({self.field} '

        def render_tail(self, r: 'Renderer'):
            return ')'

        def render_children(self, r: 'Renderer'):
            return [self.child]

    @typecheck_method(old=IR, fields=sequenceof(sized_tupleof(str, IR)), field_order=nullable(sequenceof(str)))
    def __init__(self, old, fields, field_order):
        super().__init__(old, *[ir for (f, ir) in fields])
        self.old = old
        self.fields = fields
        self.field_order = field_order

    def copy(self, *args):
        assert len(args) == len(self.fields) + 1
        return InsertFields(args[0], [(n, ir) for (n, _), ir in zip(self.fields, args[1:])], self.field_order)

    def render_children(self, r):
        return [
            self.old,
            hail.ir.RenderableStr('None' if self.field_order is None else parsable_strings(self.field_order)),
            *(InsertFields.IFRenderField(escape_id(f), x) for f, x in self.fields)
        ]

    def __eq__(self, other):
        return isinstance(other, InsertFields) and \
               other.old == self.old and \
               other.fields == self.fields and \
               other.field_order == self.field_order

    def _compute_type(self, env, agg_env):
        self.old._compute_type(env, agg_env)
        for f, x in self.fields:
            x._compute_type(env, agg_env)
        self._type = self.old.typ._insert_fields(**{f: x.typ for f, x in self.fields})
        if self.field_order:
            self._type = tstruct(**{f: self._type[f] for f in self.field_order})


class GetField(IR):
    @typecheck_method(o=IR, name=str)
    def __init__(self, o, name):
        super().__init__(o)
        self.o = o
        self.name = name

    @typecheck_method(o=IR)
    def copy(self, o):
        return GetField(o, self.name)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return self.name == other.name

    @property
    def is_nested_field(self):
        return self.o.is_nested_field

    def _compute_type(self, env, agg_env):
        self.o._compute_type(env, agg_env)
        self._type = self.o.typ[self.name]


class MakeTuple(IR):
    @typecheck_method(elements=sequenceof(IR))
    def __init__(self, elements):
        super().__init__(*elements)
        self.elements = elements

    def copy(self, *args):
        return MakeTuple(args)

    def _compute_type(self, env, agg_env):
        for x in self.elements:
            x._compute_type(env, agg_env)
        self._type = ttuple(*[x.typ for x in self.elements])


class GetTupleElement(IR):
    @typecheck_method(o=IR, idx=int)
    def __init__(self, o, idx):
        super().__init__(o)
        self.o = o
        self.idx = idx

    @typecheck_method(o=IR)
    def copy(self, o):
        return GetTupleElement(o, self.idx)

    def head_str(self):
        return self.idx

    def _eq(self, other):
        return self.idx == other.idx

    def _compute_type(self, env, agg_env):
        self.o._compute_type(env, agg_env)
        self._type = self.o.typ.types[self.idx]

class In(IR):
    @typecheck_method(i=int, typ=hail_type)
    def __init__(self, i, typ):
        super().__init__()
        self.i = i
        self._typ = typ

    @property
    def typ(self):
        return self._typ

    def copy(self):
        return In(self.i, self._typ)

    def head_str(self):
        return f'{self._typ._parsable_string()} {self.i}'

    def _eq(self, other):
        return other.i == self.i and \
               other._typ == self._typ

    def _compute_type(self, env, agg_env):
        self._type = self._typ


class Die(IR):
    @typecheck_method(message=IR, typ=hail_type)
    def __init__(self, message, typ):
        super().__init__(message)
        self.message = message
        self._typ = typ

    @property
    def typ(self):
        return self._typ

    def copy(self, message):
        return Die(message, self._typ)

    def head_str(self):
        return self._typ._parsable_string()

    def _eq(self, other):
        return other._typ == self._typ

    def _compute_type(self, env, agg_env):
        self._type = self._typ


_function_registry = defaultdict(list)
_seeded_function_registry = defaultdict(list)
_session_functions = set()

def clear_session_functions():
    global _session_functions
    for name, param_types, ret_type in _session_functions:
        remove_function(name, param_types, ret_type)

    _session_functions = set()

def remove_function(name, param_types, ret_type):
    f = (param_types, ret_type)
    bindings = _function_registry[name]
    bindings = [b for b in bindings if b != f]
    if not bindings:
        del _function_registry[name]
    else:
        _function_registry[name] = bindings

def register_session_function(name, param_types, ret_type):
    _session_functions.add((name, param_types, ret_type))
    register_function(name, param_types, ret_type)

def register_function(name, param_types, ret_type):
    _register(_function_registry, name, (param_types, ret_type))


def register_seeded_function(name, param_types, ret_type):
    _register(_seeded_function_registry, name, (param_types, ret_type))


def _lookup_function_return_type(registry, fkind, name, arg_types):
    for f in registry[name]:
        (param_types, ret_type) = f
        for p in param_types:
            p.clear()
        ret_type.clear()
        if all(p.unify(a) for p, a in zip(param_types, arg_types)):
            return ret_type.subst()
    raise KeyError(f'{fkind} {name}({ ",".join([str(t) for t in arg_types]) }) not found')


def lookup_function_return_type(name, arg_types):
    return _lookup_function_return_type(_function_registry, 'function', name, arg_types)


def lookup_seeded_function_return_type(name, arg_types):
    return _lookup_function_return_type(_seeded_function_registry, 'seeded function', name, arg_types)


class Apply(IR):
    @typecheck_method(function=str, args=IR)
    def __init__(self, function, *args):
        super().__init__(*args)
        self.function = function
        self.args = args

    def copy(self, *args):
        return Apply(self.function, *args)

    def head_str(self):
        return escape_id(self.function)

    def _eq(self, other):
        return other.function == self.function

    def _compute_type(self, env, agg_env):
        for arg in self.args:
            arg._compute_type(env, agg_env)

        self._type = lookup_function_return_type(self.function, [a.typ for a in self.args])


class ApplySeeded(IR):
    @typecheck_method(function=str, seed=int, args=IR)
    def __init__(self, function, seed, *args):
        super().__init__(*args)
        self.function = function
        self.args = args
        self.seed = seed

    def copy(self, *args):
        return ApplySeeded(self.function, self.seed, *args)

    def head_str(self):
        return f'{escape_id(self.function)} {self.seed}'

    def _eq(self, other):
        return other.function == self.function and \
               other.seed == self.seed

    def _compute_type(self, env, agg_env):
        for arg in self.args:
            arg._compute_type(env, agg_env)

        self._type = lookup_seeded_function_return_type(self.function, [a.typ for a in self.args])


class Uniroot(IR):
    @typecheck_method(argname=str, function=IR, min=IR, max=IR)
    def __init__(self, argname, function, min, max):
        super().__init__(function, min, max)
        self.argname = argname
        self.function = function
        self.min = min
        self.max = max

    @typecheck_method(function=IR, min=IR, max=IR)
    def copy(self, function, min, max):
        return Uniroot(self.argname, function, min, max)

    def head_str(self):
        return escape_id(self.argname)

    @property
    def bound_variables(self):
        return {self.argname} | super().bound_variables

    def _eq(self, other):
        return other.argname == self.argname

    def _compute_type(self, env, agg_env):
        self.function._compute_type(_env_bind(env, self.argname, tfloat64), agg_env)
        self.min._compute_type(env, agg_env)
        self.max._compute_type(env, agg_env)
        self._type = tfloat64


class TableCount(IR):
    @typecheck_method(child=TableIR)
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    @typecheck_method(child=TableIR)
    def copy(self, child):
        return TableCount(child)

    def _compute_type(self, env, agg_env):
        self.child._compute_type()
        self._type = tint64


class TableGetGlobals(IR):
    @typecheck_method(child=TableIR)
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    @typecheck_method(child=TableIR)
    def copy(self, child):
        return TableGetGlobals(child)

    def _compute_type(self, env, agg_env):
        self.child._compute_type()
        self._type = self.child.typ.global_type


class TableCollect(IR):
    @typecheck_method(child=TableIR)
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    @typecheck_method(child=TableIR)
    def copy(self, child):
        return TableCollect(child)

    def _compute_type(self, env, agg_env):
        self.child._compute_type()
        self._type = tstruct(**{'rows': tarray(self.child.typ.row_type),
                                'global': self.child.typ.global_type})


class TableAggregate(IR):
    @typecheck_method(child=TableIR, query=IR)
    def __init__(self, child, query):
        super().__init__(child, query)
        self.child = child
        self.query = query

    @typecheck_method(child=TableIR, query=IR)
    def copy(self, child, query):
        return TableAggregate(child, query)

    def _compute_type(self, env, agg_env):
        self.query._compute_type(self.child.typ.global_env(), self.child.typ.row_env())
        self._type = self.query.typ


class MatrixAggregate(IR):
    @typecheck_method(child=MatrixIR, query=IR)
    def __init__(self, child, query):
        super().__init__(child, query)
        self.child = child
        self.query = query

    @typecheck_method(child=MatrixIR, query=IR)
    def copy(self, child, query):
        return MatrixAggregate(child, query)

    def __eq__(self, other):
        return isinstance(other, MatrixAggregate) and \
               other.child == self.child and \
               other.query == self.query

    def _compute_type(self, env, agg_env):
        self.query._compute_type(self.child.typ.global_env(), self.child.typ.entry_env())
        self._type = self.query.typ


class TableWrite(IR):
    @typecheck_method(child=TableIR, writer=TableWriter)
    def __init__(self, child, writer):
        super().__init__(child)
        self.child = child
        self.writer = writer

    @typecheck_method(child=TableIR)
    def copy(self, child):
        return TableWrite(child, self.writer)

    def head_str(self):
        return f'"{self.writer.render()}"'

    def _eq(self, other):
        return other.writer == self.writer

    def _compute_type(self, env, agg_env):
        self.child._compute_type()
        self._type = tvoid

class MatrixWrite(IR):
    @typecheck_method(child=MatrixIR, matrix_writer=MatrixWriter)
    def __init__(self, child, matrix_writer):
        super().__init__(child)
        self.child = child
        self.matrix_writer = matrix_writer

    @typecheck_method(child=MatrixIR)
    def copy(self, child):
        return MatrixWrite(child, self.matrix_writer)

    def head_str(self):
        return f'"{self.matrix_writer.render()}"'

    def _eq(self, other):
        return other.matrix_writer == self.matrix_writer

    def _compute_type(self, env, agg_env):
        self.child._compute_type()
        self._type = tvoid


class MatrixMultiWrite(IR):
    @typecheck_method(children=sequenceof(MatrixIR), writer=MatrixNativeMultiWriter)
    def __init__(self, children, writer):
        super().__init__(*children)
        self.writer = writer

    def copy(self, *children):
        return MatrixMultiWrite(children, self.writer)

    def head_str(self):
        return f'"{self.writer.render()}"'

    def _eq(self, other):
        return other.writer == self.writer

    def _compute_type(self, env, agg_env):
        for x in self.children:
            x._compute_type()
        self._type = tvoid


class BlockMatrixWrite(IR):
    @typecheck_method(child=BlockMatrixIR, writer=BlockMatrixWriter)
    def __init__(self, child, writer):
        super().__init__(child)
        self.child = child
        self.writer = writer

    def copy(self, child):
        return BlockMatrixWrite(child, self.writer)

    def head_str(self):
        return f'"{self.writer.render()}"'

    def _eq(self, other):
        return self.writer == other.writer

    def _compute_type(self, env, agg_env):
        self.child._compute_type()
        self._type = tvoid


class BlockMatrixMultiWrite(IR):
    @typecheck_method(block_matrices=sequenceof(BlockMatrixIR), writer=BlockMatrixMultiWriter)
    def __init__(self, block_matrices, writer):
        super().__init__(*block_matrices)
        self.block_matrices = block_matrices
        self.writer = writer

    def copy(self, *block_matrices):
        return BlockMatrixWrite(block_matrices, self.writer)

    def head_str(self):
        return f'"{self.writer.render()}"'

    def _eq(self, other):
        return self.writer == other.writer

    def _compute_type(self, env, agg_env):
        for x in self.block_matrices:
            x._compute_type()
        self._type = tvoid


class TableToValueApply(IR):
    def __init__(self, child, config):
        super().__init__(child)
        self.child = child
        self.config = config

    @typecheck_method(child=TableIR)
    def copy(self, child):
        return TableToValueApply(child, self.config)

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return other.config == self.config

    def _compute_type(self, env, agg_env):
        name = self.config['name']
        if name == 'ForceCountTable':
            self._type = tint64
        else:
            assert name == 'NPartitionsTable', name
            self._type = tint32


class MatrixToValueApply(IR):
    def __init__(self, child, config):
        super().__init__(child)
        self.child = child
        self.config = config

    @typecheck_method(child=MatrixIR)
    def copy(self, child):
        return MatrixToValueApply(child, self.config)

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return other.config == self.config

    def _compute_type(self, env, agg_env):
        name = self.config['name']
        if name == 'ForceCountMatrixTable':
            self._type = tint64
        elif name == 'NPartitionsMatrixTable':
            self._type = tint32
        elif name == 'MatrixExportEntriesByCol':
            self._type = tvoid
        else:
            assert name == 'MatrixWriteBlockMatrix', name
            self._type = tvoid


class BlockMatrixToValueApply(IR):
    def __init__(self, child, config):
        super().__init__(child)
        self.child = child
        self.config = config

    @typecheck_method(child=BlockMatrixIR)
    def copy(self, child):
        new_instance = self.__class__
        return new_instance(child, self.config)

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return other.config == self.config

    def _compute_type(self, env, agg_env):
        assert self.config['name'] == 'GetElement'
        self._type = tfloat64


class Literal(IR):
    @typecheck_method(typ=hail_type,
                      value=anytype)
    def __init__(self, typ, value):
        super(Literal, self).__init__()
        self._typ: 'hail.HailType' = typ
        self.value = value

    def copy(self):
        return Literal(self._typ, self.value)

    def head_str(self):
        return f'{self._typ._parsable_string()} {dump_json(self._typ._convert_to_json_na(self.value))}'

    def _eq(self, other):
        return other._typ == self._typ and \
               other.value == self.value

    def _compute_type(self, env, agg_env):
        self._type = self._typ


class Join(IR):
    _idx = 0

    @typecheck_method(virtual_ir=IR,
                      temp_vars=sequenceof(str),
                      join_exprs=sequenceof(anytype),
                      join_func=func_spec(1, anytype))
    def __init__(self, virtual_ir, temp_vars, join_exprs, join_func):
        super(Join, self).__init__(virtual_ir)
        self.virtual_ir = virtual_ir
        self.temp_vars = temp_vars
        self.join_exprs = join_exprs
        self.join_func = join_func
        self.idx = Join._idx
        Join._idx += 1

    def copy(self, virtual_ir):
        # FIXME: This is pretty fucked, Joins should probably be tracked on Expression?
        new_instance = self.__class__
        new_instance = new_instance(virtual_ir,
                                    self.temp_vars,
                                    self.join_exprs,
                                    self.join_func)
        new_instance.idx = self.idx
        return new_instance

    def search(self, criteria):
        matches = []
        for e in self.join_exprs:
            matches += e._ir.search(criteria)
        matches += super(Join, self).search(criteria)
        return matches

    def render_head(self, r):
        return self.virtual_ir.render_head(r)

    def render_tail(self, r):
        return self.virtual_ir.render_tail(r)

    def render_children(self, r):
        return self.virtual_ir.render_children(r)

    def _compute_type(self, env, agg_env):
        self.virtual_ir._compute_type(env, agg_env)
        self._type = self.virtual_ir._type


class JavaIR(IR):
    def __init__(self, jir):
        super(JavaIR, self).__init__()
        self._jir = jir
        super().__init__()

    def copy(self):
        return JavaIR(self._jir)

    def render_head(self, r):
        return f'(JavaIR{r.add_jir(self._jir)}'

    def _eq(self, other):
        return self._jir == other._jir

    def _compute_type(self, env, agg_env):
        self._type = dtype(self._jir.typ().toString())


def subst(ir, env, agg_env):
    def _subst(ir, env2=None, agg_env2=None):
        return subst(ir, env2 if env2 else env, agg_env2 if agg_env2 else agg_env)

    def delete(env, name):
        new_env = copy.deepcopy(env)
        if name in new_env:
            del new_env[name]
        return new_env

    if isinstance(ir, Ref):
        return env.get(ir.name, ir)
    elif isinstance(ir, Let):
        return Let(ir.name,
                   _subst(ir.value),
                   _subst(ir.body, env))
    elif isinstance(ir, ArrayMap):
        return ArrayMap(_subst(ir.a),
                        ir.name,
                        _subst(ir.body, delete(env, ir.name)))
    elif isinstance(ir, ArrayFilter):
        return ArrayFilter(_subst(ir.a),
                           ir.name,
                           _subst(ir.body, delete(env, ir.name)))
    elif isinstance(ir, ArrayFlatMap):
        return ArrayFlatMap(_subst(ir.a),
                            ir.name,
                            _subst(ir.body, delete(env, ir.name)))
    elif isinstance(ir, ArrayFold):
        return ArrayFold(_subst(ir.a),
                         _subst(ir.zero),
                         ir.accum_name,
                         ir.value_name,
                         _subst(ir.body, delete(delete(env, ir.accum_name), ir.value_name)))
    elif isinstance(ir, ArrayScan):
        return ArrayScan(_subst(ir.a),
                         _subst(ir.zero),
                         ir.accum_name,
                         ir.value_name,
                         _subst(ir.body, delete(delete(env, ir.accum_name), ir.value_name)))
    elif isinstance(ir, ArrayFor):
        return ArrayFor(_subst(ir.a),
                        ir.value_name,
                        _subst(ir.body, delete(env, ir.value_name)))
    elif isinstance(ir, AggFilter):
        return AggFilter(_subst(ir.cond, agg_env),
                         _subst(ir.agg_ir, agg_env),
                         ir.is_scan)
    elif isinstance(ir, AggExplode):
        return AggExplode(_subst(ir.array, agg_env),
                          ir.name,
                          _subst(ir.agg_body, delete(agg_env, ir.name), delete(agg_env, ir.name)),
                          ir.is_scan)
    elif isinstance(ir, AggGroupBy):
        return AggGroupBy(_subst(ir.key, agg_env),
                          _subst(ir.agg_ir, agg_env),
                          ir.is_scan)
    elif isinstance(ir, ApplyAggOp):
        subst_constr_args = [x.map_ir(lambda x: _subst(x)) for x in ir.constructor_args]
        subst_init_op_args = [x.map_ir(lambda x: _subst(x)) for x in
                              ir.init_op_args] if ir.init_op_args else ir.init_op_args
        subst_seq_op_args = [subst(x, agg_env, {}) for x in ir.seq_op_args]
        return ApplyAggOp(ir.agg_op,
                          subst_constr_args,
                          subst_init_op_args,
                          subst_seq_op_args)
    else:
        assert isinstance(ir, IR)
        return ir.map_ir(lambda x: _subst(x))
