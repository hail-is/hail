import copy
from collections import defaultdict

import decorator

import hail
from hail.expr.types import dtype, HailType, hail_type, tint32, tint64, \
    tfloat32, tfloat64, tstr, tbool, tarray, tstream, tndarray, tset, tdict, \
    tstruct, ttuple, tinterval, tvoid, trngstate
from hail.ir.blockmatrix_writer import BlockMatrixWriter, BlockMatrixMultiWriter
from hail.typecheck import typecheck, typecheck_method, sequenceof, numeric, \
    sized_tupleof, nullable, tupleof, anytype, func_spec
from hail.utils.java import Env, HailUserError, warning
from hail.utils.misc import escape_str, dump_json, parsable_strings, escape_id
from .utils import default_row_uid, default_col_uid, rng_key, unpack_row_uid, unpack_col_uid
from .base_ir import BaseIR, IR, TableIR, MatrixIR, BlockMatrixIR, _env_bind
from .matrix_writer import MatrixWriter, MatrixNativeMultiWriter
from .renderer import Renderer, Renderable, ParensRenderer
from .table_writer import TableWriter


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        return tint32


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        return tint64


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        return tfloat32


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        return tfloat64


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        return tstr


class FalseIR(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        return FalseIR()

    def _ir_name(self):
        return 'False'

    def _compute_type(self, env, agg_env, deep_typecheck):
        return tbool


class TrueIR(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        return TrueIR()

    def _ir_name(self):
        return 'True'

    def _compute_type(self, env, agg_env, deep_typecheck):
        return tbool


class Void(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        return Void()

    def _compute_type(self, env, agg_env, deep_typecheck):
        return tvoid


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.v.compute_type(env, agg_env, deep_typecheck)
        return self._typ


class NA(IR):
    @typecheck_method(typ=hail_type)
    def __init__(self, typ):
        super().__init__()
        self._typ = typ

    def _handle_randomness(self, create_uids):
        if create_uids:
            if isinstance(self.typ.element_type, tstruct):
                new_elt_typ = self.typ.element_type._insert_field(uid_field_name, tint64)
            else:
                new_elt_typ = ttuple(tint64, self.typ.element_type)
            return NA(tstream(new_elt_typ))

    @property
    def typ(self):
        return self._typ

    def _eq(self, other):
        return self._typ == other._typ

    def copy(self):
        return NA(self._typ)

    def head_str(self):
        return self._typ._parsable_string()

    def _compute_type(self, env, agg_env, deep_typecheck):
        return self._typ


class IsNA(IR):
    @typecheck_method(value=IR)
    def __init__(self, value):
        super().__init__(value)
        self.value = value

    @typecheck_method(value=IR)
    def copy(self, value):
        return IsNA(value)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.value.compute_type(env, agg_env, deep_typecheck)
        return tbool


class If(IR):
    @typecheck_method(cond=IR, cnsq=IR, altr=IR)
    def __init__(self, cond, cnsq, altr):
        super().__init__(cond, cnsq, altr)
        self.cond = cond
        self.cnsq = cnsq
        self.altr = altr

    def _handle_randomness(self, create_uids):
        return If(self.cond,
                  self.cnsq.handle_randomness(create_uids),
                  self.altr.handle_randomness(create_uids))

    @typecheck_method(cond=IR, cnsq=IR, altr=IR)
    def copy(self, cond, cnsq, altr):
        return If(cond, cnsq, altr)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.cond.compute_type(env, agg_env, deep_typecheck)
        self.cnsq.compute_type(env, agg_env, deep_typecheck)
        self.altr.compute_type(env, agg_env, deep_typecheck)
        assert (self.cnsq.typ == self.altr.typ)
        return self.cnsq.typ

    def renderable_new_block(self, i):
        return i == 1 or i == 2


class Coalesce(IR):
    @typecheck_method(values=IR)
    def __init__(self, *values):
        super().__init__(*values)
        self.values = values

    @typecheck_method(values=IR)
    def copy(self, *values):
        return Coalesce(*values)

    def _compute_type(self, env, agg_env, deep_typecheck):
        first, *rest = self.values
        first.compute_type(env, agg_env, deep_typecheck)
        for x in rest:
            x.compute_type(env, agg_env, deep_typecheck)
            assert x.typ == first.typ
        return first.typ


class Let(IR):
    @typecheck_method(name=str, value=IR, body=IR)
    def __init__(self, name, value, body):
        if isinstance(value.typ, tstream):
            value = value.handle_randomness(False)
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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.value.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(1)), agg_env, deep_typecheck)
        return self.body.typ

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.value._type
            else:
                value = default_value
            return {self.name: value}
        else:
            return {}


class AggLet(IR):
    @typecheck_method(name=str, value=IR, body=IR, is_scan=bool)
    def __init__(self, name, value, body, is_scan):
        super().__init__(value, body)
        self.name = name
        self.value = value
        self.body = body
        self.is_scan = is_scan

    @typecheck_method(value=IR, body=IR)
    def copy(self, value, body):
        return AggLet(self.name, value, body, self.is_scan)

    def head_str(self):
        return escape_id(self.name) + " " + str(self.is_scan)

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _eq(self, other):
        return other.name == self.name and other.is_scan == self.is_scan

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.value.compute_type(agg_env, None, deep_typecheck)
        self.body.compute_type(env, _env_bind(agg_env, {self.name: self.value.typ}), deep_typecheck)
        return self.body.typ

    def renderable_agg_bindings(self, i, default_value=None):
        if not self.is_scan and i == 1:
            if default_value is None:
                value = self.value._type
            else:
                value = default_value
            return {self.name: value}
        else:
            return {}

    def renderable_scan_bindings(self, i, default_value=None):
        if self.is_scan and i == 1:
            if default_value is None:
                value = self.value._type
            else:
                value = default_value
            return {self.name: value}
        else:
            return {}

    def renderable_uses_agg_context(self, i: int) -> bool:
        return not self.is_scan and i == 0

    def renderable_uses_scan_context(self, i: int) -> bool:
        return self.is_scan and i == 0


class Ref(IR):
    @typecheck_method(name=str, type=nullable(HailType))
    def __init__(self, name, type=None):
        super().__init__()
        self.name = name
        self._free_vars = {name}
        self._typ = type

    def _handle_randomness(self, create_uids):
        if not create_uids:
            return self
        elt = Env.get_uid
        uid = Env.get_uid
        return StreamZip([self, StreamIota(I32(0), I32(1))],
                         [elt, uid],
                         pack_uid(Cast(Ref(uid, tint32), tint64), Ref(elt, tint32)))

    def copy(self):
        return Ref(self.name, self._type)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return other.name == self.name

    def _compute_type(self, env, agg_env, deep_typecheck):
        if deep_typecheck:
            assert self.name in env, f'{self.name} not found in {env}'
            if self._typ is not None:
                assert self._typ == env[self.name]
            return env[self.name]
        else:
            return self._typ


class TopLevelReference(Ref):
    @typecheck_method(name=str, type=nullable(HailType))
    def __init__(self, name, type=None):
        super().__init__(name, type)

    @property
    def is_nested_field(self):
        return True

    def copy(self):
        return TopLevelReference(self.name, self.type)

    def _ir_name(self):
        return 'Ref'


# FIXME: If body uses randomness, create a new uid induction variable
class TailLoop(IR):
    @typecheck_method(name=str, body=IR, params=sequenceof(sized_tupleof(str, IR)))
    def __init__(self, name, body, params):
        super().__init__(*([v for n, v in params] + [body]))
        self.name = name
        self.params = params
        self.body = body

    def copy(self, *children):
        params = children[:-1]
        body = children[-1]
        assert len(params) == len(self.params)
        return TailLoop(self.name, [(n, v) for (n, _), v in zip(self.params, params)], body)

    def head_str(self):
        return f'{escape_id(self.name)} ({" ".join([escape_id(n) for n, _ in self.params])})'

    def _eq(self, other):
        return self.name == other.name

    @property
    def bound_variables(self):
        return {n for n, _ in self.params} | {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        for _, b in self.params:
            b.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(len(self.params))), agg_env, deep_typecheck)
        return self.body.typ

    def renderable_bindings(self, i, default_value=None):
        if i == len(self.params):
            if default_value is None:
                return {self.name: None, **{n: v.typ for n, v in self.params}}
            else:
                value = default_value
                return {self.name: value, **{n: value for n, _ in self.params}}
        else:
            return {}


class Recur(IR):
    @typecheck_method(name=str, args=sequenceof(IR), return_type=hail_type)
    def __init__(self, name, args, return_type):
        super().__init__(*args)
        self.name = name
        self.args = args
        self.return_type = return_type
        self._free_vars = {name}

    def copy(self, args):
        return Recur(self.name, args, self.return_type)

    def head_str(self):
        return f'{escape_id(self.name)} {self.return_type._parsable_string()}'

    def _eq(self, other):
        return other.name == self.name

    def _compute_type(self, env, agg_env, deep_typecheck):
        if deep_typecheck:
            assert self.name in env
        return self.return_type


class ApplyBinaryPrimOp(IR):
    @typecheck_method(op=str, left=IR, right=IR)
    def __init__(self, op, left, right):
        super().__init__(left, right)
        self.op = op
        self.left = left
        self.right = right

    @typecheck_method(left=IR, right=IR)
    def copy(self, left, right):
        return ApplyBinaryPrimOp(self.op, left, right)

    def head_str(self):
        return escape_id(self.op)

    def _eq(self, other):
        return other.op == self.op

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.left.compute_type(env, agg_env, deep_typecheck)
        self.right.compute_type(env, agg_env, deep_typecheck)
        if self.op == '/':
            int_types = [tint32, tint64]
            if self.left.typ in int_types and self.right.typ in int_types:
                return tfloat64
            elif self.left.typ == tfloat64:
                return tfloat64
            else:
                return tfloat32
        else:
            return self.left.typ


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.x.compute_type(env, agg_env, deep_typecheck)
        return self.x.typ


class ApplyComparisonOp(IR):
    @typecheck_method(op=str, left=IR, right=IR)
    def __init__(self, op, left, right):
        super().__init__(left, right)
        self.op = op
        self.left = left
        self.right = right

    @typecheck_method(left=IR, right=IR)
    def copy(self, left, right):
        return ApplyComparisonOp(self.op, left, right)

    def head_str(self):
        return escape_id(self.op)

    def _eq(self, other):
        return other.op == self.op

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.left.compute_type(env, agg_env, deep_typecheck)
        self.right.compute_type(env, agg_env, deep_typecheck)
        if self.op == 'Compare':
            return tint32
        else:
            return tbool


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        for a in self.args:
            a.compute_type(env, agg_env, deep_typecheck)
        return tarray(self.args[0].typ)


class ArrayRef(IR):
    @typecheck_method(a=IR, i=IR, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, a, i, error_id=None, stack_trace=None):
        super().__init__(a, i)
        self.a = a
        self.i = i
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    @typecheck_method(a=IR, i=IR)
    def copy(self, a, i):
        return ArrayRef(a, i, self._error_id, self._stack_trace)

    def head_str(self):
        return str(self._error_id)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        self.i.compute_type(env, agg_env, deep_typecheck)
        return self.a.typ.element_type


class ArraySlice(IR):
    @typecheck_method(a=IR, start=IR, stop=nullable(IR), step=IR, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, a, start, stop, step, error_id=None, stack_trace=None):
        if stop is not None:
            super().__init__(a, start, stop, step)
        else:
            super().__init__(a, start, step)

        self.a = a
        self.start = start
        self.stop = stop
        self.step = step
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    @typecheck_method(a=IR, start=IR, stop=nullable(IR), step=IR)
    def copy(self, a, start, stop, step):
        return ArraySlice(a, start, stop, step, self._error_id, self._stack_trace)

    def head_str(self):
        return str(self._error_id)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        self.start.compute_type(env, agg_env, deep_typecheck)
        if self.stop is not None:
            self.stop.compute_type(env, agg_env, deep_typecheck)
        self.step.compute_type(env, agg_env, deep_typecheck)
        return self.a.typ


class ArrayLen(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return ArrayLen(a)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        return tint32


class ArrayZeros(IR):
    @typecheck_method(length=IR)
    def __init__(self, length):
        super().__init__(length)
        self.length = length

    @typecheck_method(length=IR)
    def copy(self, length):
        return ArrayZeros(length)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.length.compute_type(env, agg_env, deep_typecheck)
        return tarray(tint32)


class StreamIota(IR):
    @typecheck_method(start=IR, step=IR, requires_memory_management_per_element=bool)
    def __init__(self, start, step, requires_memory_management_per_element=False):
        super().__init__(start, step)
        self.start = start
        self.step = step
        self.requires_memory_management_per_element = requires_memory_management_per_element

    def _handle_randomness(self, create_uids):
        if not create_uids:
            return self
        elt = Env.get_uid
        return StreamMap(self, elt, MakeTuple([Cast(Ref(elt, tint32), tint64), Ref(elt, tint32)]))

    @typecheck_method(start=IR, step=IR)
    def copy(self, start, step):
        return StreamIota(start, step,
                          requires_memory_management_per_element=self.requires_memory_management_per_element)

    def head_str(self):
        return f'{self.requires_memory_management_per_element}'

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.start.compute_type(env, agg_env, deep_typecheck)
        self.step.compute_type(env, agg_env, deep_typecheck)
        return tstream(tint32)


class StreamRange(IR):
    @typecheck_method(start=IR, stop=IR, step=IR, requires_memory_management_per_element=bool,
                      error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, start, stop, step, requires_memory_management_per_element=False,
                 error_id=None, stack_trace=None):
        super().__init__(start, stop, step)
        self.start = start
        self.stop = stop
        self.step = step
        self.requires_memory_management_per_element = requires_memory_management_per_element
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    def _handle_randomness(self, create_uids):
        if not create_uids:
            return self
        elt = Env.get_uid
        return StreamMap(self, elt, MakeTuple([Cast(Ref(elt, tint32), tint64), Ref(elt, tint32)]))

    @typecheck_method(start=IR, stop=IR, step=IR)
    def copy(self, start, stop, step):
        return StreamRange(start, stop, step, error_id=self._error_id, stack_trace=self._stack_trace)

    def head_str(self):
        return f'{self._error_id} {self.requires_memory_management_per_element}'

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.start.compute_type(env, agg_env, deep_typecheck)
        self.stop.compute_type(env, agg_env, deep_typecheck)
        self.step.compute_type(env, agg_env, deep_typecheck)
        return tstream(tint32)


class StreamGrouped(IR):
    @typecheck_method(stream=IR, group_size=IR)
    def __init__(self, stream, group_size):
        super().__init__(stream, group_size)
        self.stream = stream
        self.group_size = group_size

    def _handle_randomness(self, create_uids):
        assert(not create_uids)
        self.stream.handle_randomness(False)

    @typecheck_method(stream=IR, group_size=IR)
    def copy(self, stream=IR, group_size=IR):
        return StreamGrouped(stream, group_size)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.stream.compute_type(env, agg_env, deep_typecheck)
        return tstream(self.stream.typ)


class MakeNDArray(IR):
    @typecheck_method(data=IR, shape=IR, row_major=IR, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, data, shape, row_major, error_id=None, stack_trace=None):
        super().__init__(data, shape, row_major)
        self.data = data
        self.shape = shape
        self.row_major = row_major
        self._error_id = error_id
        self._stack_trace = stack_trace

        if error_id is None or stack_trace is None:
            self.save_error_info()

    @typecheck_method(data=IR, shape=IR, row_major=IR)
    def copy(self, data, shape, row_major):
        return MakeNDArray(data, shape, row_major, self._error_id, self._stack_trace)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.data.compute_type(env, agg_env, deep_typecheck)
        self.shape.compute_type(env, agg_env, deep_typecheck)
        self.row_major.compute_type(env, agg_env, deep_typecheck)
        return tndarray(self.data.typ.element_type, len(self.shape.typ))

    def head_str(self):
        return f'{self._error_id}'


class NDArrayShape(IR):
    @typecheck_method(nd=IR)
    def __init__(self, nd):
        super().__init__(nd)
        self.nd = nd

    @typecheck_method(nd=IR)
    def copy(self, nd):
        return NDArrayShape(nd)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        return ttuple(*[tint64 for _ in range(self.nd.typ.ndim)])


class NDArrayReshape(IR):
    @typecheck_method(nd=IR, shape=IR, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, nd, shape, error_id=None, stack_trace=None):
        super().__init__(nd, shape)
        self.nd = nd
        self.shape = shape
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    def copy(self, nd, shape):
        return NDArrayReshape(nd, shape, self._error_id, self._stack_trace)

    def head_str(self):
        return str(self._error_id)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        self.shape.compute_type(env, agg_env, deep_typecheck)
        return tndarray(self.nd.typ.element_type, len(self.shape.typ))


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(1)), agg_env, deep_typecheck)
        return tndarray(self.body.typ, self.nd.typ.ndim)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.nd.typ.element_type
            else:
                value = default_value
            return {self.name: value}
        else:
            return {}


class NDArrayMap2(IR):
    @typecheck_method(left=IR, right=IR, lname=str, rname=str, body=IR, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, left, right, lname, rname, body, error_id=None, stack_trace=None):
        super().__init__(left, right, body)
        self.right = right
        self.left = left
        self.lname = lname
        self.rname = rname
        self.body = body
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    @typecheck_method(l=IR, r=IR, body=IR)
    def copy(self, left, right, body):
        return NDArrayMap2(left, right, self.lname, self.rname, body, self._error_id, self._stack_trace)

    def head_str(self):
        return f'{self._error_id} {escape_id(self.lname)} {escape_id(self.rname)}'

    def _eq(self, other):
        return self.lname == other.lname and \
            self.rname == other.rname

    @property
    def bound_variables(self):
        return {self.lname, self.rname} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.left.compute_type(env, agg_env, deep_typecheck)
        self.right.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(2)), agg_env, deep_typecheck)
        return tndarray(self.body.typ, self.left.typ.ndim)

    def renderable_bindings(self, i, default_value=None):
        if i == 2:
            if default_value is None:
                return {self.lname: self.left.typ.element_type, self.rname: self.right.typ.element_type}
            else:
                return {self.lname: default_value, self.rname: default_value}
        else:
            return {}


class NDArrayRef(IR):
    @typecheck_method(nd=IR, idxs=sequenceof(IR), error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, nd, idxs, error_id=None, stack_trace=None):
        super().__init__(nd, *idxs)
        self.nd = nd
        self.idxs = idxs
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    def copy(self, *args):
        return NDArrayRef(args[0], args[1:], self._error_id, self._stack_trace)

    def head_str(self):
        return str(self._error_id)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        [idx.compute_type(env, agg_env, deep_typecheck) for idx in self.idxs]
        return self.nd.typ.element_type


class NDArraySlice(IR):
    @typecheck_method(nd=IR, slices=IR)
    def __init__(self, nd, slices):
        super().__init__(nd, slices)
        self.nd = nd
        self.slices = slices

    def copy(self, nd, slices):
        return NDArraySlice(nd, slices)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        self.slices.compute_type(env, agg_env, deep_typecheck)

        return tndarray(self.nd.typ.element_type,
                        len([t for t in self.slices.typ.types if isinstance(t, ttuple)]))


class NDArrayReindex(IR):
    @typecheck_method(nd=IR, idx_expr=sequenceof(int))
    def __init__(self, nd, idx_expr):
        super().__init__(nd)
        self.nd = nd
        self.idx_expr = idx_expr

    @typecheck_method(nd=IR)
    def copy(self, nd):
        return NDArrayReindex(nd, self.idx_expr)

    def head_str(self):
        return f'({" ".join([str(i) for i in self.idx_expr])})'

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        n_input_dims = self.nd.typ.ndim
        n_output_dims = len(self.idx_expr)
        assert n_input_dims <= n_output_dims
        assert all([i < n_output_dims for i in self.idx_expr])
        assert all([i in self.idx_expr for i in range(n_output_dims)])

        return tndarray(self.nd.typ.element_type, n_output_dims)


class NDArrayAgg(IR):
    @typecheck_method(nd=IR, axes=sequenceof(int))
    def __init__(self, nd, axes):
        super().__init__(nd)
        self.nd = nd
        self.axes = axes

    @typecheck_method(nd=IR)
    def copy(self, nd):
        return NDArrayAgg(nd, self.axes)

    def head_str(self):
        return f'({" ".join([str(i) for i in self.axes])})'

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        assert len(set(self.axes)) == len(self.axes)
        assert all([axis < self.nd.typ.ndim for axis in self.axes])

        return tndarray(self.nd.typ.element_type, self.nd.typ.ndim - len(self.axes))


class NDArrayMatMul(IR):
    @typecheck_method(left=IR, right=IR, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, left, right, error_id=None, stack_trace=None):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    @typecheck_method(left=IR, right=IR)
    def copy(self, left, right):
        return NDArrayMatMul(left, right, self._error_id, self._stack_trace)

    def head_str(self):
        return str(self._error_id)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.left.compute_type(env, agg_env, deep_typecheck)
        self.right.compute_type(env, agg_env, deep_typecheck)

        ndim = hail.linalg.utils.misc._ndarray_matmul_ndim(self.left.typ.ndim, self.right.typ.ndim)
        from hail.expr.expressions import unify_types
        return tndarray(unify_types(self.left.typ.element_type,
                                    self.right.typ.element_type), ndim)


class NDArrayQR(IR):
    @typecheck_method(nd=IR, mode=str, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, nd, mode, error_id=None, stack_trace=None):
        super().__init__(nd)
        self.nd = nd
        self.mode = mode
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    def copy(self):
        return NDArrayQR(self.nd, self.mode, self._error_id, self._stack_trace)

    def head_str(self):
        return f'{self._error_id} "{self.mode}"'

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)

        if self.mode in ["complete", "reduced"]:
            return ttuple(tndarray(tfloat64, 2), tndarray(tfloat64, 2))
        elif self.mode == "raw":
            return ttuple(tndarray(tfloat64, 2), tndarray(tfloat64, 1))
        elif self.mode == "r":
            return tndarray(tfloat64, 2)
        else:
            raise ValueError("Cannot compute type for mode: " + self.mode)


class NDArraySVD(IR):
    @typecheck_method(nd=IR, full_matrices=bool, compute_uv=bool, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, nd, full_matrices, compute_uv, error_id=None, stack_trace=None):
        super().__init__(nd)
        self.nd = nd
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    def copy(self):
        return NDArraySVD(self.nd, self.full_matrices, self.compute_uv, self._error_id, self._stack_trace)

    def head_str(self):
        return f'{self._error_id} {self.full_matrices} {self.compute_uv}'

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        if self.compute_uv:
            return ttuple(tndarray(tfloat64, 2), tndarray(tfloat64, 1), tndarray(tfloat64, 2))
        else:
            return tndarray(tfloat64, 1)


class NDArrayInv(IR):
    @typecheck_method(nd=IR, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, nd, error_id=None, stack_trace=None):
        super().__init__(nd)
        self.nd = nd
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    def copy(self):
        return NDArrayInv(self.nd, self._error_id, self._stack_trace)

    def head_str(self):
        return str(self._error_id)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        return tndarray(tfloat64, 2)


class NDArrayConcat(IR):
    @typecheck_method(nds=IR, axis=int)
    def __init__(self, nds, axis):
        super().__init__(nds)
        self.nds = nds
        self.axis = axis

    def copy(self):
        return NDArrayConcat(self.nds, self.axis)

    def head_str(self):
        return self.axis

    def _eq(self, other):
        return other.nds == self.nds and \
            other.axis == self.axis

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nds.compute_type(env, agg_env, deep_typecheck)

        return self.nds.typ.element_type


class NDArrayWrite(IR):
    @typecheck_method(nd=IR, path=IR)
    def __init__(self, nd, path):
        super().__init__(nd, path)
        self.nd = nd
        self.path = path

    @typecheck_method(nd=IR, path=IR)
    def copy(self, nd, path):
        return NDArrayWrite(nd, path)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.nd.compute_type(env, agg_env, deep_typecheck)
        self.path.compute_type(env, agg_env, deep_typecheck)
        return tvoid

    @staticmethod
    def is_effectful() -> bool:
        return True


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

    def _eq(self, other):
        return other.l_name == self.l_name and other.r_name == self.r_name

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        return tarray(self.a.typ.element_type)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.a.typ.element_type
            else:
                value = default_value
            return {self.l_name: value, self.r_name: value}
        else:
            return {}


class ToSet(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return ToSet(a)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        return tset(self.a.typ.element_type)


class ToDict(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return ToDict(a)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        return tdict(self.a.typ['key'], self.a.typ['value'])


class ToArray(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        if a.uses_randomness:
            a = a.handle_randomness(False)
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return ToArray(a)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        return tarray(self.a.typ.element_type)


class CastToArray(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    @typecheck_method(a=IR)
    def copy(self, a):
        return CastToArray(a)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        return tarray(self.a.typ.element_type)


class ToStream(IR):
    @typecheck_method(a=IR, requires_memory_management_per_element=bool)
    def __init__(self, a, requires_memory_management_per_element=False):
        super().__init__(a)
        self.a = a
        self.requires_memory_management_per_element = requires_memory_management_per_element

    def _handle_randomness(self, create_uids):
        if not create_uids:
            return self
        uid = Env.get_uid()
        elt = Env.get_uid()
        iota = StreamIota(I32(0), I32(1))
        return StreamZip([self, iota], [elt, uid], MakeTuple([Cast(Ref(uid, tint32), tint64), Ref(elt, self.typ.element_type)]), 'TakeMinLength')

    @typecheck_method(a=IR)
    def copy(self, a):
        return ToStream(a)

    def head_str(self):
        return self.requires_memory_management_per_element

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        return tstream(self.a.typ.element_type)


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.ordered_collection.compute_type(env, agg_env, deep_typecheck)
        self.elem.compute_type(env, agg_env, deep_typecheck)
        return tint32


class GroupByKey(IR):
    @typecheck_method(collection=IR)
    def __init__(self, collection):
        super().__init__(collection)
        self.collection = collection

    @typecheck_method(collection=IR)
    def copy(self, collection):
        return GroupByKey(collection)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.collection.compute_type(env, agg_env, deep_typecheck)
        return tdict(self.collection.typ.element_type.types[0],
                     tarray(self.collection.typ.element_type.types[1]))


static_split_ctr = 0
uid_field_name = '__uid'


def get_static_split_uid():
    global static_split_ctr
    result = static_split_ctr
    assert(result <= 0xFFFF_FFFF_FFFF_FFFF)
    static_split_ctr += 1
    return result


def uid_size(type):
    if isinstance(type, ttuple):
        return len(type)
    return 1


def unify_uid_types(types, tag=False):
    size = max(uid_size(type) for type in types)
    if tag:
        size += 1
    if size == 1:
        return tint64
    return ttuple(tint64 for _ in range(size))


def pad_uid(uid, type, tag=None):
    size = uid_size(uid.typ)
    padded = uid_size(type)
    padding = padded - size
    if tag is not None:
        padding -= 1
    assert(padding >= 0)
    if size == 1:
        fields = (uid,)
    else:
        fields = (GetTupleElement(uid, i) for i in range(size))
    if tag is None:
        tuple = MakeTuple([*(I64(0) for _ in range(padding)), *fields])
    else:
        tuple = MakeTuple([I64(tag), *(I64(0) for _ in range(padding)), *fields])
    return If(IsNA(uid), NA(tuple.typ), tuple)


def concat_uids(uid1, uid2, handle_missing_left=False, handle_missing_right=False):
    size1 = uid_size(uid1)
    if size1 == 1:
        fields1 = (uid1,)
    else:
        fields1 = (GetTupleElement(uid1, i) for i in range(size1))
    if handle_missing_left:
        fields1 = (Coalesce(field, I64(0)) for field in fields1)
    size2 = uid_size(uid2)
    if size2 == 1:
        fields2 = (uid2,)
    else:
        fields2 = (GetTupleElement(uid2, i) for i in range(size2))
    if handle_missing_right:
        fields2 = (Coalesce(field, I64(0)) for field in fields2)
    tuple = MakeTuple([*fields1, *fields2])
    return If(Apply("lor", tbool, IsNA(uid1), IsNA(uid2)), NA(tuple.typ), tuple)


def unpack_uid(stream_type):
    tuple_type = stream_type.element_type
    tuple = Ref(Env.get_uid(), tuple_type)
    if isinstance(tuple_type, tstruct):
        return \
            tuple.name, \
            GetField(tuple, uid_field_name), \
            SelectFields(tuple, [field for field in tuple_type.fields if
                                 not field == uid_field_name])
    else:
        return tuple.name, GetTupleElement(tuple, 0), GetTupleElement(tuple, 1)


def pack_uid(uid, elt):
    if isinstance(elt.typ, tstruct):
        return InsertFields(elt, [uid_field_name, uid])
    else:
        return MakeTuple([uid, elt])


def with_split_rng_state(ir, split, is_scan=None) -> 'BaseIR':
    ref = Ref('__rng_state', trngstate)
    new_state = RNGSplit(ref, split)
    if is_scan is None:
        return Let('__rng_state', new_state, ir)
    else:
        return AggLet('__rng_state', new_state, ir, is_scan)


class StreamMap(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    def _handle_randomness(self, create_uids):
        if not self.body.uses_randomness and not create_uids:
            a = self.a.handle_randomness(False)
            return StreamMap(a, self.name, self.body)

        if isinstance(self.typ.element_type, tstream):
            assert(self.body.uses_randomness and not create_uids)
            a = self.a.handle_randomness(False)
            uid = Env.get_uid()
            elt = Env.get_uid()
            new_body = with_split_rng_state(Let(self.name, elt, self.body, uid))
            return StreamZip([a, StreamIota(I32(0), I32(1))], [elt, uid], new_body, 'TakeMinLength')

        a = self.a.handle_randomness(True)

        tuple, uid, elt = unpack_uid(a.typ)
        new_body = Let(self.name, elt, self.body)
        if self.body.uses_randomness:
            new_body = with_split_rng_state(new_body, uid)
        if create_uids:
            new_body = pack_uid(uid, new_body)
        return StreamMap(a, tuple, new_body)

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        return StreamMap(a, self.name, body)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return self.name == other.name

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(1)), agg_env, deep_typecheck)
        return tstream(self.body.typ)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.a.typ.element_type
            else:
                value = default_value
            return {self.name: value}
        else:
            return {}


class StreamZip(IR):
    @typecheck_method(streams=sequenceof(IR), names=sequenceof(str), body=IR, behavior=str,
                      error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, streams, names, body, behavior, error_id=None, stack_trace=None):
        super().__init__(*streams, body)
        self.streams = streams
        self.names = names
        self.body = body
        self.behavior = behavior
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    def _handle_randomness(self, create_uids):
        if not self.body.uses_randomness and not create_uids:
            new_streams = [stream.handle_randomness(False) for stream in self.streams]
            return StreamZip(new_streams, self.names, self.body, self.behavior, self._error_id, self._stack_trace)

        if self.behavior == 'ExtendNA':
            new_streams = [stream.handle_randomness(True) for stream in self.streams]
            tuples, uids, elts = zip(*(unpack_uid(stream.typ) for stream in new_streams))
            uid_type = unify_uid_types(uid.typ for uid in uids)
            uid = Coalesce(If(IsNA(uid), NA(uid_type), pad_uid(uid, uid_type, i)) for i, uid in enumerate(uids))
            new_body = self.body
            for elt, name in zip(elts, self.names):
                new_body = Let(name, elt, new_body)
            if self.body.uses_randomness:
                new_body = with_split_rng_state(new_body, uid)
            if create_uids:
                pack_uid(uid, new_body)
            return StreamZip(new_streams, tuples, new_body, self.behavior, self._error_id, self._stack_trace)

        new_streams = [self.streams[0].handle_randomness(True), *(stream.handle_randomness(False) for stream in self.streams[1:])]
        tuple, uid, elt = unpack_uid(self.streams[0].typ)
        new_body = Let(self.names[0], elt, self.body)
        if self.body.uses_randomness:
            new_body = with_split_rng_state(new_body, uid)
        if create_uids:
            pack_uid(uid, new_body)
        return StreamZip(new_streams, [tuple, *self.names[1:]], new_body, self.behavior, self._error_id, self._stack_trace)

    @typecheck_method(children=IR)
    def copy(self, *children):
        return StreamZip(children[:-1], self.names, children[-1], self.behavior, self._error_id, self._stack_trace)

    def head_str(self):
        return f'{self._error_id} {escape_id(self.behavior)} ({" ".join(map(escape_id, self.names))})'

    def _eq(self, other):
        return self.names == other.names and self.behavior == other.behavior

    @property
    def bound_variables(self):
        return set(self.names) | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        for a in self.streams:
            a.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(len(self.names))), agg_env, deep_typecheck)
        return tstream(self.body.typ)

    def renderable_bindings(self, i, default_value=None):
        if i == len(self.names):
            return {name: default_value if default_value is not None else a.typ.element_type for name, a in zip(self.names, self.streams)}
        else:
            return {}


class StreamFilter(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    def _handle_randomness(self, create_uids):
        if not self.body.uses_randomness and not create_uids:
            a = self.a.handle_randomness(False)
            return StreamFilter(a, self.name, self.body)

        a = self.a.handle_randomness(True)
        tuple, uid, elt = unpack_uid(a.typ)
        new_body = Let(self.name, elt, self.body)
        if self.body.uses_randomness:
            new_body = with_split_rng_state(new_body, uid)
        result = StreamFilter(a, tuple, new_body)
        if not create_uids:
            result = StreamMap(result, tuple, elt)
        return result

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        return StreamFilter(a, self.name, body)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return self.name == other.name

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(1)), agg_env, deep_typecheck)
        return self.a.typ

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.a.typ.element_type
            else:
                value = default_value
            return {self.name: value}
        else:
            return {}


class StreamFlatMap(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    def _handle_randomness(self, create_uids):
        if not self.body.uses_randomness and not create_uids:
            a = self.a.handle_randomness(False)
            return StreamFlatMap(a, self.name, self.body)

        a = self.a.handle_randomness(True)
        tuple, uid, elt = unpack_uid(a.typ)
        new_body = Let(self.name, elt, self.body)
        new_body = new_body.handle_randomness(create_uids)
        if create_uids:
            tuple2, uid2, elt2 = unpack_uid(new_body.typ)
            combined_uid = MakeTuple([uid, uid2])
            new_body = StreamMap(new_body, tuple2, pack_uid(combined_uid, elt2))
        if self.body.uses_randomness:
            new_body = with_split_rng_state(new_body, uid)
        return StreamFlatMap(a, tuple, new_body)

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        return StreamFlatMap(a, self.name, body)

    def head_str(self):
        return escape_id(self.name)

    def _eq(self, other):
        return self.name == other.name

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(1)), agg_env, deep_typecheck)
        return tstream(self.body.typ.element_type)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.a.typ.element_type
            else:
                value = default_value
            return {self.name: value}
        return {}


class StreamFold(IR):
    @typecheck_method(a=IR, zero=IR, accum_name=str, value_name=str, body=IR)
    def __init__(self, a, zero, accum_name, value_name, body):
        if a.uses_randomness or body.uses_randomness:
            a = a.handle_randomness(create_uids=body.uses_randomness)
        if body.uses_randomness:
            tuple, uid, elt = unpack_uid(a.typ)
            body = Let(value_name, elt, body)
            body = with_split_rng_state(body, uid)
            value_name = tuple

        super().__init__(a, zero, body)
        self.a = a
        self.zero = zero
        self.accum_name = accum_name
        self.value_name = value_name
        self.body = body

    @typecheck_method(a=IR, zero=IR, body=IR)
    def copy(self, a, zero, body):
        return StreamFold(a, zero, self.accum_name, self.value_name, body)

    def head_str(self):
        return f'{escape_id(self.accum_name)} {escape_id(self.value_name)}'

    def _eq(self, other):
        return other.accum_name == self.accum_name and \
            other.value_name == self.value_name

    @property
    def bound_variables(self):
        return {self.accum_name, self.value_name} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        self.zero.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(2)), agg_env, deep_typecheck)
        return self.zero.typ

    def renderable_bindings(self, i, default_value=None):
        if i == 2:
            if default_value is None:
                return {self.accum_name: self.zero.typ, self.value_name: self.a.typ.element_type}
            else:
                return {self.accum_name: default_value, self.value_name: default_value}
        else:
            return {}


class StreamScan(IR):
    @typecheck_method(a=IR, zero=IR, accum_name=str, value_name=str, body=IR)
    def __init__(self, a, zero, accum_name, value_name, body):
        super().__init__(a, zero, body)
        self.a = a
        self.zero = zero
        self.accum_name = accum_name
        self.value_name = value_name
        self.body = body

    def _handle_randomness(self, create_uids):
        if not self.body.uses_randomness and not create_uids:
            a = self.a.handle_randomness(False)
            return StreamScan(a, self.zero, self.accum_name, self.value_name, self.body)

        a = self.a.handle_randomness(True)
        tuple, uid, elt = unpack_uid(a.typ)
        new_body = Let(self.name, elt, self.body)
        if self.body.uses_randomness:
            new_body = with_split_rng_state(new_body, uid)
        if create_uids:
            new_body = pack_uid(uid, new_body)
        return StreamScan(a, self.zero, self.accum_name, tuple, new_body)

    @typecheck_method(a=IR, zero=IR, body=IR)
    def copy(self, a, zero, body):
        return StreamScan(a, zero, self.accum_name, self.value_name, body)

    def head_str(self):
        return f'{escape_id(self.accum_name)} {escape_id(self.value_name)}'

    def _eq(self, other):
        return other.accum_name == self.accum_name and \
            other.value_name == self.value_name

    @property
    def bound_variables(self):
        return {self.accum_name, self.value_name} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        self.zero.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(2)), agg_env, deep_typecheck)
        return tstream(self.body.typ)

    def renderable_bindings(self, i, default_value=None):
        if i == 2:
            if default_value is None:
                return {self.accum_name: self.zero.typ, self.value_name: self.a.typ.element_type}
            else:
                return {self.accum_name: default_value, self.value_name: default_value}
        else:
            return {}


class StreamJoinRightDistinct(IR):
    @typecheck_method(left=IR, right=IR, l_key=sequenceof(str), r_key=sequenceof(str), l_name=str, r_name=str, join=IR, join_type=str)
    def __init__(self, left, right, l_key, r_key, l_name, r_name, join, join_type):
        super().__init__(left, right, join)
        self.left = left
        self.right = right
        self.l_key = l_key
        self.r_key = r_key
        self.l_name = l_name
        self.r_name = r_name
        self.join = join
        self.join_type = join_type

    def _handle_randomness(self, create_uids):
        if not self.join.uses_randomness and not create_uids:
            if self.left.uses_randomness:
                left = self.left.handle_randomness(False)
            if self.right.uses_randomness:
                right = self.right.handle_randomness(False)
            return StreamJoinRightDistinct(left, right, self.l_key, self.r_key, self.l_name, self.r_name, self.join, self.join_type)

        if self.join_type == 'left' or self.join_type == 'inner':
            left = self.left.handle_randomness(True)
            if self.right.uses_randomness:
                right = self.right.handle_randomness(False)
            r_name = self.r_name
            l_name, uid, l_elt = unpack_uid(left.typ)
            new_join = Let(self.l_name, l_elt, self.join)
        elif self.join_type == 'right':
            right = self.right.handle_randomness(True)
            if self.left.uses_randomness:
                left = self.left.handle_randomness(False)
            l_name = self.l_name
            r_name, uid, r_elt = unpack_uid(right.typ)
            new_join = Let(self.r_name, r_elt, self.join)
        else:
            left = self.left.handle_randomness(True)
            right = self.right.handle_randomness(True)
            [l_name, r_name], uids, elts = zip(*(unpack_uid(left.typ), unpack_uid(right.typ)))
            uid_type = unify_uid_types((uid.typ for uid in uids), tag=True)
            uid = If(IsNA(uids[0]), pad_uid(uids[1], uid_type, 1), pad_uid(uids[0], uid_type, 0))
            new_join = Let(self.l_name, elts[0], Let(self.r_name, elts[1], self.join))
        if self.join.uses_randomness:
            new_join = with_split_rng_state(new_join, uid)
        if create_uids:
            new_join = pack_uid(uid, new_join)
        return StreamJoinRightDistinct(left, right, self.l_key, self.r_key, l_name, r_name, new_join, self.join_type)

    @typecheck_method(left=IR, right=IR, join=IR)
    def copy(self, left, right, join):
        return StreamJoinRightDistinct(left, right, self.l_key, self.r_key, self.l_name, self.r_name, join, self.join_type)

    def head_str(self):
        return '({}) ({}) {} {} {}'.format(
            ' '.join([escape_id(x) for x in self.l_key]),
            ' '.join([escape_id(x) for x in self.r_key]),
            self.l_name,
            self.r_name,
            self.join_type)

    def _eq(self, other):
        return other.l_name == self.l_name and \
            other.r_name == self.r_name and \
            other.join_type == self.join_type

    @property
    def bound_variables(self):
        return {self.l_name, self.r_name} | super().bound_variables

    def renderable_bindings(self, i, default_value=None):
        if i == 2:
            if default_value is None:
                return {self.l_name: self.left.typ.element_type,
                        self.r_name: self.right.typ.element_type}
            else:
                return {self.l_name: default_value,
                        self.r_name: default_value}
        else:
            return {}


class StreamFor(IR):
    @typecheck_method(a=IR, value_name=str, body=IR)
    def __init__(self, a, value_name, body):
        if body.uses_randomness:
            a = a.handle_randomness(True)
            tuple, uid, elt = unpack_uid(a.typ)
            body = Let(value_name, elt, body)
            body = with_split_rng_state(body, uid)
            value_name = tuple
        elif a.uses_randomness:
            a = a.handle_randomness(False)

        super().__init__(a, body)
        self.a = a
        self.value_name = value_name
        self.body = body

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        return StreamFor(a, self.value_name, body)

    def head_str(self):
        return escape_id(self.value_name)

    def _eq(self, other):
        return self.value_name == other.value_name

    @property
    def bound_variables(self):
        return {self.value_name} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.a.compute_type(env, agg_env, deep_typecheck)
        self.body.compute_type(_env_bind(env, self.bindings(1)), agg_env, deep_typecheck)
        return tvoid

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.a.typ.element_type
            else:
                value = default_value
            return {self.value_name: value}
        else:
            return {}


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.cond.compute_type(agg_env, None, deep_typecheck)
        self.agg_ir.compute_type(env, agg_env, deep_typecheck)
        return self.agg_ir.typ

    def renderable_uses_agg_context(self, i: int):
        return i == 0 and not self.is_scan

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            return {BaseIR.agg_capability: default_value}
        else:
            return {}

    def renderable_uses_scan_context(self, i: int):
        return i == 0 and self.is_scan

    @classmethod
    def uses_agg_capability(cls) -> bool:
        return True


class AggExplode(IR):
    @typecheck_method(s=IR, name=str, agg_body=IR, is_scan=bool)
    def __init__(self, s, name, agg_body, is_scan):
        if s.uses_randomness or agg_body.uses_agg_randomness(is_scan):
            if agg_body.uses_agg_randomness(is_scan):
                s = s.handle_randomness(True)
                tuple, uid, elt = unpack_uid(s.typ)
                agg_body = Let(self.name, elt, agg_body)
                agg_body = with_split_rng_state(agg_body, uid, is_scan)
            else:
                s = s.handle_randomness(False)
        super().__init__(s, agg_body)
        self.name = name
        self.s = s
        self.agg_body = agg_body
        self.is_scan = is_scan

    @typecheck_method(s=IR, agg_body=IR)
    def copy(self, s, agg_body):
        return AggExplode(s, self.name, agg_body, self.is_scan)

    def head_str(self):
        return f'{escape_id(self.name)} {self.is_scan}'

    def _eq(self, other):
        return self.name == other.name and self.is_scan == other.is_scan

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.s.compute_type(agg_env, None, deep_typecheck)
        self.agg_body.compute_type(env, _env_bind(agg_env, self.agg_bindings(1)), deep_typecheck)
        return self.agg_body.typ

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            return {BaseIR.agg_capability: default_value}
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.s.typ.element_type
            else:
                value = default_value
            return {self.name: value}
        else:
            return {}

    def renderable_scan_bindings(self, i, default_value=None):
        return self.renderable_agg_bindings(i, default_value)

    def renderable_uses_agg_context(self, i: int):
        return i == 0 and not self.is_scan

    def renderable_uses_scan_context(self, i: int):
        return i == 0 and self.is_scan

    @classmethod
    def uses_agg_capability(cls) -> bool:
        return True


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.key.compute_type(agg_env, None, deep_typecheck)
        self.agg_ir.compute_type(env, agg_env, deep_typecheck)
        return tdict(self.key.typ, self.agg_ir.typ)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            return {BaseIR.agg_capability: default_value}
        else:
            return {}

    def renderable_uses_agg_context(self, i: int):
        return i == 0 and not self.is_scan

    def renderable_uses_scan_context(self, i: int):
        return i == 0 and self.is_scan

    @classmethod
    def uses_agg_capability(cls) -> bool:
        return True


class AggArrayPerElement(IR):
    @typecheck_method(array=IR, element_name=str, index_name=str, agg_ir=IR, is_scan=bool)
    def __init__(self, array, element_name, index_name, agg_ir, is_scan):
        if agg_ir.uses_agg_randomness(is_scan):
            agg_ir = with_split_rng_state(agg_ir, Cast(Ref(index_name, tint32), tint64), is_scan)
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
        return f'{escape_id(self.element_name)} {escape_id(self.index_name)} {self.is_scan} False'

    def _eq(self, other):
        return self.element_name == other.element_name and self.index_name == other.index_name and self.is_scan == other.is_scan

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.array.compute_type(agg_env, None, deep_typecheck)
        self.agg_ir.compute_type(_env_bind(env, self.bindings(1)),
                                 _env_bind(agg_env, self.agg_bindings(1)), deep_typecheck)
        return tarray(self.agg_ir.typ)

    @property
    def bound_variables(self):
        return {self.element_name, self.index_name} | super().bound_variables

    def renderable_uses_agg_context(self, i: int):
        return i == 0 and not self.is_scan

    def renderable_uses_scan_context(self, i: int):
        return i == 0 and self.is_scan

    @classmethod
    def uses_agg_capability(cls) -> bool:
        return True

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            value = tint32 if default_value is None else default_value
            return {self.index_name: value, BaseIR.agg_capability: default_value}
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        if i == 1:
            if default_value is None:
                value = self.array.typ.element_type
            else:
                value = default_value
            return {self.element_name: value}
        else:
            return {}

    def renderable_scan_bindings(self, i, default_value=None):
        return self.renderable_agg_bindings(i, default_value)


def _register(registry, name, f):
    registry[name].append(f)


_aggregator_registry = defaultdict(list)


def register_aggregator(name, init_params, seq_params, ret_type):
    _register(_aggregator_registry, name, (init_params, seq_params, ret_type))


def lookup_aggregator_return_type(name, init_args, seq_args):
    if name in _aggregator_registry:
        fns = _aggregator_registry[name]
        for f in fns:
            (init_params, seq_params, ret_type) = f
            for p in init_params:
                p.clear()
            for p in seq_params:
                p.clear()
            if (all(p.unify(a) for p, a in zip(init_params, init_args))
                    and all(p.unify(a) for p, a in zip(seq_params, seq_args))):
                return ret_type.subst()
    raise KeyError(f'aggregator {name}({ ",".join([str(t) for t in seq_args]) }) not found')


class BaseApplyAggOp(IR):
    @typecheck_method(agg_op=str,
                      init_op_args=sequenceof(IR),
                      seq_op_args=sequenceof(IR))
    def __init__(self, agg_op, init_op_args, seq_op_args):
        super().__init__(*init_op_args, *seq_op_args)
        self.agg_op = agg_op
        self.init_op_args = init_op_args
        self.seq_op_args = seq_op_args

    def copy(self, *args):
        new_instance = self.__class__
        n_seq_op_args = len(self.seq_op_args)
        init_op_args = args[:len(self.init_op_args)]
        seq_op_args = args[-n_seq_op_args:]
        return new_instance(self.agg_op, init_op_args, seq_op_args)

    def head_str(self):
        return f'{self.agg_op}'

    # Overloaded to add space after 'agg_op' even if there are no children.
    def render_head(self, r):
        return f'({self._ir_name()} {self.agg_op} '

    def render_children(self, r):
        return [
            ParensRenderer(self.init_op_args),
            ParensRenderer(self.seq_op_args)
        ]

    @property
    def aggregations(self):
        assert all(map(lambda c: len(c.aggregations) == 0, self.children))
        return [self]

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            other.agg_op == self.agg_op and \
            other.init_op_args == self.init_op_args and \
            other.seq_op_args == self.seq_op_args

    def __hash__(self):
        return hash(tuple([self.agg_op,
                           tuple(self.init_op_args),
                           tuple(self.seq_op_args)]))

    def _compute_type(self, env, agg_env, deep_typecheck):
        for a in self.init_op_args:
            a.compute_type(env, agg_env, deep_typecheck)
        for a in self.seq_op_args:
            a.compute_type(agg_env, None, deep_typecheck)

        return lookup_aggregator_return_type(
            self.agg_op,
            [a.typ for a in self.init_op_args],
            [a.typ for a in self.seq_op_args])

    def renderable_new_block(self, i: int) -> bool:
        return i == 0

    def renderable_idx_of_child(self, i: int) -> int:
        if i < len(self.init_op_args):
            return 0
        return 1

    @classmethod
    def uses_agg_capability(cls) -> bool:
        return True


class ApplyAggOp(BaseApplyAggOp):
    @typecheck_method(agg_op=str,
                      init_op_args=sequenceof(IR),
                      seq_op_args=sequenceof(IR))
    def __init__(self, agg_op, init_op_args, seq_op_args):
        super().__init__(agg_op, init_op_args, seq_op_args)

    def renderable_uses_agg_context(self, i: int):
        return i == 1


class ApplyScanOp(BaseApplyAggOp):
    @typecheck_method(agg_op=str,
                      init_op_args=sequenceof(IR),
                      seq_op_args=sequenceof(IR))
    def __init__(self, agg_op, init_op_args, seq_op_args):
        super().__init__(agg_op, init_op_args, seq_op_args)

    def renderable_uses_scan_context(self, i: int):
        return i == 1


class AggFold(IR):
    @typecheck_method(zero=IR, seq_op=IR, comb_op=IR, accum_name=str, other_accum_name=str, is_scan=bool)
    def __init__(self, zero, seq_op, comb_op, accum_name, other_accum_name, is_scan):
        super().__init__(zero, seq_op, comb_op)
        self.zero = zero
        self.seq_op = seq_op
        self.comb_op = comb_op
        self.accum_name = accum_name
        self.other_accum_name = other_accum_name
        self.is_scan = is_scan

        if self.comb_op.free_vars - {accum_name, other_accum_name} != set([]):
            raise HailUserError("The comb_op function of fold cannot reference any fields on the Table or MatrixTable")

    def copy(self, zero, seq_op, comb_op):
        return AggFold(zero, seq_op, comb_op, self.accum_name, self.other_accum_name, self.is_scan)

    def head_str(self):
        return f"{self.accum_name} {self.other_accum_name} {self.is_scan}"

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.zero.compute_type(env, agg_env, deep_typecheck)
        self.seq_op.compute_type(_env_bind(agg_env, self.bindings(1)), None, deep_typecheck)
        self.comb_op.compute_type(self.bindings(2), None, deep_typecheck)

        assert self.zero.typ == self.seq_op.typ
        assert self.zero.typ == self.comb_op.typ

        return self.zero.typ

    def renderable_bindings(self, i: int, default_value=None):
        dict_so_far = {}
        if i == 1 or i == 2:
            if default_value is None:
                dict_so_far[self.accum_name] = self.zero.typ
            else:
                dict_so_far[self.accum_name] = default_value

        if i == 2:
            if default_value is None:
                dict_so_far[self.other_accum_name] = self.zero.typ
            else:
                dict_so_far[self.other_accum_name] = default_value

        return dict_so_far

    def renderable_new_block(self, i: int) -> bool:
        return i > 0


class Begin(IR):
    @typecheck_method(xs=sequenceof(IR))
    def __init__(self, xs):
        super().__init__(*xs)
        self.xs = xs

    def copy(self, *xs):
        return Begin(xs)

    def _compute_type(self, env, agg_env, deep_typecheck):
        for x in self.xs:
            x.compute_type(env, agg_env, deep_typecheck)
        return tvoid


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

    def __hash__(self):
        return hash(tuple(self.fields))

    def _compute_type(self, env, agg_env, deep_typecheck):
        for f, x in self.fields:
            x.compute_type(env, agg_env, deep_typecheck)
        return tstruct(**{f: x.typ for f, x in self.fields})


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.old.compute_type(env, agg_env, deep_typecheck)
        return self.old.typ._select_fields(self.fields)


class SelectedTopLevelReference(SelectFields):
    @typecheck_method(name=str, type=tstruct)
    def __init__(self, name, type=None):
        ref = TopLevelReference(name)
        super().__init__(ref, type.fields)
        self.ref = ref
        self._typ = type

    @property
    def is_nested_field(self):
        return True

    def copy(self, ref):
        if isinstance(ref, TopLevelReference):
            return SelectedTopLevelReference(ref.name, self.type)
        else:
            return SelectFields(ref, self.fields)

    def _ir_name(self):
        return 'SelectFields'

    def _compute_type(self, env, agg_env, deep_typecheck):
        if deep_typecheck:
            self.ref.compute_type(env, agg_env, deep_typecheck)
            assert(self.ref.typ._select_fields(self._typ.fields) == self._typ)
        return self._typ


class InsertFields(IR):
    class IFRenderField(Renderable):
        def __init__(self, field, child):
            super().__init__()
            self.field = field
            self.child = child

        def render_head(self, r: Renderer):
            return f'({self.field} '

        def render_tail(self, r: Renderer):
            return ')'

        def render_children(self, r: Renderer):
            return [self.child]

    @staticmethod
    @typecheck(old=IR, fields=sequenceof(sized_tupleof(str, IR)), field_order=nullable(sequenceof(str)))
    def construct_with_deduplication(old, fields, field_order):
        dd = defaultdict(int)
        for k, v in fields:
            if isinstance(v, GetField) and not isinstance(v.o, Ref):
                dd[v.o] += 1

        replacements = {}
        lets = []
        for k, v in dd.items():
            if v > 1:
                uid = Env.get_uid()
                lets.append((uid, k))
                replacements[k] = (uid, k.typ)

        insert_irs = []
        for k, v in fields:
            if isinstance(v, GetField) and v.o in replacements:
                uid, type = replacements[v.o]
                insert_irs.append((k, GetField(Ref(uid, type), v.name)))
            else:
                insert_irs.append((k, v))

        r = InsertFields(old, insert_irs, field_order)
        for uid, value in lets:
            r = Let(uid, value, r)
        return r

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
            hail.ir.RenderableStr(
                'None' if self.field_order is None else parsable_strings(self.field_order)),
            *(InsertFields.IFRenderField(escape_id(f), x) for f, x in self.fields)
        ]

    def __eq__(self, other):
        return isinstance(other, InsertFields) and \
            other.old == self.old and \
            other.fields == self.fields and \
            other.field_order == self.field_order

    def __hash__(self):
        return hash((self.old, tuple(self.fields), tuple(self.field_order) if self.field_order else None))

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.old.compute_type(env, agg_env, deep_typecheck)
        for f, x in self.fields:
            x.compute_type(env, agg_env, deep_typecheck)
        type = self.old.typ._insert_fields(**{f: x.typ for f, x in self.fields})
        if self.field_order:
            type = tstruct(**{f: type[f] for f in self.field_order})
        return type


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.o.compute_type(env, agg_env, deep_typecheck)
        return self.o.typ[self.name]


class ProjectedTopLevelReference(GetField):
    @typecheck_method(name=str, field=str, type=HailType)
    def __init__(self, name, field, type=None):
        ref = TopLevelReference(name)
        super().__init__(ref, field)
        self.ref = ref
        self.field = field
        self._typ = type

    @property
    def is_nested_field(self):
        return True

    def copy(self, ref):
        if isinstance(ref, TopLevelReference):
            return ProjectedTopLevelReference(ref.name, self.field, self._typ)
        else:
            return GetField(ref, self.field)

    def _ir_name(self):
        return 'GetField'

    def _compute_type(self, env, agg_env, deep_typecheck):
        if deep_typecheck:
            self.ref.compute_type(env, agg_env, deep_typecheck)
            assert(self.ref.typ[self.field] == self._typ)
        return self._typ


class MakeTuple(IR):
    @typecheck_method(elements=sequenceof(IR))
    def __init__(self, elements):
        super().__init__(*elements)
        self.elements = elements

    def copy(self, *args):
        return MakeTuple(args)

    def head_str(self):
        return f'({" ".join([str(i) for i in range(len(self.elements))])})'

    def _compute_type(self, env, agg_env, deep_typecheck):
        for x in self.elements:
            x.compute_type(env, agg_env, deep_typecheck)
        return ttuple(*[x.typ for x in self.elements])


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.o.compute_type(env, agg_env, deep_typecheck)
        return self.o.typ.types[self.idx]


class Die(IR):
    @typecheck_method(message=IR, typ=hail_type, error_id=nullable(int), stack_trace=nullable(str))
    def __init__(self, message, typ, error_id=None, stack_trace=None):
        super().__init__(message)
        self.message = message
        self._typ = typ
        self._error_id = error_id
        self._stack_trace = stack_trace

        if error_id is None or stack_trace is None:
            self.save_error_info()

    def copy(self, message):
        return Die(message, self._typ, self._error_id, self._stack_trace)

    def head_str(self):
        return f'{self._typ._parsable_string()} {self._error_id}'

    def _eq(self, other):
        return other._typ == self._typ

    def _compute_type(self, env, agg_env, deep_typecheck):
        return self._typ

    @staticmethod
    def is_effectful() -> bool:
        return True


class ConsoleLog(IR):
    @typecheck_method(message=IR, result=IR)
    def __init__(self, message, result):
        super().__init__(message, result)
        self.message = message
        self.result = result

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.message.compute_type(env, agg_env, deep_typecheck)
        self.result.compute_type(env, agg_env, deep_typecheck)
        return self.result.typ

    def copy(self, message, result):
        return ConsoleLog(message, result)

    @staticmethod
    def is_effectful() -> bool:
        return True


_function_registry = defaultdict(list)
_seeded_function_registry = defaultdict(list)
_udf_registry = dict()


def clear_session_functions():
    global _udf_registry
    for f in _udf_registry.values():
        remove_function(f._name, f._param_types, f._ret_type, f._type_args)

    _udf_registry = dict()


def remove_function(name, param_types, ret_type, type_args=()):
    f = (param_types, ret_type, type_args)
    bindings = _function_registry[name]
    bindings = [b for b in bindings if b != f]
    if not bindings:
        del _function_registry[name]
    else:
        _function_registry[name] = bindings


def register_function(name, param_types, ret_type, type_args=()):
    _register(_function_registry, name, (param_types, ret_type, type_args))


def register_seeded_function(name, param_types, ret_type):
    _register(_seeded_function_registry, name, (param_types, ret_type))


def udf(*param_types):

    uid = Env.get_uid()

    @decorator.decorator
    def wrapper(__original_func, *args, **kwargs):
        registry = hail.ir.ir._udf_registry
        if uid in registry:
            f = registry[uid]
        else:
            f = hail.experimental.define_function(__original_func, *param_types, _name=uid)
            registry[uid] = f
        return f(*args, **kwargs)

    return wrapper


class Apply(IR):
    @typecheck_method(function=str, return_type=hail_type, args=IR,
                      error_id=nullable(int), stack_trace=nullable(str), type_args=tupleof(hail_type))
    def __init__(self, function, return_type, *args, type_args=(), error_id=None, stack_trace=None,):
        super().__init__(*args)
        self.function = function
        self.return_type = return_type
        self.type_args = type_args
        self.args = args
        self._error_id = error_id
        self._stack_trace = stack_trace
        if error_id is None or stack_trace is None:
            self.save_error_info()

    def copy(self, *args):
        return Apply(self.function, self.return_type, *args, type_args=self.type_args, error_id=self._error_id, stack_trace=self._stack_trace,)

    def head_str(self):
        type_args = "(" + " ".join([a._parsable_string() for a in self.type_args]) + ")"
        return f'{self._error_id} {escape_id(self.function)} {type_args} {self.return_type._parsable_string()}'

    def _eq(self, other):
        return other.function == self.function and \
            other.type_args == self.type_args and \
            other.return_type == self.return_type

    def _compute_type(self, env, agg_env, deep_typecheck):
        for arg in self.args:
            arg.compute_type(env, agg_env, deep_typecheck)

        return self.return_type


class ApplySeeded(IR):
    @typecheck_method(function=str, seed=int, rng_state=IR, return_type=hail_type, args=IR)
    def __init__(self, function, seed, rng_state, return_type, *args):
        if hail.current_backend().requires_lowering:
            warning("Seeded randomness is currently unreliable on the service. "
                    "You may observe some unexpected behavior. Don't use for real work yet.")
        super().__init__(rng_state, *args)
        self.function = function
        self.args = args
        self.rng_state = rng_state
        self.seed = seed
        self.return_type = return_type

    def copy(self, rng_state, *args):
        return ApplySeeded(self.function, self.seed, rng_state, self.return_type, *args)

    def head_str(self):
        return f'{escape_id(self.function)} {self.seed} {self.return_type._parsable_string()}'

    def _eq(self, other):
        return other.function == self.function and \
            other.seed == self.seed and \
            other.return_type == self.return_type

    def _compute_type(self, env, agg_env, deep_typecheck):
        for arg in self.args:
            arg.compute_type(env, agg_env, deep_typecheck)

        return self.return_type


class RNGStateLiteral(IR):
    @typecheck_method(key=sized_tupleof(int, int, int, int))
    def __init__(self, key):
        for k in key:
            assert 0 <= k < 0xFFFFFFFF_FFFFFFFF
        super().__init__()
        self.key = key

    def copy(self):
        return RNGStateLiteral(self.key)

    def head_str(self):
        return f'({" ".join(map(str, self.key))})'

    def _eq(self, other):
        return other.key == self.key

    def _compute_type(self, env, agg_env, deep_typecheck):
        return trngstate


class RNGSplit(IR):
    @typecheck_method(parent_state=IR, dyn_bitstring=IR)
    def __init__(self, parent_state, dyn_bitstring):
        super().__init__(parent_state, dyn_bitstring)
        self.parent_state = parent_state
        self.dyn_bitstring = dyn_bitstring

    def copy(self, parent_state, dyn_bitstring):
        return RNGSplit(parent_state, dyn_bitstring)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.parent_state.compute_type(env, agg_env, deep_typecheck)
        self.dyn_bitstring.compute_type(env, agg_env, deep_typecheck)
        return trngstate


class TableCount(IR):
    @typecheck_method(child=TableIR)
    def __init__(self, child):
        child = child.handle_randomness(None)
        super().__init__(child)
        self.child = child

    @typecheck_method(child=TableIR)
    def copy(self, child):
        return TableCount(child)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return tint64


class TableGetGlobals(IR):
    @typecheck_method(child=TableIR)
    def __init__(self, child):
        child = child.handle_randomness(None)
        super().__init__(child)
        self.child = child

    @typecheck_method(child=TableIR)
    def copy(self, child):
        return TableGetGlobals(child)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ.global_type


class TableCollect(IR):
    @typecheck_method(child=TableIR)
    def __init__(self, child):
        child = child.handle_randomness(None)
        super().__init__(child)
        self.child = child

    @typecheck_method(child=TableIR)
    def copy(self, child):
        return TableCollect(child)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return tstruct(**{'rows': tarray(self.child.typ.row_type),
                          'global': self.child.typ.global_type})


class TableAggregate(IR):
    @typecheck_method(child=TableIR, query=IR)
    def __init__(self, child, query):
        if query.uses_randomness:
            child = child.handle_randomness(uid_field_name)
            uid = GetField(Ref('row', child.typ.row_type), uid_field_name)
            if query.uses_value_randomness:
                query = Let('__rng_state', RNGStateLiteral(rng_key), query)
            if query.uses_agg_randomness(is_scan=False):
                query = AggLet('__rng_state', RNGSplit(RNGStateLiteral(rng_key), uid), query, is_scan=False)
        else:
            child = child.handle_randomness(None)
        super().__init__(child, query)
        self.child = child
        self.query = query

    @typecheck_method(child=TableIR, query=IR)
    def copy(self, child, query):
        return TableAggregate(child, query)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.query.compute_type(self.child.typ.global_env(), self.child.typ.row_env(), deep_typecheck)
        return self.query.typ

    def renderable_new_block(self, i: int):
        return i == 1

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.global_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        return self.child.typ.row_env(default_value) if i == 1 else {}


class MatrixCount(IR):
    @typecheck_method(child=MatrixIR)
    def __init__(self, child):
        child = child.handle_randomness(None, None)
        super().__init__(child)
        self.child = child

    @typecheck_method(child=MatrixIR)
    def copy(self, child):
        return TableCount(child)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return ttuple(tint64, tint32)


class MatrixAggregate(IR):
    @typecheck_method(child=MatrixIR, query=IR)
    def __init__(self, child, query):
        if query.uses_value_randomness:
            query = Let('__rng_state', RNGStateLiteral(rng_key), query)

        if query.uses_agg_randomness(is_scan=False):
            child = child.handle_randomness(default_row_uid, default_col_uid)
            row_uid, old_row = unpack_row_uid(child.typ.row_type, default_row_uid)
            col_uid, old_col = unpack_col_uid(child.typ.col_type, default_col_uid)
            entry_uid = concat_uids(row_uid, col_uid)
            query = AggLet('__rng_state', RNGSplit(RNGStateLiteral(rng_key), entry_uid), query, is_scan=False)
        else:
            child = child.handle_randomness(None, None)

        super().__init__(child, query)
        self.child = child
        self.query = query

    @typecheck_method(child=MatrixIR, query=IR)
    def copy(self, child, query):
        return MatrixAggregate(child, query)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.query.compute_type(self.child.typ.global_env(), self.child.typ.entry_env(), deep_typecheck)
        return self.query.typ

    def renderable_new_block(self, i: int):
        return i == 1

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.global_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        return self.child.typ.entry_env(default_value) if i == 1 else {}


class TableWrite(IR):
    @typecheck_method(child=TableIR, writer=TableWriter)
    def __init__(self, child, writer):
        child = child.handle_randomness(None)
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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return tvoid

    @staticmethod
    def is_effectful() -> bool:
        return True


class MatrixWrite(IR):
    @typecheck_method(child=MatrixIR, matrix_writer=MatrixWriter)
    def __init__(self, child, matrix_writer):
        child = child.handle_randomness(None, None)
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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return tvoid

    @staticmethod
    def is_effectful() -> bool:
        return True


class MatrixMultiWrite(IR):
    @typecheck_method(children=sequenceof(MatrixIR), writer=MatrixNativeMultiWriter)
    def __init__(self, children, writer):
        children = (child.handle_randomness(None, None) for child in children)
        super().__init__(*children)
        self.writer = writer

    def copy(self, *children):
        return MatrixMultiWrite(children, self.writer)

    def head_str(self):
        return f'"{self.writer.render()}"'

    def _eq(self, other):
        return other.writer == self.writer

    def _compute_type(self, env, agg_env, deep_typecheck):
        for x in self.children:
            x.compute_type(deep_typecheck)
        return tvoid

    @staticmethod
    def is_effectful() -> bool:
        return True


class BlockMatrixCollect(IR):
    @typecheck_method(child=BlockMatrixIR)
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def copy(self, child):
        return BlockMatrixCollect(self.child)

    def _eq(self, other):
        return isinstance(other, BlockMatrixCollect) and self.child == other.child

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return tndarray(tfloat64, 2)


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.writer._type()

    @staticmethod
    def is_effectful() -> bool:
        return True


class BlockMatrixMultiWrite(IR):
    @typecheck_method(block_matrices=sequenceof(BlockMatrixIR), writer=BlockMatrixMultiWriter)
    def __init__(self, block_matrices, writer):
        super().__init__(*block_matrices)
        self.block_matrices = block_matrices
        self.writer = writer

    def copy(self, *block_matrices):
        return BlockMatrixMultiWrite(block_matrices, self.writer)

    def head_str(self):
        return f'"{self.writer.render()}"'

    def _eq(self, other):
        return self.writer == other.writer

    def _compute_type(self, env, agg_env, deep_typecheck):
        for x in self.block_matrices:
            x.compute_type(deep_typecheck)
        return tvoid

    @staticmethod
    def is_effectful() -> bool:
        return True


class TableToValueApply(IR):
    def __init__(self, child, config):
        child = child.handle_randomness(None)
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

    def _compute_type(self, env, agg_env, deep_typecheck):
        name = self.config['name']
        if name == 'ForceCountTable':
            return tint64
        elif name == 'TableCalculateNewPartitions':
            return tarray(tinterval(self.child.typ.key_type))
        else:
            assert name == 'NPartitionsTable', name
            return tint32


class MatrixToValueApply(IR):
    def __init__(self, child, config):
        child = child.handle_randomness(None, None)
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

    def _compute_type(self, env, agg_env, deep_typecheck):
        name = self.config['name']
        if name == 'ForceCountMatrixTable':
            return tint64
        elif name == 'NPartitionsMatrixTable':
            return tint32
        elif name == 'MatrixExportEntriesByCol':
            return tvoid
        else:
            assert name == 'MatrixWriteBlockMatrix', name
            return tvoid


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        assert self.config['name'] == 'GetElement'
        self.child.compute_type(deep_typecheck)
        return tfloat64


class Literal(IR):
    @typecheck_method(typ=hail_type,
                      value=anytype)
    def __init__(self, typ, value):
        super(Literal, self).__init__()
        self._typ: HailType = typ
        self.value = value

    def copy(self):
        return Literal(self._typ, self.value)

    def head_str(self):
        return f'{self._typ._parsable_string()} {dump_json(self._typ._convert_to_json_na(self.value))}'

    def _eq(self, other):
        return other._typ == self._typ and \
            other.value == self.value

    def _compute_type(self, env, agg_env, deep_typecheck):
        return self._typ


class LiftMeOut(IR):
    @typecheck_method(child=IR)
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def copy(self, child):
        return LiftMeOut(child)

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.child.compute_type(env, agg_env, deep_typecheck)
        return self.child.typ


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        self.virtual_ir.compute_type(env, agg_env, deep_typecheck)
        return self.virtual_ir.typ


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

    def _compute_type(self, env, agg_env, deep_typecheck):
        return dtype(self._jir.typ().toString())


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
                   _subst(ir.body, delete(env, ir.name)))
    elif isinstance(ir, AggLet):
        return AggLet(ir.name,
                      _subst(ir.value, agg_env, {}),
                      _subst(ir.body, delete(env, ir.name)),
                      ir.is_scan)
    elif isinstance(ir, StreamMap):
        return StreamMap(_subst(ir.a),
                         ir.name,
                         _subst(ir.body, delete(env, ir.name)))
    elif isinstance(ir, StreamFilter):
        return StreamFilter(_subst(ir.a),
                            ir.name,
                            _subst(ir.body, delete(env, ir.name)))
    elif isinstance(ir, StreamFlatMap):
        return StreamFlatMap(_subst(ir.a),
                             ir.name,
                             _subst(ir.body, delete(env, ir.name)))
    elif isinstance(ir, StreamFold):
        return StreamFold(_subst(ir.a),
                          _subst(ir.zero),
                          ir.accum_name,
                          ir.value_name,
                          _subst(ir.body, delete(delete(env, ir.accum_name), ir.value_name)))
    elif isinstance(ir, StreamScan):
        return StreamScan(_subst(ir.a),
                          _subst(ir.zero),
                          ir.accum_name,
                          ir.value_name,
                          _subst(ir.body, delete(delete(env, ir.accum_name), ir.value_name)))
    elif isinstance(ir, StreamFor):
        return StreamFor(_subst(ir.a),
                         ir.value_name,
                         _subst(ir.body, delete(env, ir.value_name)))
    elif isinstance(ir, AggFilter):
        return AggFilter(_subst(ir.cond, agg_env),
                         _subst(ir.agg_ir, agg_env),
                         ir.is_scan)
    elif isinstance(ir, AggExplode):
        return AggExplode(_subst(ir.s, agg_env),
                          ir.name,
                          _subst(ir.agg_body, delete(agg_env, ir.name), delete(agg_env, ir.name)),
                          ir.is_scan)
    elif isinstance(ir, AggGroupBy):
        return AggGroupBy(_subst(ir.key, agg_env),
                          _subst(ir.agg_ir, agg_env),
                          ir.is_scan)
    elif isinstance(ir, ApplyAggOp):
        subst_init_op_args = [x.map_ir(lambda x: _subst(x)) for x in ir.init_op_args]
        subst_seq_op_args = [subst(x, agg_env, {}) for x in ir.seq_op_args]
        return ApplyAggOp(ir.agg_op,
                          subst_init_op_args,
                          subst_seq_op_args)
    elif isinstance(ir, AggFold):
        subst_seq_op = subst(ir.seq_op, agg_env, {})
        return AggFold(ir.zero, subst_seq_op, ir.comb_op, ir.accum_name, ir.other_accum_name, ir.is_scan)
    elif isinstance(ir, AggArrayPerElement):
        return AggArrayPerElement(_subst(ir.array, agg_env),
                                  ir.element_name,
                                  ir.index_name,
                                  _subst(ir.agg_ir, delete(env, ir.index_name),
                                         delete(agg_env, ir.element_name)),
                                  ir.is_scan)
    else:
        assert isinstance(ir, IR)
        return ir.map_ir(lambda x: _subst(x))
