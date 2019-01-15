import copy

import hail
from hail.utils.java import escape_str, escape_id, dump_json, parsable_strings
from hail.expr.types import *
from hail.typecheck import *
from .base_ir import *
from .matrix_writer import MatrixWriter, MatrixNativeMultiWriter


def _env_bind(env, k, v):
    env = env.copy()
    env[k] = v
    return env


class I32(IR):
    @typecheck_method(x=int)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.x)

    def render(self, r):
        return '(I32 {})'.format(self.x)

    def __eq__(self, other):
        return isinstance(other, I32) and \
               other.x == self.x

    def _compute_type(self, env, agg_env):
        self._type = tint32


class I64(IR):
    @typecheck_method(x=int)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.x)

    def render(self, r):
        return '(I64 {})'.format(self.x)

    def __eq__(self, other):
        return isinstance(other, I64) and \
               other.x == self.x

    def _compute_type(self, env, agg_env):
        self._type = tint64


class F32(IR):
    @typecheck_method(x=numeric)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.x)

    def render(self, r):
        return '(F32 {})'.format(self.x)

    def __eq__(self, other):
        return isinstance(other, F32) and \
               other.x == self.x

    def _compute_type(self, env, agg_env):
        self._type = tfloat32

class F64(IR):
    @typecheck_method(x=numeric)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.x)

    def render(self, r):
        return '(F64 {})'.format(self.x)

    def __eq__(self, other):
        return isinstance(other, F64) and \
               other.x == self.x

    def _compute_type(self, env, agg_env):
        self._type = tfloat64


class Str(IR):
    @typecheck_method(x=str)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.x)

    def render(self, r):
        return '(Str "{}")'.format(escape_str(self.x))

    def __eq__(self, other):
        return isinstance(other, Str) and \
               other.x == self.x

    def _compute_type(self, env, agg_env):
        self._type = tstr


class FalseIR(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        new_instance = self.__class__
        return new_instance()

    def render(self, r):
        return '(False)'

    def __eq__(self, other):
        return isinstance(other, FalseIR)

    def _compute_type(self, env, agg_env):
        self._type = tbool


class TrueIR(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        new_instance = self.__class__
        return new_instance()

    def render(self, r):
        return '(True)'

    def __eq__(self, other):
        return isinstance(other, TrueIR)

    def _compute_type(self, env, agg_env):
        self._type = tbool


class Void(IR):
    def __init__(self):
        super().__init__()

    def copy(self):
        new_instance = self.__class__
        return new_instance()

    def render(self, r):
        return '(Void)'

    def __eq__(self, other):
        return isinstance(other, Void)

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

    @typecheck_method(v=IR)
    def copy(self, v):
        new_instance = self.__class__
        return new_instance(v, self._typ)

    def render(self, r):
        return '(Cast {} {})'.format(self._typ._parsable_string(), r(self.v))

    def __eq__(self, other):
        return isinstance(other, Cast) and \
        other.v == self.v and \
        other._typ == self._typ

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

    def copy(self):
        new_instance = self.__class__
        return new_instance(self._typ)

    def render(self, r):
        return '(NA {})'.format(self._typ._parsable_string())

    def __eq__(self, other):
        return isinstance(other, NA) and \
               other._typ == self._typ

    def _compute_type(self, env, agg_env):
        self._type = self._typ


class IsNA(IR):
    @typecheck_method(value=IR)
    def __init__(self, value):
        super().__init__(value)
        self.value = value

    @typecheck_method(value=IR)
    def copy(self, value):
        new_instance = self.__class__
        return new_instance(value)

    def render(self, r):
        return '(IsNA {})'.format(r(self.value))

    def __eq__(self, other):
        return isinstance(other, IsNA) and \
               other.value == self.value

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
        new_instance = self.__class__
        return new_instance(cond, cnsq, altr)

    def render(self, r):
        return '(If {} {} {})'.format(r(self.cond), r(self.cnsq), r(self.altr))

    def __eq__(self, other):
        return isinstance(other, If) and \
               other.cond == self.cond and \
               other.cnsq == self.cnsq and \
               other.altr == self.altr

    def _compute_type(self, env, agg_env):
        self.cond._compute_type(env, agg_env)
        self.cnsq._compute_type(env, agg_env)
        self.altr._compute_type(env, agg_env)
        assert(self.cnsq.typ == self.altr.typ)
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
        new_instance = self.__class__
        return new_instance(self.name, value, body)

    def render(self, r):
        return '(Let {} {} {})'.format(escape_id(self.name), r(self.value), r(self.body))

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, Let) and \
               other.name == self.name and \
               other.value == self.value and \
               other.body == self.body

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
        new_instance = self.__class__
        return new_instance(self.name)

    def render(self, r):
        return '(Ref {})'.format(escape_id(self.name))

    def __eq__(self, other):
        return isinstance(other, Ref) and \
               other.name == self.name

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
        new_instance = self.__class__
        return new_instance(self.name)

    def __eq__(self, other):
        return isinstance(other, TopLevelReference) and \
               other.name == self.name

    def _compute_type(self, env, agg_env):
        assert self.name in env, f'{self.name} not found in {env}'
        self._type = env[self.name]


class ApplyBinaryOp(IR):
    @typecheck_method(op=str, l=IR, r=IR)
    def __init__(self, op, l, r):
        super().__init__(l, r)
        self.op = op
        self.l = l
        self.r = r

    @typecheck_method(l=IR, r=IR)
    def copy(self, l, r):
        new_instance = self.__class__
        return new_instance(self.op, l, r)

    def render(self, r):
        return '(ApplyBinaryPrimOp {} {} {})'.format(escape_id(self.op), r(self.l), r(self.r))

    def __eq__(self, other):
        return isinstance(other, ApplyBinaryOp) and \
               other.op == self.op and \
               other.l == self.l and \
               other.r == self.r

    def _compute_type(self, env, agg_env):
        self.l._compute_type(env, agg_env)
        self.r._compute_type(env, agg_env)
        assert self.l.typ == self.r.typ
        if self.op == '/':
            if self.l.typ == tfloat64:
                self._type = tfloat64
            else:
                self._type = tfloat32
        else:
            self._type = self.l.typ
                

class ApplyUnaryOp(IR):
    @typecheck_method(op=str, x=IR)
    def __init__(self, op, x):
        super().__init__(x)
        self.op = op
        self.x = x

    @typecheck_method(x=IR)
    def copy(self, x):
        new_instance = self.__class__
        return new_instance(self.op, x)

    def render(self, r):
        return '(ApplyUnaryPrimOp {} {})'.format(escape_id(self.op), r(self.x))

    def __eq__(self, other):
        return isinstance(other, ApplyUnaryOp) and \
               other.op == self.op and \
               other.x == self.x

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
        new_instance = self.__class__
        return new_instance(self.op, l, r)

    def render(self, r):
        return '(ApplyComparisonOp {} {} {})'.format(escape_id(self.op), r(self.l), r(self.r))

    def __eq__(self, other):
        return isinstance(other, ApplyComparisonOp) and \
               other.op == self.op and \
               other.l == self.l and \
               other.r == self.r

    def _compute_type(self, env, agg_env):
        self.l._compute_type(env, agg_env)
        self.r._compute_type(env, agg_env)
        self._type = tbool


class MakeArray(IR):
    @typecheck_method(args=sequenceof(IR), element_type=nullable(hail_type))
    def __init__(self, args, element_type):
        super().__init__(*args)
        self.args = args
        self._element_type = element_type

    def copy(self, *args):
        new_instance = self.__class__
        return new_instance(list(args), self._element_type)

    def render(self, r):
        return '(MakeArray {} {})'.format(
            self._element_type._parsable_string() if self._element_type is not None else 'None',
            ' '.join([r(x) for x in self.args]))

    def __eq__(self, other):
        return isinstance(other, MakeArray) and \
               other.args == self.args and \
               other._element_type == self._element_type

    def _compute_type(self, env, agg_env):
        for a in self.args:
            a._compute_type(env, agg_env)
        if self._element_type:
            self._type = self._element_type
        else:
            self._type = tarray(self.args[0].typ)


class ArrayRef(IR):
    @typecheck_method(a=IR, i=IR)
    def __init__(self, a, i):
        super().__init__(a, i)
        self.a = a
        self.i = i

    @typecheck_method(a=IR, i=IR)
    def copy(self, a, i):
        new_instance = self.__class__
        return new_instance(a, i)

    def render(self, r):
        return '(ArrayRef {} {})'.format(r(self.a), r(self.i))

    def __eq__(self, other):
        return isinstance(other, ArrayRef) and \
               other.a == self.a and \
               other.i == self.i

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
        new_instance = self.__class__
        return new_instance(a)

    def render(self, r):
        return '(ArrayLen {})'.format(r(self.a))

    def __eq__(self, other):
        return isinstance(other, ArrayLen) and \
               other.a == self.a

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
        new_instance = self.__class__
        return new_instance(start, stop, step)

    def render(self, r):
        return '(ArrayRange {} {} {})'.format(r(self.start), r(self.stop), r(self.step))

    def __eq__(self, other):
        return isinstance(other, ArrayRange) and \
               other.start == self.start and \
               other.stop == self.stop and \
               other.step == self.step

    def _compute_type(self, env, agg_env):
        self.start._compute_type(env, agg_env)
        self.stop._compute_type(env, agg_env)
        self.step._compute_type(env, agg_env)
        self._type = tarray(tint32)


class ArraySort(IR):
    @typecheck_method(a=IR, ascending=IR, on_key=bool)
    def __init__(self, a, ascending, on_key):
        super().__init__(a, ascending)
        self.a = a
        self.ascending = ascending
        self.on_key = on_key

    @typecheck_method(a=IR, ascending=IR)
    def copy(self, a, ascending):
        new_instance = self.__class__
        return new_instance(a, ascending, self.on_key)

    def render(self, r):
        return '(ArraySort {} {} {})'.format(self.on_key, r(self.a), r(self.ascending))

    def __eq__(self, other):
        return isinstance(other, ArraySort) and \
               other.a == self.a and \
               other.ascending == self.ascending and \
               other.on_key == self.on_key

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
        new_instance = self.__class__
        return new_instance(a)

    def render(self, r):
        return '(ToSet {})'.format(r(self.a))

    def __eq__(self, other):
        return isinstance(other, ToSet) and \
               other.a == self.a

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
        new_instance = self.__class__
        return new_instance(a)

    def render(self, r):
        return '(ToDict {})'.format(r(self.a))

    def __eq__(self, other):
        return isinstance(other, ToDict) and \
               other.a == self.a

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
        new_instance = self.__class__
        return new_instance(a)

    def render(self, r):
        return '(ToArray {})'.format(r(self.a))

    def __eq__(self, other):
        return isinstance(other, ToArray) and \
               other.a == self.a

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
        new_instance = self.__class__
        return new_instance(ordered_collection, elem, self.on_key)

    def render(self, r):
        return '(LowerBoundOnOrderedCollection {} {} {})'.format(self.on_key, r(self.ordered_collection), r(self.elem))

    def __eq__(self, other):
        return isinstance(other, LowerBoundOnOrderedCollection) and \
               other.ordered_collection == self.ordered_collection and \
               other.elem == self.elem and \
               other.on_key == self.on_key

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.elem._compute_type(env, agg_env)
        self._type = tint32

class GroupByKey(IR):
    @typecheck_method(collection=IR)
    def __init__(self, collection):
        super().__init__(collection)
        self.collection = collection

    @typecheck_method(collection=IR)
    def copy(self, collection):
        new_instance = self.__class__
        return new_instance(collection)

    def render(self, r):
        return '(GroupByKey {})'.format(r(self.collection))

    def __eq__(self, other):
        return isinstance(other, GroupByKey) and \
               other.collection == self.collection

    def _compute_type(self, env, agg_env):
        self.collection._compute_type(env, agg_env)
        self._type = tdict(self.collection.typ.element_type.types[0],
                           self.collection.typ.element_type.types[1])

class ArrayMap(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        new_instance = self.__class__
        return new_instance(a, self.name, body)

    def render(self, r):
        return '(ArrayMap {} {} {})'.format(escape_id(self.name), r(self.a), r(self.body))

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, ArrayMap) and \
               other.a == self.a and \
               other.name == self.name and \
               other.body == self.body

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
        new_instance = self.__class__
        return new_instance(a, self.name, body)

    def render(self, r):
        return '(ArrayFilter {} {} {})'.format(escape_id(self.name), r(self.a), r(self.body))

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, ArrayFilter) and \
               other.a == self.a and \
               other.name == self.name and \
               other.body == self.body

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
        new_instance = self.__class__
        return new_instance(a, self.name, body)

    def render(self, r):
        return '(ArrayFlatMap {} {} {})'.format(escape_id(self.name), r(self.a), r(self.body))

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, ArrayFlatMap) and \
               other.a == self.a and \
               other.name == self.name and \
               other.body == self.body

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
        new_instance = self.__class__
        return new_instance(a, zero, self.accum_name, self.value_name, body)

    def render(self, r):
        return '(ArrayFold {} {} {} {} {})'.format(
            escape_id(self.accum_name), escape_id(self.value_name),
            r(self.a), r(self.zero), r(self.body))

    @property
    def bound_variables(self):
        return {self.accum_name, self.value_name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, ArrayFold) and \
               other.a == self.a and \
               other.zero == self.zero and \
               other.accum_name == self.accum_name and \
               other.value_name == self.value_name and \
               other.body == self.body

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
        new_instance = self.__class__
        return new_instance(a, zero, self.accum_name, self.value_name, body)

    def render(self, r):
        return '(ArrayScan {} {} {} {} {})'.format(
            escape_id(self.accum_name), escape_id(self.value_name),
            r(self.a), r(self.zero), r(self.body))

    @property
    def bound_variables(self):
        return {self.accum_name, self.value_name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, ArrayScan) and \
               other.a == self.a and \
               other.zero == self.zero and \
               other.accum_name == self.accum_name and \
               other.value_name == self.value_name and \
               other.body == self.body

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
        new_instance = self.__class__
        return new_instance(left, right, self.l_name, self.r_name, compare, join)

    def render(self, r):
        return '(ArrayLeftJoinDistinct {} {} {} {} {} {})'.format(
            escape_id(self.l_name), escape_id(self.r_name),
            r(self.left), r(self.right), r(self.compare), r(self.join))

    @property
    def bound_variables(self):
        return {self.l_name, self.r_name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, ArrayLeftJoinDistinct) and \
               other.left == self.left and \
               other.right == self.right and \
               other.l_name == self.l_name and \
               other.r_name == self.r_name and \
               other.compare == self.compare and \
               other.join == self.join


class ArrayFor(IR):
    @typecheck_method(a=IR, value_name=str, body=IR)
    def __init__(self, a, value_name, body):
        super().__init__(a, body)
        self.a = a
        self.value_name = value_name
        self.body = body

    @typecheck_method(a=IR, body=IR)
    def copy(self, a, body):
        new_instance = self.__class__
        return new_instance(a, self.value_name, body)

    def render(self, r):
        return '(ArrayFor {} {} {})'.format(escape_id(self.value_name), r(self.a), r(self.body))

    @property
    def bound_variables(self):
        return {self.value_name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, ArrayFor) and \
               other.a == self.a and \
               other.value_name == self.value_name and \
               other.body == self.body

    def _compute_type(self, env, agg_env):
        self.a._compute_type(env, agg_env)
        self.body._compute_type(_env_bind(env, self.value_name, self.a.typ.element_type), agg_env)
        self._type = tvoid


class AggFilter(IR):
    @typecheck_method(cond=IR, agg_ir=IR)
    def __init__(self, cond, agg_ir):
        super().__init__(cond, agg_ir)
        self.cond = cond
        self.agg_ir = agg_ir

    @typecheck_method(cond=IR, agg_ir=IR)
    def copy(self, cond, agg_ir):
        new_instance = self.__class__
        return new_instance(cond, agg_ir)

    def render(self, r):
        return '(AggFilter {} {})'.format(r(self.cond), r(self.agg_ir))

    def __eq__(self, other):
        return isinstance(other, AggFilter) and \
               other.cond == self.cond and \
               other.agg_ir == self.agg_ir

    def _compute_type(self, env, agg_env):
        self.cond._compute_type(agg_env, None)
        self.agg_ir._compute_type(env, agg_env)
        self._type = self.agg_ir.typ


class AggExplode(IR):
    @typecheck_method(array=IR, name=str, agg_body=IR)
    def __init__(self, array, name, agg_body):
        super().__init__(array, agg_body)
        self.name = name
        self.array = array
        self.agg_body = agg_body

    @typecheck_method(array=IR, agg_body=IR)
    def copy(self, array, agg_body):
        new_instance = self.__class__
        return new_instance(array, self.name, agg_body)

    def render(self, r):
        return '(AggExplode {} {} {})'.format(escape_id(self.name), r(self.array), r(self.agg_body))

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, AggExplode) and \
               other.array == self.array and \
               other.name == self.name and \
               other.agg_body == self.agg_body

    def _compute_type(self, env, agg_env):
        self.array._compute_type(agg_env, None)
        self.agg_body._compute_type(env, _env_bind(agg_env, self.name, self.array.typ.element_type))
        self._type = self.agg_body.typ


class AggGroupBy(IR):
    @typecheck_method(key=IR, agg_ir=IR)
    def __init__(self, key, agg_ir):
        super().__init__(key, agg_ir)
        self.key = key
        self.agg_ir = agg_ir

    @typecheck_method(key=IR, agg_ir=IR)
    def copy(self, key, agg_ir):
        new_instance = self.__class__
        return new_instance(key, agg_ir)

    def render(self, r):
        return '(AggGroupBy {} {})'.format(r(self.key), r(self.agg_ir))

    def __eq__(self, other):
        return isinstance(other, AggGroupBy) and \
               other.key == self.key and \
               other.agg_ir == self.agg_ir

    def _compute_type(self, env, agg_env):
        self.key._compute_type(agg_env, None)
        self.agg_ir._compute_type(env, agg_env)
        self._type = tdict(self.key.typ, self.agg_ir.typ)

def _register(registry, name, f):
    if name in registry:
        registry[name].append(f)
    else:
        registry[name] = [f]

_aggregator_registry = {}

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

    def render(self, r):
        return '({} {} ({}) {} ({}))'.format(
            self.__class__.__name__,
            self.agg_op,
            ' '.join([r(x) for x in self.constructor_args]),
            '(' + ' '.join([r(x) for x in self.init_op_args]) + ')' if self.init_op_args else 'None',
            ' '.join([r(x) for x in self.seq_op_args]))

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
        new_instance = self.__class__
        return new_instance(list(xs))

    def render(self, r):
        return '(Begin {})'.format(' '.join([r(x) for x in self.xs]))

    def __eq__(self, other):
        return isinstance(other, Begin) \
               and other.xs == self.xs

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
        new_instance = self.__class__
        assert len(irs) == len(self.fields)
        return new_instance([(n, ir) for (n, _), ir in zip(self.fields, irs)])

    def render(self, r):
        return '(MakeStruct {})'.format(' '.join(['({} {})'.format(escape_id(f), r(x)) for (f, x) in self.fields]))

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
        new_instance = self.__class__
        return new_instance(old, self.fields)

    def render(self, r):
        return '(SelectFields ({}) {})'.format(' '.join(map(escape_id, self.fields)), r(self.old))

    def __eq__(self, other):
        return isinstance(other, SelectFields) and \
               other.old == self.old and \
               other.fields == self.fields

    def _compute_type(self, env, agg_env):
        self.old._compute_type(env, agg_env)
        self._type = self.old.typ._select_fields(self.fields)


class InsertFields(IR):
    @typecheck_method(old=IR, fields=sequenceof(sized_tupleof(str, IR)), field_order=nullable(sequenceof(str)))
    def __init__(self, old, fields, field_order):
        super().__init__(old, *[ir for (f, ir) in fields])
        self.old = old
        self.fields = fields
        self.field_order = field_order

    def copy(self, *args):
        new_instance = self.__class__
        assert len(args) == len(self.fields) + 1
        return new_instance(args[0], [(n, ir) for (n, _), ir in zip(self.fields, args[1:])], self.field_order)

    def render(self, r):
        return '(InsertFields {} {} {})'.format(
            self.old,
            'None' if self.field_order is None else parsable_strings(self.field_order),
            ' '.join(['({} {})'.format(escape_id(f), r(x)) for (f, x) in self.fields]))

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
        new_instance = self.__class__
        return new_instance(o, self.name)

    def render(self, r):
        return '(GetField {} {})'.format(escape_id(self.name), r(self.o))

    @property
    def is_nested_field(self):
        return self.o.is_nested_field

    def __eq__(self, other):
        return isinstance(other, GetField) and \
               other.o == self.o and \
               other.name == self.name

    def _compute_type(self, env, agg_env):
        self.o._compute_type(env, agg_env)
        self._type = self.o.typ[self.name]


class MakeTuple(IR):
    @typecheck_method(elements=sequenceof(IR))
    def __init__(self, elements):
        super().__init__(*elements)
        self.elements = elements

    def copy(self, *args):
        new_instance = self.__class__
        return new_instance(list(args))

    def render(self, r):
        return '(MakeTuple {})'.format(' '.join([r(x) for x in self.elements]))

    def __eq__(self, other):
        return isinstance(other, MakeTuple) and \
               other.elements == self.elements

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
        new_instance = self.__class__
        return new_instance(o, self.idx)

    def render(self, r):
        return '(GetTupleElement {} {})'.format(self.idx, r(self.o))

    def __eq__(self, other):
        return isinstance(other, GetTupleElement) and \
               other.o == self.o and \
               other.idx == self.idx

    def _compute_type(self, env, agg_env):
        self.o._compute_type(env, agg_env)
        self._type = self.o.typ.types[self.idx]


class StringSlice(IR):
    @typecheck_method(s=IR, start=IR, end=IR)
    def __init__(self, s, start, end):
        super().__init__(s, start, end)
        self.s = s
        self.start = start
        self.end = end

    @typecheck_method(s=IR, start=IR, end=IR)
    def copy(self, s, start, end):
        new_instance = self.__class__
        return new_instance(s, start, end)

    def render(self, r):
        return '(StringSlice {} {} {})'.format(r(self.s), r(self.start), r(self.end))

    def __eq__(self, other):
        return isinstance(other, StringSlice) and \
               other.s == self.s and \
               other.start == self.start and \
               other.end == self.end

    def _compute_type(self, env, agg_env):
        self.s._compute_type(env, agg_env)
        self.start._compute_type(env, agg_env)
        self.end._compute_type(env, agg_env)
        self._type = tstr


class StringLength(IR):
    @typecheck_method(s=IR)
    def __init__(self, s):
        super().__init__(s)
        self.s = s

    @typecheck_method(s=IR)
    def copy(self, s):
        new_instance = self.__class__
        return new_instance(s)

    def render(self, r):
        return '(StringLength {})'.format(r(self.s))

    def __eq__(self, other):
        return isinstance(other, StringLength) and \
               other.s == self.s

    def _compute_type(self, env, agg_env):
        self._type = tint32


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
        new_instance = self.__class__
        return new_instance(self.i, self._typ)

    def render(self, r):
        return '(In {} {})'.format(self._typ._parsable_string(), self.i)

    def __eq__(self, other):
        return isinstance(other, In) and \
               other.i == self.i and \
               other._typ == self._typ

    def _compute_type(self, env, agg_env):
        self._type = self._typ


class Die(IR):
    @typecheck_method(message=IR, typ=hail_type)
    def __init__(self, message, typ):
        super().__init__()
        self.message = message
        self._typ = typ

    @property
    def typ(self):
        return self._typ

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.message, self._typ)

    def render(self, r):
        return '(Die {} {})'.format(self._typ._parsable_string(), r(self.message))

    def __eq__(self, other):
        return isinstance(other, Die) and \
               other.message == self.message and \
               other._typ == self._typ

    def _compute_type(self, env, agg_env):
        self._type = self._typ


_function_registry = {}
_seeded_function_registry = {}

def register_function(name, param_types, ret_type):
    _register(_function_registry, name, (param_types, ret_type))

def register_seeded_function(name, param_types, ret_type):
    _register(_seeded_function_registry, name, (param_types, ret_type))

def _lookup_function_return_type(registry, fkind, name, arg_types):
    if name in registry:
        fns = registry[name]
        for f in fns:
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
        new_instance = self.__class__
        return new_instance(self.function, *args)

    def render(self, r):
        return '(Apply {} {})'.format(escape_id(self.function), ' '.join([r(x) for x in self.args]))

    def __eq__(self, other):
        return isinstance(other, Apply) and \
               other.function == self.function and \
               other.args == self.args

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
        new_instance = self.__class__
        return new_instance(self.function, self.seed, *args)

    def render(self, r):
        return '(ApplySeeded {} {} {})'.format(
            escape_id(self.function),
            self.seed,
            ' '.join([r(x) for x in self.args]))

    def __eq__(self, other):
        return isinstance(other, Apply) and \
               other.function == self.function and \
               other.args == self.args

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
        new_instance = self.__class__
        return new_instance(self.argname, function, min, max)

    def render(self, r):
        return '(Uniroot {} {} {} {})'.format(
            escape_id(self.argname), r(self.function), r(self.min), r(self.max))

    @property
    def bound_variables(self):
        return {self.argname} | super().bound_variables

    def __eq__(self, other):
        return isinstance(other, Uniroot) and \
               other.argname == self.argname and \
               other.function == self.function and \
               other.min == self.min and \
               other.max == self.max

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
        new_instance = self.__class__
        return new_instance(child)

    def render(self, r):
        return '(TableCount {})'.format(r(self.child))

    def __eq__(self, other):
        return isinstance(other, TableCount) and \
               other.child == self.child

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
        new_instance = self.__class__
        return new_instance(child)

    def render(self, r):
        return '(TableGetGlobals {})'.format(r(self.child))

    def __eq__(self, other):
        return isinstance(other, TableGetGlobals) and \
               other.child == self.child

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
        new_instance = self.__class__
        return new_instance(child)

    def render(self, r):
        return '(TableCollect {})'.format(r(self.child))

    def __eq__(self, other):
        return isinstance(other, TableCollect) and \
               other.child == self.child

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
        new_instance = self.__class__
        return new_instance(child, query)

    def render(self, r):
        return '(TableAggregate {} {})'.format(r(self.child), r(self.query))

    def __eq__(self, other):
        return isinstance(other, TableAggregate) and \
               other.child == self.child and \
               other.query == self.query

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
        new_instance = self.__class__
        return new_instance(child, query)

    def render(self, r):
        return '(MatrixAggregate {} {})'.format(r(self.child), r(self.query))

    def __eq__(self, other):
        return isinstance(other, MatrixAggregate) and \
               other.child == self.child and \
               other.query == self.query

    def _compute_type(self, env, agg_env):
        self.query._compute_type(self.child.typ.global_env(), self.child.typ.entry_env())
        self._type = self.query.typ


class TableWrite(IR):
    @typecheck_method(child=TableIR, path=str, overwrite=bool, stage_locally=bool, _codec_spec=nullable(str))
    def __init__(self, child, path, overwrite, stage_locally, _codec_spec):
        super().__init__(child)
        self.child = child
        self.path = path
        self.overwrite = overwrite
        self.stage_locally = stage_locally
        self._codec_spec = _codec_spec

    @typecheck_method(child=TableIR)
    def copy(self, child):
        new_instance = self.__class__
        return new_instance(child, self.path, self.overwrite, self.stage_locally, self._codec_spec)

    def render(self, r):
        return '(TableWrite "{}" {} {} {} {})'.format(escape_str(self.path), self.overwrite, self.stage_locally,
                                                      "\"" + escape_str(self._codec_spec) + "\"" if self._codec_spec else "None",
                                                        r(self.child))

    def __eq__(self, other):
        return isinstance(other, TableWrite) and \
               other.child == self.child and \
               other.path == self.path and \
               other.overwrite == self.overwrite and \
               other.stage_locally == self.stage_locally and \
               other._codec_spec == self._codec_spec

    def _compute_type(self, env, agg_env):
        self.child._compute_type()
        self._type = tvoid


class TableExport(IR):
    @typecheck_method(child=TableIR,
                      path=str,
                      types_file=nullable(str),
                      header=bool,
                      export_type=int)
    def __init__(self, child, path, types_file, header, export_type):
        super().__init__(child)
        self.child = child
        self.path = path
        self.types_file = types_file
        self.header = header
        self.export_type = export_type

    @typecheck_method(child=TableIR)
    def copy(self, child):
        new_instance = self.__class__
        return new_instance(child, self.path, self.types_file, self.header, self.export_type)

    def render(self, r):
        return '(TableExport {} "{}" "{}" {} {})'.format(
            r(self.child),
            escape_str(self.path),
            escape_str(self.types_file) if self.types_file else 'None',
            self.header,
            self.export_type)

    def __eq__(self, other):
        return isinstance(other, TableExport) and \
               other.child == self.child and \
               other.path == self.path and \
               other.types_file == self.types_file and \
               other.header == self.header and \
               other.export_type == self.export_type

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
        new_instance = self.__class__
        return new_instance(child, self.matrix_writer)

    def render(self, r):
        return '(MatrixWrite "{}" {})'.format(
            r(self.matrix_writer),
            r(self.child))

    def __eq__(self, other):
        return isinstance(other, MatrixWrite) and \
               other.child == self.child and \
               other.matrix_writer == self.matrix_writer

    def _compute_type(self, env, agg_env):
        self.child._compute_type()
        self._type = tvoid


class MatrixMultiWrite(IR):
    @typecheck_method(children=sequenceof(MatrixIR), writer=MatrixNativeMultiWriter)
    def __init__(self, children, writer):
        super().__init__(*children)
        self.writer = writer

    def copy(self, *children):
        new_instance = self.__class__
        return new_instance(list(children), self.writer)

    def render(self, r):
        return '(MatrixMultiWrite "{}" {})'.format(
            r(self.writer),
            ' '.join(map(r, self.children)))

    def __eq__(self, other):
        return isinstance(other, MatrixMultiWrite) and \
               other.children == self.children and \
               other.writer == self.writer


class TableToValueApply(IR):
    def __init__(self, child, config):
        super().__init__(child)
        self.child = child
        self.config = config

    @typecheck_method(child=TableIR)
    def copy(self, child):
        new_instance = self.__class__
        return new_instance(child, self.config)

    def render(self, r):
        return f'(TableToValueApply {dump_json(self.config)} {r(self.child)})'

    def __eq__(self, other):
        return isinstance(other, TableToValueApply) and other.child == self.child and other.config == self.config

    def _compute_type(self, env, agg_env):
        name = self.config['name']
        assert name == 'ForceCountTable'
        self._type = tint64


class MatrixToValueApply(IR):
    def __init__(self, child, config):
        super().__init__(child)
        self.child = child
        self.config = config

    @typecheck_method(child=MatrixIR)
    def copy(self, child):
        new_instance = self.__class__
        return new_instance(child, self.config)

    def render(self, r):
        return f'(MatrixToValueApply {dump_json(self.config)} {r(self.child)})'

    def __eq__(self, other):
        return isinstance(other, MatrixToValueApply) and other.child == self.child and other.config == self.config

    def _compute_type(self, env, agg_env):
        name = self.config['name']
        assert name == 'ForceCountMatrixTable'
        self._type = tint64


class Literal(IR):
    @typecheck_method(typ=hail_type,
                      value=anytype)
    def __init__(self, typ, value):
        super(Literal, self).__init__()
        self._typ: 'hail.HailType' = typ
        self.value = value

    def copy(self):
        return Literal(self._typ, self.value)

    def render(self, r):
        return f'(Literal {self._typ._parsable_string()} ' \
               f'"{escape_str(self._typ._to_json(self.value))}")'

    def __eq__(self, other):
        return isinstance(other, Literal) and \
               other._typ == self._typ and \
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

    def render(self, r):
        return r(self.virtual_ir)

    def _compute_type(self, env, agg_env):
        self.virtual_ir._compute_type(env, agg_env)
        self._type = self.virtual_ir._type


class JavaIR(IR):
    def __init__(self, jir):
        super(JavaIR, self).__init__()
        self._jir = jir
        super().__init__()

    def render(self, r):
        return f'(JavaIR {r.add_jir(self._jir)})'


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
                         _subst(ir.agg_ir, agg_env))
    elif isinstance(ir, AggExplode):
        return AggExplode(_subst(ir.array, agg_env),
                          ir.name,
                          _subst(ir.agg_body, delete(agg_env, ir.name), delete(agg_env, ir.name)))
    elif isinstance(ir, AggGroupBy):
        return AggGroupBy(_subst(ir.key, agg_env),
                          _subst(ir.agg_ir, agg_env))
    elif isinstance(ir, ApplyAggOp):
        subst_constr_args = [x.map_ir(lambda x: _subst(x)) for x in ir.constructor_args]
        subst_init_op_args = [x.map_ir(lambda x: _subst(x)) for x in ir.init_op_args] if ir.init_op_args else ir.init_op_args
        subst_seq_op_args = [subst(x, agg_env, {}) for x in ir.seq_op_args]
        return ApplyAggOp(ir.agg_op,
                          subst_constr_args,
                          subst_init_op_args,
                          subst_seq_op_args)
    else:
        assert isinstance(ir, IR)
        return ir.map_ir(lambda x: _subst(x))
