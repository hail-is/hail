import hail
from hail.expr.types import hail_type
from hail.typecheck import *
from hail.utils.java import escape_str, escape_id
from .aggsig import AggSignature
from .base_ir import *


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


class Cast(IR):
    @typecheck_method(v=IR, typ=hail_type)
    def __init__(self, v, typ):
        super().__init__(v)
        self.v = v
        self.typ = typ

    @typecheck_method(v=IR)
    def copy(self, v):
        new_instance = self.__class__
        return new_instance(v, self.typ)

    def render(self, r):
        return '(Cast {} {})'.format(self.typ._jtype.parsableString(), r(self.v))

    def __eq__(self, other):
        return isinstance(other, Cast) and \
        other.v == self.v and \
        other.typ == self.typ


class NA(IR):
    @typecheck_method(typ=hail_type)
    def __init__(self, typ):
        super().__init__()
        self.typ = typ

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.typ)

    def render(self, r):
        return '(NA {})'.format(self.typ._jtype.parsableString())

    def __eq__(self, other):
        return isinstance(other, NA) and \
               other.typ == self.typ


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


class Ref(IR):
    @typecheck_method(name=str, typ=nullable(hail_type))
    def __init__(self, name, typ):
        super().__init__()
        self.name = name
        self.typ = typ

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.name, self.typ)

    def render(self, r):
        if self.typ is None:
            return '(Ref {})'.format(escape_id(self.name))
        return '(Ref {} {})'.format(self.typ._jtype.parsableString(), escape_id(self.name))

    def __eq__(self, other):
        return isinstance(other, Ref) and \
               other.name == self.name and \
               other.typ == self.typ


class TopLevelReference(Ref):
    @typecheck_method(name=str)
    def __init__(self, name):
        super().__init__(name, None)

    @property
    def is_nested_field(self):
        return True

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.name)

    def __eq__(self, other):
        return isinstance(other, TopLevelReference) and \
               other.name == self.name


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
        return '(ApplyComparisonOp ({}) {} {})'.format(escape_id(self.op), r(self.l), r(self.r))

    def __eq__(self, other):
        return isinstance(other, ApplyComparisonOp) and \
               other.op == self.op and \
               other.l == self.l and \
               other.r == self.r


class MakeArray(IR):
    @typecheck_method(args=sequenceof(IR), typ=nullable(hail_type))
    def __init__(self, args, typ):
        super().__init__(*args)
        self.args = args
        self.typ = typ

    def copy(self, *args):
        new_instance = self.__class__
        return new_instance(list(args), self.typ)

    def render(self, r):
        return '(MakeArray {} {})'.format(
            self.typ._jtype.parsableString() if self.typ is not None else 'None',
            ' '.join([r(x) for x in self.args]))

    def __eq__(self, other):
        return isinstance(other, MakeArray) and \
               other.args == self.args and \
               other.typ == self.typ


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


class BaseApplyAggOp(IR):
    @typecheck_method(seq_op_args=sequenceof(IR),
                      constructor_args=sequenceof(IR),
                      init_op_args=nullable(sequenceof(IR)),
                      agg_sig=AggSignature,
                      typ=hail_type)
    def __init__(self, seq_op_args, constructor_args, init_op_args, agg_sig, typ):
        init_op_children = [] if init_op_args is None else init_op_args
        super().__init__(*seq_op_args, *constructor_args, *init_op_children)
        self.seq_op_args = seq_op_args
        self.constructor_args = constructor_args
        self.init_op_args = init_op_args
        self.agg_sig = agg_sig
        self.typ = typ

    def copy(self, *args):
        new_instance = self.__class__
        n_seq_op_args = len(self.seq_op_args)
        n_constructor_args = len(self.constructor_args)
        seq_op_args = args[:n_seq_op_args]
        constr_args = args[n_seq_op_args:n_constructor_args + n_seq_op_args]
        init_op_args = args[n_constructor_args + n_seq_op_args:]
        return new_instance(seq_op_args, constr_args, init_op_args if len(init_op_args) != 0 else None, self.agg_sig, self.typ)

    def render(self, r):
        return '({} {} ({}) ({}) {})'.format(
            self.__class__.__name__,
            self.agg_sig,
            ' '.join([r(x) for x in self.seq_op_args]),
            ' '.join([r(x) for x in self.constructor_args]),
            '(' + ' '.join([r(x) for x in self.init_op_args]) + ')' if self.init_op_args else 'None')

    @property
    def aggregations(self):
        assert all(map(lambda c: len(c.aggregations) == 0, self.children))
        return [self]

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               other.seq_op_args == self.seq_op_args and \
               other.constructor_args == self.constructor_args and \
               other.init_op_args == self.init_op_args and \
               other.agg_sig == self.agg_sig


class ApplyAggOp(BaseApplyAggOp):
    @typecheck_method(seq_op_args=sequenceof(IR),
                      constructor_args=sequenceof(IR),
                      init_op_args=nullable(sequenceof(IR)),
                      agg_sig=AggSignature,
                      typ=hail_type)
    def __init__(self, seq_op_args, constructor_args, init_op_args, agg_sig, typ):
        super().__init__(seq_op_args, constructor_args, init_op_args, agg_sig, typ)


class ApplyScanOp(BaseApplyAggOp):
    @typecheck_method(seq_op_args=sequenceof(IR),
                      constructor_args=sequenceof(IR),
                      init_op_args=nullable(sequenceof(IR)),
                      agg_sig=AggSignature,
                      typ=hail_type)
    def __init__(self, seq_op_args, constructor_args, init_op_args, agg_sig, typ):
        super().__init__(seq_op_args, constructor_args, init_op_args, agg_sig, typ)


class InitOp(IR):
    @typecheck_method(i=IR, args=sequenceof(IR), agg_sig=AggSignature)
    def __init__(self, i, args, agg_sig):
        super().__init__(i, *args)
        self.i = i
        self.args = args
        self.agg_sig = agg_sig

    def copy(self, i, *args):
        new_instance = self.__class__
        return new_instance(i, list(args), self.agg_sig)

    def render(self, r):
        return '(InitOp {} {} ({}))'.format(self.agg_sig, r(self.i), ' '.join([r(x) for x in self.args]))

    def __eq__(self, other):
        return isinstance(other, InitOp) and \
               other.i == self.i and \
               other.args == self.args and \
               other.agg_sig == self.agg_sig

class SeqOp(IR):
    @typecheck_method(i=IR, args=sequenceof(IR), agg_sig=AggSignature)
    def __init__(self, i, args, agg_sig):
        super().__init__(i, *args)
        self.i = i
        self.args = args
        self.agg_sig = agg_sig

    def copy(self, i, *args):
        new_instance = self.__class__
        return new_instance(i, list(args), self.agg_sig)

    def render(self, r):
        return '(SeqOp {} {} ({}))'.format(self.agg_sig, r(self.i), ' '.join([r(x) for x in self.args]))

    def __eq__(self, other):
        return isinstance(other, SeqOp) and \
               other.i == self.i and \
               other.args == self.args and \
               other.agg_sig == self.agg_sig


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


class InsertFields(IR):
    @typecheck_method(old=IR, fields=sequenceof(sized_tupleof(str, IR)))
    def __init__(self, old, fields):
        super().__init__(old, *[ir for (f, ir) in fields])
        self.old = old
        self.fields = fields

    def copy(self, *args):
        new_instance = self.__class__
        assert len(args) == len(self.fields) + 1
        return new_instance(args[0], [(n, ir) for (n, _), ir in zip(self.fields, args[1:])])

    def render(self, r):
        return '(InsertFields {} {})'.format(
            self.old,
            ' '.join(['({} {})'.format(escape_id(f), r(x)) for (f, x) in self.fields]))

    def __eq__(self, other):
        return isinstance(other, InsertFields) and \
               other.old == self.old and \
               other.fields == self.fields


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


class In(IR):
    @typecheck_method(i=int, typ=hail_type)
    def __init__(self, i, typ):
        super().__init__()
        self.i = i
        self.typ = typ

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.i, self.typ)

    def render(self, r):
        return '(In {} {})'.format(self.typ._jtype.parsableString(), self.i)

    def __eq__(self, other):
        return isinstance(other, In) and \
               other.i == self.i and \
               other.typ == self.typ


class Die(IR):
    @typecheck_method(message=str, typ=hail_type)
    def __init__(self, message, typ):
        super().__init__()
        self.message = message
        self.typ = typ

    def copy(self):
        new_instance = self.__class__
        return new_instance(self.message, self.typ)

    def render(self, r):
        return '(Die {} "{}")'.format(self.typ._jtype.parsableString(), escape_str(self.message))

    def __eq__(self, other):
        return isinstance(other, Die) and \
               other.message == self.message and \
               other.typ == self.typ


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


class TableWrite(IR):
    @typecheck_method(child=TableIR, path=str, overwrite=bool)
    def __init__(self, child, path, overwrite):
        super().__init__(child)
        self.child = child
        self.path = path
        self.overwrite = overwrite

    @typecheck_method(child=TableIR)
    def copy(self, child):
        new_instance = self.__class__
        return new_instance(child, self.path, self.overwrite)

    def render(self, r):
        return '(TableWrite "{}" {} {})'.format(escape_str(self.path), self.overwrite, r(self.child))

    def __eq__(self, other):
        return isinstance(other, TableWrite) and \
               other.child == self.child and \
               other.path == self.path and \
               other.overwrite == self.overwrite


class TableExport(IR):
    @typecheck_method(child=TableIR,
                      path=str,
                      types_file=str,
                      header=bool,
                      export_type=hail_type)
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
        return '(TableExport "{}" "{}" "{}" {} {})'.format(
            escape_str(self.path),
            escape_str(self.types_file),
            escape_str(self.header),
            self.export_type._jtype.parsableString(),
            r(self.child))

    def __eq__(self, other):
        return isinstance(other, TableExport) and \
               other.child == self.child and \
               other.path == self.path and \
               other.types_file == self.types_file and \
               other.header == self.header and \
               other.export_type == self.export_type


class MatrixWrite(IR):
    @typecheck_method(child=MatrixIR, matrix_writer=str)
    def __init__(self, child, matrix_writer):
        super().__init__(child)
        self.child = child
        self.matrix_writer = matrix_writer

    @typecheck_method(child=MatrixIR)
    def copy(self, child):
        new_instance = self.__class__
        return new_instance(child, self.matrix_writer)

    def render(self, r):
        return '(MatrixWrite {} {})'.format(
            self.matrix_writer, r(self.child))

    def __eq__(self, other):
        return isinstance(other, MatrixWrite) and \
               other.child == self.child and \
               other.matrix_writer == self.matrix_writer


class Literal(IR):
    @typecheck_method(dtype=hail_type,
                      value=anytype)
    def __init__(self, dtype, value):
        super(Literal, self).__init__()
        self.dtype: 'hail.HailType' = dtype
        self.value = value

    def copy(self):
        return Literal(self.dtype, self.value)

    def render(self, r):
        return f'(Literal {self.dtype._jtype.parsableString()} ' \
               f'"{escape_str(self.dtype._to_json(self.value))}")'

    def __eq__(self, other):
        return isinstance(other, Literal) and \
               other.dtype == self.dtype and \
               other.value == self.value


class Join(IR):
    _idx = 0

    @typecheck_method(virtual_ir=IR,
                      temp_vars=sequenceof(str),
                      join_exprs=sequenceof(anytype),
                      join_func=func_spec(1, anytype))
    def __init__(self, virtual_ir, temp_vars, join_exprs, join_func):
        super(Join, self).__init__(*(e._ir for e in join_exprs))
        self.virtual_ir = virtual_ir
        self.temp_vars = temp_vars
        self.join_exprs = join_exprs
        self.join_func = join_func
        self.idx = Join._idx
        Join._idx += 1

    def render(self, r):
        return r(self.virtual_ir)

class JavaIR(IR):
    def __init__(self, jir):
        self._jir = jir

    def render(self, r):
        return f'(JavaIR {r.add_jir(self)})'
