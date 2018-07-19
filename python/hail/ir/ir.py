from typing import *

from .aggsig import AggSignature
from .base_ir import *
from hail.expr.types import hail_type
from hail.typecheck.check import *
from hail.utils.java import escape_str, escape_id, Env


class I32(IR):
    @typecheck_method(x=int)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(I32 {})'.format(self.x)


class I64(IR):
    @typecheck_method(x=int)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(I64 {})'.format(self.x)


class F32(IR):
    @typecheck_method(x=numeric)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(F32 {})'.format(self.x)


class F64(IR):
    @typecheck_method(x=numeric)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(F64 {})'.format(self.x)


class Str(IR):
    @typecheck_method(x=str)
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(Str "{}")'.format(escape_str(self.x))


class FalseIR(IR):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return '(False)'


class TrueIR(IR):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return '(True)'


class Void(IR):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return '(Void)'


class Cast(IR):
    @typecheck_method(v=IR, typ=hail_type)
    def __init__(self, v, typ):
        super().__init__(v)
        self.v = v
        self.typ = typ

    def __str__(self):
        return '(Cast {} {})'.format(self.typ._jtype.parsableString(), self.v)


class NA(IR):
    @typecheck_method(typ=hail_type)
    def __init__(self, typ):
        super().__init__()
        self.typ = typ

    def __str__(self):
        return '(NA {})'.format(self.typ._jtype.parsableString())


class IsNA(IR):
    @typecheck_method(value=IR)
    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        return '(IsNA {})'.format(self.value)


class If(IR):
    @typecheck_method(cond=IR, cnsq=IR, altr=IR)
    def __init__(self, cond, cnsq, altr):
        super().__init__(cond, cnsq, altr)
        self.cond = cond
        self.cnsq = cnsq
        self.altr = altr

    def __str__(self):
        return '(If {} {} {})'.format(self.cond, self.cnsq, self.altr)


class Let(IR):
    @typecheck_method(name=str, value=IR, body=IR)
    def __init__(self, name, value, body):
        super().__init__(value, body)
        self.name = name
        self.value = value
        self.body = body

    def __str__(self):
        return '(Let {} {} {})'.format(escape_id(self.name), self.value, self.body)

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables


class Ref(IR):
    @typecheck_method(name=str, typ=nullable(hail_type))
    def __init__(self, name, typ):
        super().__init__()
        self.name = name
        self.typ = typ

    def __str__(self):
        if self.typ is None:
            return '(Ref {})'.format(escape_id(self.name))
        return '(Ref {} {})'.format(self.typ._jtype.parsableString(), escape_id(self.name))


class TopLevelReference(Ref):
    @typecheck_method(name=str)
    def __init__(self, name):
        super().__init__(name, None)

    @property
    def is_nested_field(self):
        return True


class ApplyBinaryOp(IR):
    @typecheck_method(op=str, l=IR, r=IR)
    def __init__(self, op, l, r):
        super().__init__(l, r)
        self.op = op
        self.l = l
        self.r = r

    def __str__(self):
        return '(ApplyBinaryPrimOp {} {} {})'.format(escape_id(self.op), self.l, self.r)


class ApplyUnaryOp(IR):
    @typecheck_method(op=str, x=IR)
    def __init__(self, op, x):
        super().__init__(x)
        self.op = op
        self.x = x

    def __str__(self):
        return '(ApplyUnaryPrimOp {} {})'.format(escape_id(self.op), self.x)


class ApplyComparisonOp(IR):
    @typecheck_method(op=str, l=IR, r=IR)
    def __init__(self, op, l, r):
        super().__init__(l, r)
        self.op = op
        self.l = l
        self.r = r

    def __str__(self):
        return '(ApplyComparisonOp ({}) {} {})'.format(escape_id(self.op), self.l, self.r)


class MakeArray(IR):
    @typecheck_method(args=sequenceof(IR), typ=hail_type)
    def __init__(self, args, typ):
        super().__init__(*args)
        self.args = args
        self.typ = typ

    def __str__(self):
        return '(MakeArray {} {})'.format(self.typ._jtype.parsableString(), ' '.join([str(x) for x in self.args]))


class ArrayRef(IR):
    @typecheck_method(a=IR, i=IR)
    def __init__(self, a, i):
        super().__init__(a, i)
        self.a = a
        self.i = i

    def __str__(self):
        return '(ArrayRef {} {})'.format(self.a, self.i)


class ArrayLen(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a
        
    def __str__(self):
        return '(ArrayLen {})'.format(self.a)


class ArrayRange(IR):
    @typecheck_method(start=IR, stop=IR, step=IR)
    def __init__(self, start, stop, step):
        super().__init__(start, stop, step)
        self.start = start
        self.stop = stop
        self.step = step

    def __str__(self):
        return '(ArrayRange {} {} {})'.format(self.start, self.stop, self.step)


class ArraySort(IR):
    @typecheck_method(a=IR, ascending=IR, on_key=bool)
    def __init__(self, a, ascending, on_key):
        super().__init__(a, ascending)
        self.a = a
        self.ascending = ascending
        self.on_key = on_key

    def __str__(self):
        return '(ArraySort {} {} {})'.format(self.on_key, self.a, self.ascending)


class ToSet(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    def __str__(self):
        return '(ToSet {})'.format(self.a)


class ToDict(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    def __str__(self):
        return '(ToDict {})'.format(self.a)


class ToArray(IR):
    @typecheck_method(a=IR)
    def __init__(self, a):
        super().__init__(a)
        self.a = a

    def __str__(self):
        return '(ToArray {})'.format(self.a)


class LowerBoundOnOrderedCollection(IR):
    @typecheck_method(ordered_collection=IR, elem=IR, on_key=bool)
    def __init__(self, ordered_collection, elem, on_key):
        super().__init__(ordered_collection, elem)
        self.ordered_collection = ordered_collection
        self.elem = elem
        self.on_key = on_key

    def __str__(self):
        return '(LowerBoundOnOrderedCollection {} {} {})'.format(self.on_key, self.ordered_collection, self.elem)


class GroupByKey(IR):
    @typecheck_method(collection=IR)
    def __init__(self, collection):
        super().__init__(collection)
        self.collection = collection

    def __str__(self):
        return '(GroupByKey {})'.format(self.collection)


class ArrayMap(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    def __str__(self):
        return '(ArrayMap {} {} {})'.format(escape_id(self.name), self.a, self.body)

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables


class ArrayFilter(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    def __str__(self):
        return '(ArrayFilter {} {} {})'.format(escape_id(self.name), self.a, self.body)

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables


class ArrayFlatMap(IR):
    @typecheck_method(a=IR, name=str, body=IR)
    def __init__(self, a, name, body):
        super().__init__(a, body)
        self.a = a
        self.name = name
        self.body = body

    def __str__(self):
        return '(ArrayFlatMap {} {} {})'.format(escape_id(self.name), self.a, self.body)

    @property
    def bound_variables(self):
        return {self.name} | super().bound_variables


class ArrayFold(IR):
    @typecheck_method(a=IR, zero=IR, accum_name=str, value_name=str, body=IR)
    def __init__(self, a, zero, accum_name, value_name, body):
        super().__init__(a, zero, body)
        self.a = a
        self.zero = zero
        self.accum_name = accum_name
        self.value_name = value_name
        self.body = body

    def __str__(self):
        return '(ArrayFold {} {} {} {} {})'.format(
            escape_id(self.accum_name), escape_id(self.value_name), 
            self.a, self.zero, self.body)

    @property
    def bound_variables(self):
        return {self.accum_name, self.value_name} | super().bound_variables


class ArrayFor(IR):
    @typecheck_method(a=IR, value_name=str, body=IR)
    def __init__(self, a, value_name, body):
        super().__init__(a, body)
        self.a = a
        self.value_name = value_name
        self.body = body

    def __str__(self):
        return '(ArrayFor {} {} {})'.format(escape_id(self.value_name), self.a, self.body)

    @property
    def bound_variables(self):
        return {self.value_name} | super().bound_variables


class ApplyAggOp(IR):
    @typecheck_method(a=IR,
                      constructor_args=sequenceof(IR),
                      init_op_args=nullable(sequenceof(IR)),
                      agg_sig=AggSignature)
    def __init__(self, a, constructor_args, init_op_args, agg_sig):
        init_op_children = [] if init_op_args is None else init_op_args
        super().__init__(a, *constructor_args, *init_op_children)
        self.a = a
        self.constructor_args = constructor_args
        self.init_op_args = init_op_args
        self.agg_sig = agg_sig

    def __str__(self):
        return '(ApplyAggOp {} {} ({}) {})'.format(
            self.agg_sig,
            self.a,
            ' '.join([str(x) for x in self.constructor_args]),
            '(' + ' '.join([str(x) for x in self.init_op_args]) + ')' if self.init_op_args else 'None')

    @property
    def aggregations(self):
        assert all(map(lambda c: len(c.aggregations) == 0, self.children))
        return [self]


class InitOp(IR):
    @typecheck_method(i=IR, args=sequenceof(IR), agg_sig=AggSignature)
    def __init__(self, i, args, agg_sig):
        super().__init__(i, *args)
        self.i = i
        self.args = args
        self.agg_sig = agg_sig

    def __str__(self):
        return '(InitOp {} {} ({}))'.format(self.agg_sig, self.i, ' '.join([str(x) for x in self.args]))


class SeqOp(IR):
    @typecheck_method(i=IR, args=sequenceof(IR), agg_sig=AggSignature)
    def __init__(self, i, args, agg_sig):
        super().__init__(i, *args)
        self.i = i
        self.args = args
        self.agg_sig = agg_sig

    def __str__(self):
        return '(SeqOp {} {} ({}))'.format(self.agg_sig, self.i, ' '.join([str(x) for x in self.args]))


class Begin(IR):
    @typecheck_method(xs=sequenceof(IR))
    def __init__(self, xs):
        super().__init__(*xs)
        self.xs = xs

    def __str__(self):
        return '(Begin {})'.format(' '.join([str(x) for x in self.xs]))


class MakeStruct(IR):
    @typecheck_method(fields=sequenceof(sized_tupleof(str, IR)))
    def __init__(self, fields):
        super().__init__(*[ir for (n, ir) in fields])
        self.fields = fields

    def __str__(self):
        return '(MakeStruct {})'.format(' '.join(['({} {})'.format(escape_id(f), x) for (f, x) in self.fields]))


class SelectFields(IR):
    @typecheck_method(old=IR, fields=sequenceof(str))
    def __init__(self, old, fields):
        super().__init__(old)
        self.old = old
        self.fields = fields

    def __str__(self):
        return '(SelectFields ({}) {})'.format(' '.join(map(escape_id, self.fields)), self.old)


class InsertFields(IR):
    @typecheck_method(old=IR, fields=sequenceof(sized_tupleof(str, IR)))
    def __init__(self, old, fields):
        super().__init__(old, *[ir for (f, ir) in fields])
        self.old = old
        self.fields = fields

    def __str__(self):
        return '(InsertFields {} {})'.format(
            self.old,
            ' '.join(['({} {})'.format(escape_id(f), x) for (f, x) in self.fields]))


class GetField(IR):
    @typecheck_method(o=IR, name=str)
    def __init__(self, o, name):
        super().__init__(o)
        self.o = o
        self.name = name

    def __str__(self):
        return '(GetField {} {})'.format(escape_id(self.name), self.o)

    @property
    def is_nested_field(self):
        return self.o.is_nested_field


class MakeTuple(IR):
    @typecheck_method(elements=sequenceof(IR))
    def __init__(self, elements):
        super().__init__(*elements)
        self.elements = elements

    def __str__(self):
        return '(MakeTuple {})'.format(' '.join([str(x) for x in self.elements]))


class GetTupleElement(IR):
    @typecheck_method(o=IR, idx=int)
    def __init__(self, o, idx):
        super().__init__(o)
        self.o = o
        self.idx = idx

    def __str__(self):
        return '(GetTupleElement {} {})'.format(self.idx, self.o)


class StringSlice(IR):
    @typecheck_method(s=IR, start=IR, end=IR)
    def __init__(self, s, start, end):
        super().__init__(s, start, end)
        self.s = s
        self.start = start
        self.end = end

    def __str__(self):
        return '(StringSlice {} {} {})'.format(self.s, self.start, self.end)


class StringLength(IR):
    @typecheck_method(s=IR)
    def __init__(self, s):
        super().__init__(s)
        self.s = s

    def __str__(self):
        return '(StringLength {})'.format(self.s)


class In(IR):
    @typecheck_method(i=int, typ=hail_type)
    def __init__(self, i, typ):
        super().__init__()
        self.i = i
        self.typ = typ

    def __str__(self):
        return '(In {} {})'.format(self.typ._jtype.parsableString(), self.i)


class Die(IR):
    @typecheck_method(message=str, typ=hail_type)
    def __init__(self, message, typ):
        super().__init__()
        self.message = message
        self.typ = typ

    def __str__(self):
        return '(Die {} "{}")'.format(self.typ._jtype.parsableString(), escape_str(self.message))


class Apply(IR):
    @typecheck_method(function=str, args=IR)
    def __init__(self, function, *args):
        super().__init__(*args)
        self.function = function
        self.args = args

    def __str__(self):
        return '(Apply {} {})'.format(escape_id(self.function), ' '.join([str(x) for x in self.args]))


class Uniroot(IR):
    @typecheck_method(argname=str, function=IR, min=IR, max=IR)
    def __init__(self, argname, function, min, max):
        super().__init__(function, min, max)
        self.argname = argname
        self.function = function
        self.min = min
        self.max = max

    def __str__(self):
        return '(Uniroot {} {} {} {})'.format(
            escape_id(self.argname), self.function, self.min, self.max)

    @property
    def bound_variables(self):
        return {self.argname} | super().bound_variables


class TableCount(IR):
    @typecheck_method(child=TableIR)
    def __init__(self, child):
        super().__init__()
        self.child = child

    def __str__(self):
        return '(TableCount {})'.format(self.child)


class TableAggregate(IR):
    @typecheck_method(child=TableIR, query=IR)
    def __init__(self, child, query):
        super().__init__(query)
        self.child = child
        self.query = query

    def __str__(self):
        return '(TableAggregate {} {})'.format(self.child, self.query)


class MatrixAggregate(IR):
    @typecheck_method(child=MatrixIR, query=IR)
    def __init__(self, child, query):
        super().__init__(query)
        self.child = child
        self.query = query

    def __str__(self):
        return '(MatrixAggregate {} {})'.format(self.child, self.query)


class TableWrite(IR):
    @typecheck_method(child=TableIR, path=str, overwrite=bool)
    def __init__(self, child, path, overwrite):
        super().__init__()
        self.child = child
        self.path = path
        self.overwrite = overwrite

    def __str__(self):
        return '(TableWrite "{}" {} {})'.format(escape_str(self.path), self.overwrite, self.child)


class TableExport(IR):
    @typecheck_method(child=TableIR,
                      path=str,
                      types_file=str,
                      header=bool,
                      export_type=hail_type)
    def __init__(self, child, path, types_file, header, export_type):
        super().__init__()
        self.child = child
        self.path = path
        self.types_file = types_file
        self.header = header
        self.export_type = export_type

    def __str__(self):
        return '(TableExport "{}" "{}" "{}" {} {})'.format(
            escape_str(self.path),
            escape_str(self.types_file),
            escape_str(self.header),
            self.export_type._jtype.parsableString(),
            self.child)


class MatrixWrite(IR):
    @typecheck_method(child=MatrixIR, matrix_writer=str)
    def __init__(self, child, matrix_writer):
        super().__init__()
        self.child = child
        self.matrix_writer = matrix_writer

    def __str__(self):
        return '(MatrixWrite {} {})'.format(
            self.matrix_writer, self.child)


class Broadcast(IR):
    @typecheck_method(value=anytype, dtype=hail_type)
    def __init__(self, value, dtype):
        super(Broadcast, self).__init__()
        self.value = value
        self.dtype = dtype
        self.uid = Env.get_uid()

    def __str__(self):
        return str(GetField(TopLevelReference('global'), self.uid))


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

    def __str__(self):
        return str(self.virtual_ir)
