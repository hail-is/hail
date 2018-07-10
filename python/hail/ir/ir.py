from hail.ir.base_ir import *
from hail.utils.java import escape_str, escape_id

class I32(IR):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(I32 {})'.format(self.x)

class I64(IR):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(I64 {})'.format(self.x)

class F32(IR):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(F32 {})'.format(self.x)

class F64(IR):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return '(F64 {})'.format(self.x)

class Str(IR):
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
    def __init__(self, v, typ):
        super().__init__()
        self.v = v
        self.typ = typ

    def __str__(self):
        return '(Cast {} {})'.format(self.typ._jtype.parsableString(), self.v)

class NA(IR):
    def __init__(self, typ):
        super().__init__()
        self.typ = typ

    def __str__(self):
        return '(NA {})'.format(self.typ._jtype.parsableString())

class IsNA(IR):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __str__(self):
        return '(IsNA {})'.format(self.value)

class If(IR):
    def __init__(self, cond, cnsq, altr):
        super().__init__()
        self.cond = cond
        self.cnsq = cnsq
        self.altr = altr

    def __str__(self):
        return '(If {} {} {})'.format(self.cond, self.cnsq, self.altr)

class Let(IR):
    def __init__(self, name, value, body):
        super().__init__()
        self.name = name
        self.value = value
        self.body = body

    def __str__(self):
        return '(Let {} {} {})'.format(escape_id(self.name), self.value, self.body)

class Ref(IR):
    def __init__(self, name, typ):
        super().__init__()
        self.name = name
        self.typ = typ

    def __str__(self):
        return '(Ref {} {})'.format(self.typ._jtype.parsableString(), escape_id(self.name))

class ApplyBinaryOp(IR):
    def __init__(self, op, l, r):
        super().__init__()
        self.op = op
        self.l = l
        self.r = r

    def __str__(self):
        return '(ApplyBinaryPrimOp {} {} {})'.format(escape_id(self.op), self.l, self.r)

class ApplyUnaryOp(IR):
    def __init__(self, op, x):
        super().__init__()
        self.op = op
        self.x = x

    def __str__(self):
        return '(ApplyUnaryPrimOp {} {})'.format(escape_id(self.op), self.x)

class ApplyComparisonOp(IR):
    def __init__(self, op, l, r):
        super().__init__()
        self.op = op
        self.l = l
        self.r = r

    def __str__(self):
        return '(ApplyComparisonOp {} {} {})'.format(self.op, self.l, self.r)

class MakeArray(IR):
    def __init__(self, args, typ):
        super().__init__()
        self.args = args
        self.typ = typ

    def __str__(self):
        return '(MakeArray {} {})'.format(self.typ._jtype.parsableString(), ' '.join([str(x) for x in self.args]))

class ArrayRef(IR):
    def __init__(self, a, i):
        super().__init__()
        self.a = a
        self.i = i

    def __str__(self):
        return '(ArrayRef {} {})'.format(self.a, self.i)

class ArrayLen(IR):
    def __init__(self, a):
        super().__init__()
        self.a = a
        
    def __str__(self):
        return '(ArrayLen {})'.format(self.a)

class ArrayRange(IR):
    def __init__(self, start, stop, step):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def __str__(self):
        return '(ArrayRange {} {} {})'.format(self.start, self.stop, self.step)

class ArraySort(IR):
    def __init__(self, a, ascending, on_key):
        super().__init__()
        self.a = a
        self.ascending = ascending
        self.on_key = on_key

    def __str__(self):
        return '(ArraySort {} {} {})'.format(self.on_key, self.ascending, self.a)

class ToSet(IR):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def __str__(self):
        return '(ToSet {})'.format(self.a)

class ToDict(IR):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def __str__(self):
        return '(ToDict {})'.format(self.a)

class ToArray(IR):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def __str__(self):
        return '(ToArray {})'.format(self.a)

class LowerBoundOnOrderedCollection(IR):
    def __init__(self, ordered_collection, elem, on_key):
        super().__init__()
        self.ordered_collection = ordered_collection
        self.elem = elem
        self.on_key = on_key

    def __str__(self):
        return '(LowerBoundOnOrderedCollection {} {} {})'.format(self.on_key, self.ordered_collection, self.elem)

class GroupByKey(IR):
    def __init__(self, collection):
        super().__init__()
        self.collection = collection

    def __str__(self):
        return '(GroupByKey {})'.format(self.collection)

class ArrayMap(IR):
    def __init__(self, a, name, body):
        super().__init__()
        self.a = a
        self.name = name
        self.body = body

    def __str__(self):
        return '(ArrayMap {} {} {})'.format(escape_id(self.name), self.a, self.body)

class ArrayFilter(IR):
    def __init__(self, a, name, body):
        super().__init__()
        self.a = a
        self.name = name
        self.body = body

    def __str__(self):
        return '(ArrayFilter {} {} {})'.format(escape_id(self.name), self.a, self.body)

class ArrayFlatMap(IR):
    def __init__(self, a, name, body):
        super().__init__()
        self.a = a
        self.name = name
        self.body = body

    def __str__(self):
        return '(ArrayFlatMap {} {} {})'.format(escape_id(self.name), self.a, self.body)

class ArrayFold(IR):
    def __init__(self, a, zero, accum_name, value_name, body):
        super().__init__()
        self.a = a
        self.zero = zero
        self.accum_name = accum_name
        self.value_name = value_name
        self.body = body

    def __str__(self):
        return '(ArrayFold {} {} {} {} {})'.format(
            escape_id(self.accum_name), escape_id(self.value_name), 
            self.a, self.zero, self.body)

class ArrayFor(IR):
    def __init__(self, a, value_name, body):
        super().__init__()
        self.a = a
        self.value_name = value_name
        self.body = body

    def __str__(self):
        return '(ArrayFor {} {} {})'.format(escape_id(self.value_name), self.a, self.body)

class ApplyAggOp(IR):
    def __init__(self, a, constructor_args, init_op_args, agg_sig):
        super().__init__()
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

class InitOp(IR):
    def __init__(self, i, args, agg_sig):
        super().__init__()
        self.i = i
        self.args = args
        self.agg_sig = agg_sig

    def __str__(self):
        return '(InitOp {} {} ({}))'.format(self.agg_sig, self.i, ' '.join([str(x) for x in self.args]))

class SeqOp(IR):
    def __init__(self, i, args, agg_sig):
        super().__init__()
        self.i = i
        self.args = args
        self.agg_sig = agg_sig

    def __str__(self):
        return '(SeqOp {} {} ({}))'.format(self.agg_sig, self.i, ' '.join([str(x) for x in self.args]))

class Begin(IR):
    def __init__(self, xs):
        super().__init__()
        self.xs = xs

    def __str__(self):
        return '(Begin {})'.format(' '.join([str(x) for x in self.xs]))

class MakeStruct(IR):
    def __init__(self, fields):
        super().__init__()
        self.fields = fields

    def __str__(self):
        return '(MakeStruct {})'.format(' '.join(['({} {})'.format(escape_id(f), x) for (f, x) in self.fields]))

class SelectFields(IR):
    def __init__(self, old, fields):
        super().__init__()
        self.old = old
        self.fields = fields

    def __str__(self):
        return '(SelectFields ({}) {})'.format(' '.join(self.fields), self.old)

class InsertFields(IR):
    def __init__(self, old, fields):
        super().__init__()
        self.old = old
        self.fields = fields

    def __str__(self):
        return '(InsertFields {} {})'.format(
            self.old,
            ' '.join(['({} {})'.format(escape_id(f), x) for (f, x) in self.fields]))

class GetField(IR):
    def __init__(self, o, name):
        super().__init__()
        self.o = o
        self.name = name

    def __str__(self):
        return '(GetField {} {})'.format(escape_id(self.name), self.o)

class MakeTuple(IR):
    def __init__(self, types):
        super().__init__()
        self.types = types

    def __str__(self):
        return '(MakeTuple {})'.format(' '.join([str(x) for x in self.types]))

class GetTupleElement(IR):
    def __init__(self, o, idx):
        super().__init__()
        self.o = o
        self.idx = idx

    def __str__(self):
        return '(GetTupleElement {} {})'.format(self.idx, self.o)

class StringSlice(IR):
    def __init__(self, s, start, end):
        super().__init__()
        self.s = s
        self.start = start
        self.end = end

    def __str__(self):
        return '(StringSlice {} {} {})'.format(self.s, self.start, self.end)

class StringLength(IR):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def __str__(self):
        return '(StringLength {})'.format(self.s)

class In(IR):
    def __init__(self, i, typ):
        super().__init__()
        self.i = i
        self.typ = typ

    def __str__(self):
        return '(In {} {})'.format(self.typ._jtype.parsableString(), self.i)

class Die(IR):
    def __init__(self, message, typ):
        super().__init__()
        self.message = message
        self.typ = typ

    def __str__(self):
        return '(Die {} "{}")'.format(self.typ._jtype.parsableString(), escape_str(self.message))

class ApplyIR(IR):
    def __init__(self, function, args):
        super().__init__()
        self.function = function
        self.args = args

    def __str__(self):
        return '(ApplyIR {} {})'.format(escape_id(self.function), ' '.join([str(x) for x in self.args]))

class Apply(IR):
    def __init__(self, function, args):
        super().__init__()
        self.function = function
        self.args = args

    def __str__(self):
        return '(Apply {} {})'.format(escape_id(self.function), ' '.join([str(x) for x in self.args]))

class ApplySpecial(IR):
    def __init__(self, function, args):
        super().__init__()
        self.function = function
        self.args = args

    def __str__(self):
        return '(ApplySpecial {} {})'.format(escape_id(self.function), ' '.join([str(x) for x in self.args]))

class Uniroot(IR):
    def __init__(self, argname, function, min, max):
        super().__init__()
        self.argname = argname
        self.function = function
        self.min = min
        self.max = max

    def __str__(self):
        return '(Uniroot {} {} {} {})'.format(
            escape_id(self.argname), self.function, self.min, self.max)

class TableCount(IR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def __str__(self):
        return '(TableCount {})'.format(self.child)

class TableAggregate(IR):
    def __init__(self, child, query):
        super().__init__()
        self.child = child
        self.query = query

    def __str__(self):
        return '(TableAggregate {} {})'.format(self.child, self.query)

class TableWrite(IR):
    def __init__(self, child, path, overwrite):
        super().__init__()
        self.child = child
        self.path = path
        self.overwrite = overwrite

    def __str__(self):
        return '(TableWrite "{}" {} {})'.format(escape_str(self.path), self.overwrite, self.child)

class TableExport(IR):
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
    def __init__(self, child, matrix_writer):
        super().__init__()
        self.child = child
        self.matrix_writer = matrix_writer

    def __str__(self):
        return '(MatrixWrite {} {})'.format(
            self.matrix_writer, self.child)
