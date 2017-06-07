from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.java import *
from hail.htypes import *
from hail.representation import *


def to_expr(arg):
    if isinstance(arg, Column):
        return arg.expr
    elif isinstance(arg, str):
        return "\"" + arg + "\""
    elif isinstance(arg, bool):
        return "true" if arg else "false"
    elif isinstance(arg, long):
        return "{}.toInt64()".format(arg)
    elif isinstance(arg, list):
        return "[" + ", ".join([to_expr(a) for a in arg]) + "]"
    elif isinstance(arg, set):
        return "let xfsjd = [" + ", ".join([to_expr(a) for a in arg]) + "] in xfsjd.toSet()"
    elif isinstance(arg, Variant):
        return "Variant(\"" + str(arg) + "\")"
    elif isinstance(arg, Locus):
        return "Locus(\"" + str(arg) + "\")"
    elif isinstance(arg, Interval):
        return "Interval(\"" + str(arg) + "\")"
    elif isinstance(arg, Struct):
        return "{" + ", ".join([k + ":" + to_expr(v) for k, v in arg._attrs.iteritems()]) + "}"  # FIXME: Struct constructor should take kwargs (ordered in Python 3.6)
    elif callable(arg):
        return arg
    else:
        return str(arg)


@decorator
def args_to_expr(func, *args):
    exprs = [to_expr(arg) if not arg is None else arg for arg in args]
    return func(*exprs)


def convert_column(x):
    if isinstance(x, Column) and x.typ.__class__ in typ_to_column:
        x = typ_to_column[x.typ.__class__](x.expr, x.typ, x.parent)
    else:
        raise NotImplementedError("Can't convert column with type `" + str(x.typ.__class__) + "'.")

    if isinstance(x, ArrayColumn) and x._elt_type.__class__ in elt_typ_to_array_column:
        return elt_typ_to_array_column[x._elt_type.__class__](x.expr, x.typ, x.parent)

    elif isinstance(x, SetColumn) and x._elt_type.__class__ in elt_typ_to_set_column:
        return elt_typ_to_set_column[x._elt_type.__class__](x.expr, x.typ, x.parent)

    elif isinstance(x, AggregableColumn) and isinstance(x._elt_type, TArray):
        return elt_typ_to_agg_column[TArray][x._elt_type.element_type.__class__](x.expr, x.typ, x.parent)

    elif isinstance(x, AggregableColumn) and x._elt_type.__class__ in elt_typ_to_agg_column:
        return elt_typ_to_agg_column[x._elt_type.__class__](x.expr, x.typ, x.parent)

    else:
        return x


def is_numeric(x):
    numeric_types = set([float, long, int, Int32Column, Int64Column, Float32Column, Float64Column])
    return type(x) in numeric_types


def convert_numeric_typ(xs):
    typ_to_priority = {int: 0, Int32Column: 0, long: 1, Int64Column: 1, Float32Column: 2, float: 3, Float64Column: 3}
    ret_types = [Int32Column, Int64Column, Float32Column, Float64Column]
    priority = 0
    for x in xs:
        assert(is_numeric(x)), "Cannot implicitly convert non-numeric types. Found {}.".format(type(x))
        x_priority = typ_to_priority[x]
        if x_priority > priority:
            priority = x_priority
    return ret_types[priority]


def get_typ(x):
    if isinstance(x, Column):
        return x.typ
    elif isinstance(x, Variant):
        return TVariant()
    elif isinstance(x, Locus):
        return TLocus()
    elif isinstance(x, Interval):
        return TInterval()
    elif isinstance(x, AltAllele):
        return TAltAllele()
    elif isinstance(x, Genotype):
        return TGenotype()
    elif isinstance(x, Call):
        return TCall()
    elif isinstance(x, Struct):
        ids, typs = zip(*[(k, get_typ(v)) for k, v in x._attrs.iteritems()])
        return TStruct(ids, typs)
    elif isinstance(x, int):
        return TInt32()
    elif isinstance(x, long):
        return TInt64()
    elif isinstance(x, float):
        return TFloat64()
    elif isinstance(x, bool):
        return TBoolean()
    elif isinstance(x, str):
        return TString()
    elif isinstance(x, list):
        elements = set([get_typ(e) for e in x])
        if len(elements) == 0:
            raise NotImplementedError("Don't support empty lists.")
        elif len(elements) == 1:
            return TArray(list(elements)[0])
        elif all([is_numeric(e) for e in elements]):
            return TArray(convert_numeric_typ(elements))
        else:
            raise NotImplementedError("Don't support lists with multiple element types.")
    else:
        raise NotImplementedError(type(x))


class Column(object):
    def __init__(self, expr, typ=None, parent=None):
        self._expr = parent + "." + expr if parent else expr
        self._typ = typ
        self._parent = parent

    def __str__(self):
        return self._expr

    def __repr__(self):
        return self._expr

    @property
    def expr(self):
        return self._expr

    @property
    def typ(self):
        return self._typ

    @property
    def parent(self):
        return self._parent

    def _unary_op(self, name):
        @args_to_expr
        def _(column):
            return "{name}({col})".format(name=name, col=column)
        return convert_column(Column(_(self), self.typ))

    def _bin_op(self, name, other, ret_typ):
        @args_to_expr
        def _(column, other):
            return "({left} {op} {right})".format(left=column, op=name, right=other)
        return convert_column(Column(_(self, other), ret_typ))

    def _bin_op_reverse(self, name, other, ret_typ):
        @args_to_expr
        def _(column, other):
            return "({left} {op} {right})".format(left=other, op=name, right=column)
        return convert_column(Column(_(self, other), ret_typ))

    def _bin_op_comparison(self, name, other):
        @args_to_expr
        def _(column, other):
            return "({left} {op} {right})".format(left=column, op=name, right=other)
        return convert_column(Column(_(self, other), TBoolean()))

    def _field(self, name, ret_typ):
        @args_to_expr
        def _(column):
            return "{col}.{name}".format(col=column, name=name)
        return convert_column(Column(_(self), ret_typ))

    def _method(self, name, ret_typ, *args):
        @args_to_expr
        def _(column, *args):
            return "{col}.{name}({args})".format(col=column, name=name, args=", ".join(args))
        return convert_column(Column(_(self, *args), ret_typ))

    def _getter_key(self, ret_typ, args):
        @args_to_expr
        def _(column, args):
            return "{col}[{args}]".format(col=column, args=args)
        return convert_column(Column(_(self, args), ret_typ))

    def _getter_index(self, ret_typ, args):
        def _(column, args):
            return "{col}[{args}]".format(col=column, args=args)
        return convert_column(Column(_(self, args), ret_typ))

    def _bin_lambda_method(self, name, f, inp_typ, ret_typ_f, *args):
        @args_to_expr
        def _(column, result, *args):
            if args:
                return "{col}.{name}({new_id} => {result}, {args})".format(col=column, name=name, new_id=new_id,
                                                                           result=result, args=", ".join(args))
            else:
                return "{col}.{name}({new_id} => {result})".format(col=column, name=name, new_id=new_id,
                                                                   result=result)

        new_id = "x"
        lambda_result = f(convert_column(Column(new_id, inp_typ)))
        lambda_ret_typ = get_typ(lambda_result)
        result = convert_column(Column(to_expr(lambda_result), lambda_ret_typ))
        return convert_column(Column(_(self, result, *args), ret_typ_f(lambda_ret_typ)))

    def __eq__(self, other):
        return self._bin_op_comparison("==", other)

    def __ne__(self, other):
        return self._bin_op_comparison("!=", other)

    @staticmethod
    def null(typ):
        return convert_column(Column("NA: {}".format(typ), typ))


class CollectionColumn(Column):
    def __init__(self, expr, typ=None, parent=None):
        self._elt_type = typ.element_type
        super(CollectionColumn, self).__init__(expr, typ, parent)

    def exists(self, f):
        return self._bin_lambda_method("exists", f, self._elt_type, lambda t: TBoolean())

    def filter(self, f):
        return self._bin_lambda_method("filter", f, self._elt_type, lambda t: self.typ)

    def find(self, f):
        return self._bin_lambda_method("find", f, self._elt_type, lambda t: self._elt_type)

    def flat_map(self, f):
        return self._bin_lambda_method("flatMap", f, self._elt_type, lambda t: t)

    def forall(self, f):
        return self._bin_lambda_method("forall", f, self._elt_type, lambda t: TBoolean())

    def group_by(self, f):
        return self._bin_lambda_method("groupBy", f, self._elt_type, lambda t: TDict(t, self.typ))

    def head(self):
        return self._method("head", self._elt_type)

    def is_empty(self):
        return self._method("isEmpty", TBoolean())

    def map(self, f):
        return self._bin_lambda_method("map", f, self._elt_type, lambda t: self.typ.__class__(t))

    def size(self):
        return self._method("size", TInt32())

    def tail(self):
        return self._method("tail", self.typ)

    def to_array(self):
        return self._method("toArray", TArray(self._elt_type))

    def to_set(self):
        return self._method("toSet", TSet(self._elt_type))


class CollectionNumericColumn(CollectionColumn):
    def max(self):
        return self._method("max", self._elt_type)

    def mean(self):
        return self._method("mean", TFloat64())

    def median(self):
        return self._method("median", self._elt_type)

    def min(self):
        return self._method("min", self._elt_type)

    def product(self):
        return self._method("product", self._elt_type)

    def sort(self, ascending=True):
        return self._method("sort", self.typ, ascending)

    def sum(self):
        return self._method("sum", self._elt_type)


class ArrayColumn(CollectionColumn):

    def __getitem__(self, item):
        if isinstance(item, slice):
            args = "{start}:{end}".format(start=item.start if item.start else '',
                                          end=item.stop if item.stop else '')
            return self._getter_index(self.typ, args)
        elif isinstance(item, int) or isinstance(item, Int32Column):
            return self._getter_index(self._elt_type, item)
        else:
            raise NotImplementedError

    def append(self, x):
        return self._method("append", self.typ, x)

    def extend(self, a):
        return self._method("extend", self.typ, a)

    def length(self):
        return self._method("length", TInt32())

    def sort_by(self, f, ascending=True):
        return self._bin_lambda_method("sortBy", f, self._elt_type, lambda t: self.typ, ascending)


class ArrayBooleanColumn(ArrayColumn):
    def sort(self, ascending=True):
        return self._method("sort", self.typ, ascending)


class ArrayNumericColumn(ArrayColumn, CollectionNumericColumn):
    def _bin_op_ret_typ(self, other):
        if isinstance(self, ArrayNumericColumn) and isinstance(other, float):
            return TArray(TFloat64())
        
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, Float64Column):
            return TArray(TFloat64())
        elif isinstance(self, Float64Column) and isinstance(other, ArrayNumericColumn):
            return TArray(TFloat64())
        elif isinstance(self, ArrayFloat64Column) and isinstance(other, ArrayNumericColumn):
            return TArray(TFloat64())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, ArrayFloat64Column):
            return TArray(TFloat64())

        elif isinstance(self, Float32Column) and isinstance(other, ArrayNumericColumn):
            return TArray(TFloat32())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, Float32Column):
            return TArray(TFloat32())
        elif isinstance(self, ArrayFloat32Column) and isinstance(other, ArrayNumericColumn):
            return TArray(TFloat32())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, ArrayFloat32Column):
            return TArray(TFloat32())

        elif isinstance(self, ArrayInt64Column) and isinstance(other, int):
            return TArray(TInt64())
        elif isinstance(self, Int64Column) and isinstance(other, ArrayNumericColumn):
            return TArray(TInt64())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, Int64Column):
            return TArray(TInt64())
        elif isinstance(self, ArrayInt64Column) and isinstance(other, ArrayNumericColumn):
            return TArray(TInt64())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, ArrayInt64Column):
            return TArray(TInt64())

        elif isinstance(self, ArrayInt32Column) and isinstance(other, int):
            return TArray(TInt32())
        elif isinstance(self, Int32Column) and isinstance(other, ArrayNumericColumn):
            return TArray(TInt32())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, Int32Column):
            return TArray(TInt32())
        elif isinstance(self, ArrayInt32Column) and isinstance(other, ArrayNumericColumn):
            return TArray(TInt32())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, ArrayInt32Column):
            return TArray(TInt32())        

        else:
            raise NotImplementedError("Error in return type for numeric conversion.",
                                      "\nself =", type(self),
                                      "\nother =", type(other))

    def _bin_op_numeric(self, name, other):
        ret_typ = self._bin_op_ret_typ(other)
        return self._bin_op(name, other, ret_typ)

    def _bin_op_numeric_reverse(self, name, other):
        ret_typ = self._bin_op_ret_typ(other)
        return self._bin_op_reverse(name, other, ret_typ)

    def __add__(self, other):
        return self._bin_op_numeric("+", other)

    def __radd__(self, other):
        return self._bin_op_numeric_reverse("+", other)

    def __sub__(self, other):
        return self._bin_op_numeric("-", other)

    def __rsub__(self, other):
        return self._bin_op_numeric_reverse("-", other)

    def __mul__(self, other):
        return self._bin_op_numeric("*", other)

    def __rmul__(self, other):
        return self._bin_op_numeric_reverse("*", other)

    def __div__(self, other):
        return self._bin_op("/", other, TArray(TFloat64()))

    def __rdiv__(self, other):
        return self._bin_op_reverse("/", other, TArray(TFloat64()))


class ArrayFloat64Column(ArrayNumericColumn):
    pass


class ArrayFloat32Column(ArrayNumericColumn):
    pass


class ArrayInt32Column(ArrayNumericColumn):
    pass


class ArrayInt64Column(ArrayNumericColumn):
    pass


class ArrayStringColumn(ArrayColumn):
    def mkstring(self, delimiter):
        return self._method("mkString", TString(), delimiter)

    def sort(self, ascending=True):
        return self._method("sort", self.typ, ascending)


class ArrayStructColumn(ArrayColumn):
    pass


class ArrayArrayColumn(ArrayColumn):
    def flatten(self):
        return self._method("flatten", self._elt_type)


class SetColumn(CollectionColumn):
    def add(self, x):
        return self._method("add", self.typ, x)

    def contains(self, x):
        return self._method("contains", TBoolean(), x)

    def difference(self, s):
        return self._method("difference", self.typ, s)

    def intersection(self, s):
        return self._method("intersection", self.typ, s)

    def is_subset(self, s):
        return self._method("isSubset", TBoolean(), s)

    def union(self, s):
        return self._method("union", self.typ, s)


class SetFloat64Column(SetColumn, CollectionNumericColumn):
    pass


class SetFloat32Column(SetColumn, CollectionNumericColumn):
    pass


class SetInt32Column(SetColumn, CollectionNumericColumn):
    pass


class SetInt64Column(SetColumn, CollectionNumericColumn):
    pass


class SetStringColumn(SetColumn):
    def mkstring(self, delimiter):
        return self._method("mkString", TString(), delimiter)


class SetSetColumn(SetColumn):
    def flatten(self):
        return self._method("flatten", self._elt_type)


class DictColumn(Column):
    def __init__(self, expr, typ=None, parent=None):
        self._key_typ = typ.key_type
        self._value_typ = typ.value_type
        super(DictColumn, self).__init__(expr, typ, parent)

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise NotImplementedError
        else:
            return self._getter_key(self._value_typ, item)

    def contains(self, k):
        return self._method("contains", TBoolean(), k)

    def get(self, k):
        return self._method("get", self._value_typ, k)

    def is_empty(self):
        return self._method("isEmpty", TBoolean())

    def key_set(self):
        return self._method("keySet", TSet(self._key_typ))

    def keys(self):
        return self._method("keys", TArray(self._key_typ))

    def map_values(self, f):
        return self._bin_lambda_method("mapValues", f, self._value_typ, lambda t: TDict(self._key_typ, t))

    def size(self):
        return self._method("size", TInt32())

    def values(self):
        return self._method("values", TArray(self._value_typ))


class AggregableColumn(CollectionColumn):
    def collect(self):
        return self._method("collect", TArray(self._elt_type))

    def count(self):
        return self._method("count", TInt64())

    def counter(self):
        return self._method("counter", TDict(self._elt_type, TInt64()))

    def filter(self, f):
        return self._bin_lambda_method("filter", f, self._elt_type, lambda t: TAggregable(self._elt_type))

    def flat_map(self, f):
        return self._bin_lambda_method("flatMap", f, self._elt_type, lambda t: TAggregable(t.element_type))

    def fraction(self, f):
        return self._bin_lambda_method("fraction", f, self._elt_type, lambda t: TFloat64())

    def map(self, f):
        return self._bin_lambda_method("map", f, self._elt_type, lambda t: TAggregable(t))

    def take(self, n):
        return self._method("take", TArray(self._elt_type), n)

    def take_by(self, f, n):
        return self._bin_lambda_method("takeBy", f, self._elt_type, lambda t: TArray(self._elt_type), n)


class AggregableGenotypeColumn(AggregableColumn):
    def call_stats(self, f):
        ret_typ = TStruct(['AC', 'AF', 'AN', 'GC'], [TArray(TInt32()), TArray(TFloat64()), TInt32(), TArray(TInt32())])
        return self._bin_lambda_method("callStats", f, self._elt_type, lambda t: ret_typ)

    def hardy_weinberg(self):
        return self._method("hardyWeinberg", TStruct(['rExpectedHetFrequency', 'pHWE'], [TFloat64(), TFloat64()]))

    def inbreeding(self, af):
        ret_typ = TStruct(['Fstat', 'nTotal', 'nCalled', 'expectedHoms', 'observedHoms'], [TFloat64(), TInt64(), TInt64(), TFloat64(), TInt64()])
        return self._bin_lambda_method("inbreeding", af, self._elt_type, lambda t: ret_typ)


class AggregableNumericColumn(AggregableColumn):
    def max(self):
        return self._method("max", self._elt_type)

    def min(self):
        return self._method("min", self._elt_type)

    def sum(self):
        return self._method("sum", self._elt_type)


class AggregableFloat64Column(AggregableNumericColumn):
    def hist(self, start, end, bins):
        ret_typ = TStruct(['binEdges', 'binFrequencies', 'nLess', 'nGreater'], [TArray(TFloat64()), TArray(TInt64()), TInt64(), TInt64()])
        return self._method("hist", ret_typ, start, end, bins)

    def product(self):
        return self._method("product", TFloat64())

    def stats(self):
        return self._method("stats", TStruct(['mean', 'stdev', 'min', 'max', 'nNotMissing', 'sum'], [TFloat64(), TFloat64(), TFloat64(), TFloat64(), TInt64(), TFloat64()]))


class AggregableInt64Column(AggregableNumericColumn):
    def product(self):
        return self._method("product", TInt64())


class AggregableInt32Column(AggregableNumericColumn):
    pass


class AggregableFloat32Column(AggregableNumericColumn):
    pass


class AggregableArrayNumericColumn(AggregableColumn):
    def sum(self):
        return self._method("sum", self._elt_type)


class AggregableArrayFloat64Column(AggregableArrayNumericColumn):
    def info_score(self):
        return self._method("infoScore", TStruct(['score', 'nIncluded'], [TFloat64(), TInt32()]))


class AggregableArrayFloat32Column(AggregableArrayNumericColumn):
    pass


class AggregableArrayInt32Column(AggregableArrayNumericColumn):
    pass


class AggregableArrayInt64Column(AggregableArrayNumericColumn):
    pass


class StructColumn(Column):

    def __init__(self, expr, typ, parent=None):
        assert(isinstance(typ, TStruct)), "StructColumn requires `typ' to be TStruct."

        super(StructColumn, self).__init__(expr, typ, parent)

        self.fields = []

        for fd in self._typ.fields:
            column = typ_to_column[fd.typ.__class__](fd.name, fd.typ, self._expr)
            self.__setattr__(fd.name, column)
            self.fields.append(column)

    def __getitem__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            print("Could not find field `" + str(item) + "' in schema.")

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.fields)


class AtomicColumn(Column):
    def to_float64(self):
        return self._method("toFloat64", TFloat64())

    def to_float32(self):
        return self._method("toFloat32", TFloat32())

    def to_int64(self):
        return self._method("toInt64", TInt64())

    def to_int32(self):
        return self._method("toInt32", TInt32())

    def max(self, other):
        assert(isinstance(other, self.__class__))
        return self._method("max", self.typ, other)

    def min(self, other):
        assert(isinstance(other, self.__class__))
        return self._method("min", self.typ, other)


class BooleanColumn(AtomicColumn):
    def _bin_op_logical(self, name, other):
        assert(isinstance(other, BooleanColumn))
        return self._bin_op(name, other, TBoolean())

    def __and__(self, other):
        return self._bin_op_logical("&&", other)

    def __or__(self, other):
        return self._bin_op_logical("||", other)


class NumericColumn(AtomicColumn):
    def _bin_op_ret_typ(self, other):
        if isinstance(self, NumericColumn) and isinstance(other, float):
            return TFloat64()
        elif isinstance(self, Float64Column) and isinstance(other, NumericColumn):
            return TFloat64()
        elif isinstance(self, NumericColumn) and isinstance(other, Float64Column):
            return TFloat64()
        elif isinstance(self, Float32Column) and isinstance(other, NumericColumn):
            return TFloat32()
        elif isinstance(self, NumericColumn) and isinstance(other, Float32Column):
            return TFloat32()
        elif isinstance(self, Int64Column) and (isinstance(other, NumericColumn) or isinstance(other, int) or isinstance(other, long)):
            return TInt64()
        elif isinstance(self, NumericColumn) and isinstance(other, Int64Column):
            return TInt64()
        elif isinstance(self, Int32Column) and (isinstance(other, NumericColumn) or isinstance(other, int)):
            return TInt32()
        elif isinstance(self, NumericColumn) or isinstance(other, Int32Column):
            return TInt32()
        else:
            raise NotImplementedError("Error in return type for numeric conversion.",
                                      "\nself =", type(self),
                                      "\nother =", type(other))

    def _bin_op_numeric(self, name, other):
        ret_typ = self._bin_op_ret_typ(other)
        return self._bin_op(name, other, ret_typ)

    def _bin_op_numeric_reverse(self, name, other):
        ret_typ = self._bin_op_ret_typ(other)
        return self._bin_op_reverse(name, other, ret_typ)

    def __lt__(self, other):
        return self._bin_op_comparison("<", other)

    def __le__(self, other):
        return self._bin_op_comparison("<=", other)

    def __gt__(self, other):
        return self._bin_op_comparison(">", other)

    def __ge__(self, other):
        return self._bin_op_comparison(">=", other)

    def __neg__(self):
        return self._unary_op("-")

    def __pos__(self):
        return self._unary_op("+")

    def __add__(self, other):
        return self._bin_op_numeric("+", other)

    def __radd__(self, other):
        return self._bin_op_numeric_reverse("+", other)

    def __sub__(self, other):
        return self._bin_op_numeric("-", other)

    def __rsub__(self, other):
        return self._bin_op_numeric_reverse("-", other)

    def __mul__(self, other):
        return self._bin_op_numeric("*", other)

    def __rmul__(self, other):
        return self._bin_op_numeric_reverse("*", other)

    def __div__(self, other):
        return self._bin_op_numeric("/", other)

    def __rdiv__(self, other):
        return self._bin_op_numeric_reverse("/", other)

    def signum(self):
        return self._method("signum", TInt32())

    def abs(self):
        return self._method("abs", self.typ)


class Float64Column(NumericColumn):
    pass


class Float32Column(NumericColumn):
    pass


class Int32Column(NumericColumn):
    pass


class Int64Column(NumericColumn):
    pass


class StringColumn(AtomicColumn):

    def __getitem__(self, item):
        if isinstance(item, slice):
            args = "{start}:{end}".format(start=item.start if item.start else "",
                                          end=item.stop if item.stop else "")
            return self._getter_index(TString(), args)
        elif isinstance(item, int):
            return self._getter_index(TString(), item)
        else:
            raise NotImplementedError

    def __add__(self, other):
        assert(isinstance(other, StringColumn) or isinstance(other, str))
        return self._bin_op("+", other, TString())

    def __radd__(self, other):
        assert(isinstance(other, StringColumn) or isinstance(other, str))
        return self._bin_op_reverse("+", other, TString())

    def length(self):
        return self._method("length", TInt32())

    def replace(self, pattern1, pattern2):
        return self._method("replace", TString(), pattern1, pattern2)

    def split(self, delim, n=None):
        if n:
            return self._method("split", TArray(TString()), delim, n)
        else:
            return self._method("split", TArray(TString()), delim)


class CallColumn(Column):

    @staticmethod
    @args_to_expr
    def from_int32(i):
        expr = "Call({})".format(i)
        return CallColumn(expr, TCall())

    @property
    def gt(self):
        return self._field("gt", TInt32())

    def gtj(self):
        return self._method("gtj", TInt32())

    def gtk(self):
        return self._method("gtk", TInt32())

    def is_called(self):
        return self._method("isCalled", TBoolean())

    def is_called_nonref(self):
        return self._method("isCalledNonRef", TBoolean())

    def is_het(self):
        return self._method("isHet", TBoolean())

    def is_het_nonref(self):
        return self._method("isHetNonRef", TBoolean())

    def is_het_ref(self):
        return self._method("isHetRef", TBoolean())

    def is_hom_ref(self):
        return self._method("isHomRef", TBoolean())

    def is_hom_var(self):
        return self._method("isHomVar", TBoolean())

    def is_linear_scale(self):
        return self._method("isLinearScale", TBoolean())

    def is_not_called(self):
        return self._method("isNotCalled", TBoolean())

    def num_nonref_alleles(self):
        return self._method("nNonRefAlleles", TInt32())

    def one_hot_alleles(self, v):
        return self._method("oneHotAlleles", TArray(TInt32()), v)

    def one_hot_genotype(self, v):
        return self._method("oneHotGenotype", TArray(TInt32()), v)

    def to_genotype(self):
        return self._method("toGenotype", TGenotype())


class GenotypeColumn(Column):

    @staticmethod
    @args_to_expr
    def from_call(call):
        expr = "Genotype({})".format(call)
        return GenotypeColumn(expr, TGenotype())

    @staticmethod
    @args_to_expr
    def pl_genotype(v, call, ad, dp, gq, pl):
        expr = "Genotype({}, {}, {}, {}, {}, {})".format(v, call, ad, dp, gq, pl)
        return GenotypeColumn(expr, TGenotype())

    @staticmethod
    @args_to_expr
    def dosage_genotype(v, prob, call=None):
        if call:
            expr = "Genotype({}, {}, {})".format(v, call, prob)
        else:
            expr = "Genotype({}, {})".format(v, prob)
        return GenotypeColumn(expr, TGenotype())

    @property
    def ad(self):
        return self._field("ad", TArray(TInt32()))

    def call(self):
        return self._method("call", TCall())

    @property
    def dosage(self):
        return self._field("dosage", TFloat64())

    @property
    def dp(self):
        return self._field("dp", TInt32())

    @property
    def fake_ref(self):
        return self._field("fakeRef", TBoolean())

    def fraction_reads_ref(self):
        return self._method("fractionReadsRef", TFloat64())

    @property
    def gp(self):
        return self._field("gp", TArray(TFloat64()))

    @property
    def gq(self):
        return self._field("gq", TInt32())

    @property
    def gt(self):
        return self._field("gt", TInt32())

    def gtj(self):
        return self._method("gtj", TInt32())

    def gtk(self):
        return self._method("gtk", TInt32())

    def is_called(self):
        return self._method("isCalled", TBoolean())

    def is_called_nonref(self):
        return self._method("isCalledNonRef", TBoolean())

    def is_het(self):
        return self._method("isHet", TBoolean())

    def is_het_nonref(self):
        return self._method("isHetNonRef", TBoolean())

    def is_het_ref(self):
        return self._method("isHetRef", TBoolean())

    def is_hom_ref(self):
        return self._method("isHomRef", TBoolean())

    def is_hom_var(self):
        return self._method("isHomVar", TBoolean())

    def is_linear_scale(self):
        return self._method("isLinearScale", TBoolean())

    def is_not_called(self):
        return self._method("isNotCalled", TBoolean())

    def num_nonref_alleles(self):
        return self._method("nNonRefAlleles", TInt32())

    def od(self):
        return self._method("od", TInt32())

    def one_hot_alleles(self, v):
        return self._method("oneHotAlleles", TArray(TInt32()), v)

    def one_hot_genotype(self, v):
        return self._method("oneHotGenotype", TArray(TInt32()), v)

    def p_ab(self):
        return self._method("pAB", TFloat64())

    @property
    def pl(self):
        return self._field("pl", TArray(TInt32()))


class IntervalColumn(Column):

    @staticmethod
    @args_to_expr
    def from_args(contig, start, end):
        expr = "Interval({}, {}, {})".format(contig, start, end)
        return IntervalColumn(expr, TInterval())

    @staticmethod
    @args_to_expr
    def parse(s):
        expr = "Interval({})".format(s)
        return IntervalColumn(expr, TInterval())

    @staticmethod
    @args_to_expr
    def from_loci(l1, l2):
        expr = "Interval({}, {})".format(l1, l2)
        return IntervalColumn(expr, TInterval())

    def contains(self, locus):
        return self._method("contains", TBoolean(), locus)

    @property
    def end(self):
        return self._field("end", TLocus())

    @property
    def start(self):
        return self._field("start", TLocus())


class LocusColumn(Column):

    @staticmethod
    @args_to_expr
    def from_args(contig, pos):
        expr = "Locus({}, {})".format(contig, pos)
        return LocusColumn(expr, TLocus())

    @staticmethod
    @args_to_expr
    def parse(s):
        expr = "Locus({})".format(s)
        return LocusColumn(expr, TLocus())

    @property
    def contig(self):
        return self._field("contig", TString())

    @property
    def position(self):
        return self._field("position", TInt32())


class AltAlleleColumn(Column):

    @property
    def alt(self):
        return self._field("alt", TString())

    def category(self):
        return self._method("category", TString())

    def is_complex(self):
        return self._method("isComplex", TBoolean())

    def is_deletion(self):
        return self._method("isDeletion", TBoolean())

    def is_indel(self):
        return self._method("isIndel", TBoolean())

    def is_insertion(self):
        return self._method("isInsertion", TBoolean())

    def is_mnp(self):
        return self._method("isMNP", TBoolean())

    def is_snp(self):
        return self._method("isSNP", TBoolean())

    def is_star(self):
        return self._method("isStar", TBoolean())

    def is_transition(self):
        return self._method("isTransition", TBoolean())

    def is_transversion(self):
        return self._method("isTransversion", TBoolean())

    @property
    def ref(self):
        return self._field("ref", TString())


class VariantColumn(Column):

    @staticmethod
    @args_to_expr
    def from_args(contig, pos, ref, alts):
        expr = "Variant({}, {}, {}, {})".format(contig, pos, ref, alts)
        return VariantColumn(expr, TVariant())

    @staticmethod
    @args_to_expr
    def parse(s):
        expr = "Variant({})".format(s)
        return VariantColumn(expr, TVariant())

    def alt(self):
        return self._method("alt", TString())

    def alt_allele(self):
        return self._method("altAllele", TAltAllele())

    @property
    def alt_alleles(self):
        return self._field("altAlleles", TArray(TAltAllele()))

    @property
    def contig(self):
        return self._field("contig", TString())

    def in_x_nonpar(self):
        return self._method("inXNonPar", TBoolean())

    def in_x_par(self):
        return self._method("inXPar", TBoolean())

    def in_y_nonpar(self):
        return self._method("inYNonPar", TBoolean())

    def in_y_par(self):
        return self._method("inYPar", TBoolean())

    def is_autosomal(self):
        return self._method("isAutosomal", TBoolean())

    def is_biallelic(self):
        return self._method("isBiallelic", TBoolean())

    def locus(self):
        return self._method("locus", TLocus())

    def num_alleles(self):
        return self._method("nAlleles", TInt32())

    def num_alt_alleles(self):
        return self._method("nAltAlleles", TInt32())

    def num_genotypes(self):
        return self._method("nGenotypes", TInt32())

    @property
    def ref(self):
        return self._field("ref", TString())

    @property
    def start(self):
        return self._field("start", TInt32())


typ_to_column = {
    TBoolean: BooleanColumn,
    TInt32: Int32Column,
    TInt64: Int64Column,
    TFloat64: Float64Column,
    TFloat32: Float32Column,
    TLocus: LocusColumn,
    TInterval: IntervalColumn,
    TVariant: VariantColumn,
    TGenotype: GenotypeColumn,
    TCall: CallColumn,
    TAltAllele: AltAlleleColumn,
    TString: StringColumn,
    TDict: DictColumn,
    TArray: ArrayColumn,
    TSet: SetColumn,
    TStruct: StructColumn,
    TAggregable: AggregableColumn
}

elt_typ_to_array_column = {
    TBoolean: ArrayBooleanColumn,
    TInt32: ArrayInt32Column,
    TFloat64: ArrayFloat64Column,
    TInt64: ArrayInt64Column,
    TFloat32: ArrayFloat32Column,
    TString: ArrayStringColumn,
    TStruct: ArrayStructColumn,
    TArray: ArrayArrayColumn
}

elt_typ_to_set_column = {
    TInt32: SetInt32Column,
    TFloat64: SetFloat64Column,
    TFloat32: SetFloat32Column,
    TInt64: SetFloat32Column,
    TString: SetStringColumn,
    TSet: SetSetColumn
}

elt_typ_to_agg_column = {
    TInt32: AggregableInt32Column,
    TInt64: AggregableInt64Column,
    TFloat64: AggregableFloat64Column,
    TFloat32: AggregableFloat32Column,
    TGenotype: AggregableGenotypeColumn,
    TArray: {
        TFloat64: AggregableArrayFloat64Column,
        TFloat32: AggregableArrayFloat32Column,
        TInt32: AggregableArrayInt32Column,
        TInt64: AggregableArrayInt64Column
    }
}
