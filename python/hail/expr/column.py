from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.java import *
from hail.types import *
from hail.representation import *
from hail.representation.annotations import to_dict
from hail.typecheck import *


#### TO DO ####
# 1. :NA on Struct construction
# 2. if else # where()
# 3. Group by? add new columns???
# 4. let (decided to take small performance hit)
# 5. Struct constructor in exprs
# 6. Expr language tests
# 7. Regular expression string
# 8. Add pow, exp
# 9. check flat map recursive?
# 11. getitems and attr
# 12. query
# 14. sum / len built in python functions
# 15. array[struct] column
# 16. list numeric conversion


def to_expr(arg):
    if isinstance(arg, Column):
        return arg.expr
    elif isinstance(arg, str):
        return "\"" + arg + "\""
    elif isinstance(arg, bool):
        return "true" if arg else "false"
    elif isinstance(arg, list):
        return "[" + ", ".join([to_expr(a) for a in arg]) + "]"
    elif isinstance(arg, set):
        return "let a = [" + ", ".join([to_expr(a) for a in arg]) + "] in a.toSet()"
    elif isinstance(arg, Variant):
        return "Variant(\"" + str(arg) + "\")"
    elif isinstance(arg, Struct):
        return "{" + ", ".join([k + ":" + to_expr(v) for k, v in to_dict(arg).items()]) + "}" # FIXME: order of fields not preserved
    elif callable(arg):
        return arg
    else:
        return str(arg)


@decorator
def args_to_expr(func, *args):
    exprs = [to_expr(arg) for arg in args]
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
        return TCall
    elif isinstance(x, Struct):
        return TStruct
    elif isinstance(x, int):
        return TInt()
    elif isinstance(x, float):
        return TFloat()
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

    def _getter(self, ret_typ, args):
        @args_to_expr
        def _(column, args):
            return "{col}[{args}]".format(col=column, args=args)
        return convert_column(Column(_(self, args), ret_typ))

    def _bin_lambda_method(self, name, f, inp_typ, ret_typ_f, *args):
        @args_to_expr
        def _(column, result, *args):
            if args:
                return "{col}.{name}({new_id} => {result}, {args})".format(col=column, name=name, new_id=new_id, result=result, args=", ".join(args))
            else:
                return "{col}.{name}({new_id} => {result})".format(col=column, name=name, new_id=new_id, result=result)
        new_id = "x"
        lambda_result = f(convert_column(Column(new_id, inp_typ)))
        lambda_ret_typ = get_typ(lambda_result)
        print(lambda_ret_typ)
        result = convert_column(Column(to_expr(lambda_result), lambda_ret_typ))
        return convert_column(Column(_(self, result, *args), ret_typ_f(lambda_ret_typ)))

    def __eq__(self, other):
        return self._bin_op_comparison("==", other)

    def __ne__(self, other):
        return self._bin_op_comparison("!=", other)


class CollectionColumn(Column):
    def __init__(self, expr, typ=None, parent=None):
        self._elt_type = typ.element_type
        self._subcol = typ_to_column[self._elt_type.__class__]("x", typ.element_type)
        super(CollectionColumn, self).__init__(expr, typ, parent)

    def exists(self, f):
        return self._bin_lambda_method("exists", f, self._elt_type, lambda t: TBoolean())

    def filter(self, f):
        return self._bin_lambda_method("filter", f, self._elt_type, lambda t: self.typ)

    def find(self, f):
        return self._bin_lambda_method("find", f, self._elt_type, lambda t: self._elt_type)

    def flat_map(self, f):
        return self._bin_lambda_method("flatMap", f, self._elt_type, lambda t: self.typ.__class__(self._elt_type))

    def forall(self, f):
        return self._bin_lambda_method("forall", f, self._elt_type, lambda t: TBoolean())

    def group_by(self, f):
        return self._bin_lambda_method("groupBy", f, self._elt_type, lambda t: TDict(t, self.typ))

    def head(self):
        return self._method("head", self._elt_type)

    def is_empty(self):
        return self._method("isEmpty", TBoolean())

    def map(self, f):
        return self._bin_lambda_method("map", f, self._elt_type, lambda t: self.typ.__class__(self._elt_type))

    def size(self):
        return self._method("size", TInt())

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
        return self._method("mean", TDouble())

    def median(self):
        return self._method("median", self._elt_type)

    def min(self):
        return self._method("min", self._elt_type)

    def product(self):
        return self._method("product", self._elt_type)

    def sort(self, ascending=True):
        return self._method("sort", self.typ, ascending)

    def sum(self):
        return self._method("max", self._elt_type)


class ArrayColumn(CollectionColumn):

    def __getitem__(self, item):
        if isinstance(item, slice):
            args = "{start}:{end}".format(start=item.start if item.start else "",
                                          end=item.stop if item.stop else "")
            return self._getter(self.typ, args)
        elif isinstance(item, int) or isinstance(item, IntColumn):
            return self._getter(self._elt_type, item)
        else:
            raise NotImplementedError

    def append(self, x):
        return self._method("append", self.typ, x)

    def extend(self, a):
        return self._method("extend", self.typ, a)

    def length(self):
        return self._method("length", TInt())

    def sort_by(self, f, ascending=True):
        return self._bin_lambda_method("sortBy", f, self._elt_type, lambda t: self.typ, ascending)


class ArrayBooleanColumn(ArrayColumn):
    def sort(self, ascending=True):
        return self._method("sort", self.typ, ascending)


class ArrayNumericColumn(ArrayColumn, CollectionNumericColumn):
    def _bin_op_ret_typ(self, other):
        if isinstance(self, ArrayNumericColumn) and isinstance(other, float):
            return TArray(TDouble())
        
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, DoubleColumn):
            return TArray(TDouble())
        elif isinstance(self, DoubleColumn) and isinstance(other, ArrayNumericColumn):
            return TArray(TDouble())
        elif isinstance(self, ArrayDoubleColumn) and isinstance(other, ArrayNumericColumn):
            return TArray(TDouble())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, ArrayDoubleColumn):
            return TArray(TDouble())

        elif isinstance(self, FloatColumn) and isinstance(other, ArrayNumericColumn):
            return TArray(TFloat())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, FloatColumn):
            return TArray(TFloat())
        elif isinstance(self, ArrayFloatColumn) and isinstance(other, ArrayNumericColumn):
            return TArray(TFloat())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, ArrayFloatColumn):
            return TArray(TFloat())

        elif isinstance(self, ArrayLongColumn) and isinstance(other, int):
            return TArray(TLong())
        elif isinstance(self, LongColumn) and isinstance(other, ArrayNumericColumn):
            return TArray(TLong())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, LongColumn):
            return TArray(TLong())
        elif isinstance(self, ArrayLongColumn) and isinstance(other, ArrayNumericColumn):
            return TArray(TLong())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, ArrayLongColumn):
            return TArray(TLong())

        elif isinstance(self, ArrayIntColumn) and isinstance(other, int):
            return TArray(TInt())
        elif isinstance(self, IntColumn) and isinstance(other, ArrayNumericColumn):
            return TArray(TInt())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, IntColumn):
            return TArray(TInt())
        elif isinstance(self, ArrayIntColumn) and isinstance(other, ArrayNumericColumn):
            return TArray(TInt())
        elif isinstance(self, ArrayNumericColumn) and isinstance(other, ArrayIntColumn):
            return TArray(TInt())        

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
        return self._bin_op("/", other, TArray(TDouble()))

    def __rdiv__(self, other):
        return self._bin_op_reverse("/", other, TArray(TDouble()))


class ArrayDoubleColumn(ArrayNumericColumn):
    pass


class ArrayFloatColumn(ArrayNumericColumn):
    pass


class ArrayIntColumn(ArrayNumericColumn):
    pass


class ArrayLongColumn(ArrayNumericColumn):
    pass


class ArrayStringColumn(ArrayColumn):
    def mkstring(self, delimiter):
        return self._method("mkString", TString(), delimiter)

    def sort(self, ascending=True):
        return self._method("sort", self.typ, ascending)


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


class SetDoubleColumn(SetColumn, CollectionNumericColumn):
    pass


class SetFloatColumn(SetColumn, CollectionNumericColumn):
    pass


class SetIntColumn(SetColumn, CollectionNumericColumn):
    pass


class SetLongColumn(SetColumn, CollectionNumericColumn):
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
            return self._getter(self._value_typ, item)

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
        return self._method("size", TInt())

    def values(self):
        return self._method("values", TArray(self._value_typ))


class AggregableColumn(CollectionColumn):
    def collect(self):
        return self._method("collect", TArray(self._elt_type))

    def count(self):
        return self._method("count", TLong())

    def counter(self):
        return self._method("counter", TDict(self._elt_type, TLong()))

    def filter(self, f):
        return self._bin_lambda_method("filter", f, self._elt_type, lambda t: TAggregable(self._elt_type))

    def flatMap(self, f):
        return self._bin_lambda_method("flatMap", f, self._elt_type, lambda t: TAggregable(t))

    def fraction(self, f):
        return self._bin_lambda_method("fraction", f, self._elt_type, lambda t: TDouble())

    def map(self, f):
        return self._bin_lambda_method("map", f, self._elt_type, lambda t: TAggregable(t))

    def take(self, n):
        return self._method("take", TArray(self._elt_type), n)

    def take_by(self, f, n):
        return self._bin_lambda_method("takeBy", f, self._elt_type, lambda t: TArray(self._elt_type), n)


class AggregableGenotypeColumn(AggregableColumn):
    def call_stats(self, f):
        ret_typ = TStruct(['AC', 'AF', 'AN', 'GC'], [TArray(TInt()), TArray(TDouble()), TInt(), TArray(TInt())])
        return self._bin_lambda_method("callStats", f, self._elt_type, lambda t: ret_typ)

    def hardy_weinberg(self):
        return self._method("hardyWeinberg", TStruct(['rExpectedHetFrequency', 'pHWE'], [TDouble(), TDouble()]))

    def inbreeding(self, af):
        ret_typ = TStruct(['Fstat', 'nTotal', 'nCalled', 'expectedHoms', 'observedHoms'], [TDouble(), TLong(), TLong(), TDouble(), TLong()])
        return self._bin_lambda_method("inbreeding", af, self._elt_type, lambda t: ret_typ)

    def info_score(self):
        return self._method("infoScore", TStruct(['score', 'nIncluded'], [TDouble(), TInt()]))


class AggregableNumericColumn(AggregableColumn):
    def max(self):
        return self._method("max", self._elt_type)

    def min(self):
        return self._method("min", self._elt_type)

    def sum(self):
        return self._method("sum", self._elt_type)


class AggregableDoubleColumn(AggregableNumericColumn):
    def hist(self, start, end, bins):
        ret_typ = TStruct(['binEdges', 'binFrequencies', 'nLess', 'nGreater'], [TArray(TDouble()), TArray(TLong()), TLong(), TLong()])
        return self._method("hist", ret_typ, start, end, bins)

    def product(self):
        return self._method("product", TDouble())

    def stats(self):
        return self._method("stats", TStruct(['mean', 'stdev', 'min', 'max', 'nNotMissing', 'sum'], [TDouble(), TDouble(), TDouble(), TDouble(), TLong(), TDouble()]))


class AggregableLongColumn(AggregableNumericColumn):
    def product(self):
        return self._method("product", TLong())


class AggregableIntColumn(AggregableNumericColumn):
    pass


class AggregableFloatColumn(AggregableNumericColumn):
    pass

class AggregableArrayNumericColumn(AggregableColumn):
    def sum(self):
        return self._method("sum", self._elt_type)


class AggregableArrayDoubleColumn(AggregableArrayNumericColumn):
    pass


class AggregableArrayFloatColumn(AggregableArrayNumericColumn):
    pass


class AggregableArrayIntColumn(AggregableArrayNumericColumn):
    pass


class AggregableArrayLongColumn(AggregableArrayNumericColumn):
    pass


class StructColumn(Column):

    def __init__(self, expr, typ, parent=None):
        # assert(isinstance(typ, TStruct), "StructColumn requires `typ' to be TStruct.")

        super(StructColumn, self).__init__(expr, typ, parent)

        for fd in self._typ.fields:
            column = typ_to_column[fd.typ.__class__]
            self.__setattr__(fd.name, column(fd.name, fd.typ, self._expr))

    def __getitem__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            print("Could not find field `" + str(item) + "' in schema.")

    def __setattr__(self, key, value):
        self.__dict__[key] = value


class AtomicColumn(Column):
    def to_double(self):
        return self._method("toDouble", TDouble())

    def to_float(self):
        return self._method("toFloat", TFloat())

    def to_long(self):
        return self._method("toLong", TLong())

    def to_int(self):
        return self._method("toInt", TInt())

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
            return TDouble()
        elif isinstance(self, DoubleColumn) and isinstance(other, NumericColumn):
            return TDouble()
        elif isinstance(self, NumericColumn) and isinstance(other, DoubleColumn):
            return TDouble()
        elif isinstance(self, FloatColumn) and isinstance(other, NumericColumn):
            return TFloat()
        elif isinstance(self, NumericColumn) and isinstance(other, FloatColumn):
            return TFloat()
        elif isinstance(self, LongColumn) and (isinstance(other, NumericColumn) or isinstance(other, int)):
            return TLong()
        elif isinstance(self, NumericColumn) and isinstance(other, LongColumn):
            return TLong()
        elif isinstance(self, IntColumn) and (isinstance(other, NumericColumn) or isinstance(other, int)):
            return TInt()
        elif isinstance(self, NumericColumn) or isinstance(other, IntColumn):
            return TInt()
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
        return self._method("signum", TInt())

    def abs(self):
        return self._method("abs", self.typ)


class DoubleColumn(NumericColumn):
    pass


class FloatColumn(NumericColumn):
    pass


class IntColumn(NumericColumn):
    pass


class LongColumn(NumericColumn):
    pass


class StringColumn(AtomicColumn):

    def __getitem__(self, item):
        if isinstance(item, slice):
            args = "{start}:{end}".format(start=item.start if item.start else "",
                                          end=item.stop if item.stop else "")
            return self._getter(TString(), args)
        elif isinstance(item, int):
            return self._getter(TString(), item)
        else:
            raise NotImplementedError

    def __add__(self, other):
        assert(isinstance(other, StringColumn) or isinstance(other, str))
        return self._bin_op("+", other, TString())

    def __radd__(self, other):
        assert(isinstance(other, StringColumn) or isinstance(other, str))
        return self._bin_op_reverse("+", other, TString())

    def length(self):
        return self._method("length", TInt())

    def replace(self, pattern1, pattern2):
        return self._method("replace", TString(), pattern1, pattern2)

    def split(self, delim, n=None):
        if n:
            return self._method("split", TArray(TString()), delim, n)
        else:
            return self._method("split", TArray(TString()), delim)


class CallColumn(Column):

    @property
    def gt(self):
        return self._field("gt", TInt())

    def gtj(self):
        return self._method("gtj", TInt())

    def gtk(self):
        return self._method("gtk", TInt())

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
        return self._method("nNonRefAlleles", TInt())

    def one_hot_alleles(self, v):
        return self._method("oneHotAlleles", TArray(TInt()), v)

    def one_hot_genotype(self, v):
        return self._method("oneHotGenotype", TArray(TInt()), v)

    def to_genotype(self):
        return self._method("toGenotype", TGenotype())


class GenotypeColumn(Column):

    @property
    def ad(self):
        return self._field("ad", TArray(TInt()))

    def call(self):
        return self._method("call", TCall())

    @property
    def dosage(self):
        return self._field("dosage", TDouble())

    @property
    def dp(self):
        return self._field("dp", TInt())

    @property
    def fake_ref(self):
        return self._field("fakeRef", TBoolean())

    def fraction_reads_ref(self):
        return self._method("fractionReadsRef", TDouble())

    @property
    def gp(self):
        return self._field("gp", TArray(TDouble()))

    @property
    def gq(self):
        return self._field("gq", TInt())

    @property
    def gt(self):
        return self._field("gt", TInt())

    def gtj(self):
        return self._method("gtj", TInt())

    def gtk(self):
        return self._method("gtk", TInt())

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
        return self._method("nNonRefAlleles", TInt())

    def od(self):
        return self._method("od", TInt())

    def one_hot_alleles(self, v):
        return self._method("oneHotAlleles", TArray(TInt()), v)

    def one_hot_genotype(self, v):
        return self._method("oneHotGenotype", TArray(TInt()), v)

    def p_ab(self):
        return self._method("pAB", TDouble())

    @property
    def pl(self):
        return self._field("pl", TArray(TInt()))


class IntervalColumn(Column):

    def contains(self, locus):
        return self._method("contains", TBoolean(), locus)

    @property
    def end(self):
        return self._field("end", TLocus())

    @property
    def start(self):
        return self._field("start", TLocus())


class LocusColumn(Column):

    @property
    def contig(self):
        return self._field("contig", TString())

    @property
    def position(self):
        return self._field("position", TInt())


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
        return self._method("nAlleles", TInt())

    def num_alt_alleles(self):
        return self._method("nAltAlleles", TInt())

    def num_genotypes(self):
        return self._method("nGenotypes", TInt())

    @property
    def ref(self):
        return self._field("ref", TString())

    @property
    def start(self):
        return self._field("start", TInt())


typ_to_column = {
    TBoolean: BooleanColumn,
    TInt: IntColumn,
    TLong: LongColumn,
    TDouble: DoubleColumn,
    TFloat: FloatColumn,
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
    TInt: ArrayIntColumn,
    TDouble: ArrayDoubleColumn,
    TLong: ArrayLongColumn,
    TFloat: ArrayFloatColumn,
    TString: ArrayStringColumn,
    TArray: ArrayArrayColumn
}

elt_typ_to_set_column = {
    TInt: SetIntColumn,
    TDouble: SetDoubleColumn,
    TFloat: SetFloatColumn,
    TLong: SetFloatColumn,
    TString: SetStringColumn,
    TSet: SetSetColumn
}

elt_typ_to_agg_column = {
    TInt: AggregableIntColumn,
    TLong: AggregableLongColumn,
    TDouble: AggregableDoubleColumn,
    TFloat: AggregableFloatColumn,
    TGenotype: AggregableGenotypeColumn,
    TArray: {
        TDouble: AggregableArrayDoubleColumn,
        TFloat: AggregableArrayFloatColumn,
        TInt: AggregableArrayIntColumn,
        TLong: AggregableArrayLongColumn
    }
}
