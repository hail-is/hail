from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.expr.ast import *
from hail.expr.types import *
from hail.utils.java import *
import hail


def to_expr(e):
    if isinstance(e, Expression):
        return e
    elif isinstance(e, str) or isinstance(e, unicode):
        return Expression(Literal('"{}"'.format(e)), TString())
    elif isinstance(e, bool):
        return Expression(Literal("true" if e else "false"), TBoolean())
    elif isinstance(e, int):
        return Expression(Literal(str(e)), TInt32())
    elif isinstance(e, long):
        return Expression(ClassMethod('toInt64', Literal('"{}"'.format(e))), TInt64())
    elif isinstance(e, float):
        return Expression(Literal(str(e)), TFloat64())
    elif isinstance(e, hail.genetics.Variant):
        return Expression(ApplyMethod('Variant', Literal('"{}"'.format(str(e)))), TVariant(e.reference_genome))
    elif isinstance(e, hail.genetics.Locus):
        return Expression(ApplyMethod('Locus', Literal('"{}"'.format(str(e)))), TLocus(e.reference_genome))
    elif isinstance(e, hail.genetics.Interval):
        return Expression(ApplyMethod('Interval', Literal('"{}"'.format(str(e)))), TInterval(e.reference_genome))
    elif isinstance(e, Struct):
        attrs = e._attrs.items()
        cols = [to_expr(x) for _, x in attrs]
        names = [k for k, _ in attrs]
        indices, aggregations, joins = unify_all(*cols)
        t = TStruct(names, [col._type for col in cols])
        return Expression(StructDeclaration(names, [c._ast for c in cols]),
                          t, indices, aggregations, joins)
    elif isinstance(e, list):
        cols = [to_expr(x) for x in e]
        types = list({col._type for col in cols})
        if len(cols) == 0:
            raise ValueError("Don't support empty lists.")
        elif len(types) == 1:
            t = TArray(types[0])
        elif all([is_numeric(t) for t in types]):
            t = TArray(convert_numeric_typ(*types))
        else:
            raise ValueError("Don't support lists with multiple element types.")
        indices, aggregations, joins = unify_all(*cols)
        return Expression(ArrayDeclaration([col._ast for col in cols]),
                          t, indices, aggregations, joins)
    elif isinstance(e, set):
        cols = [to_expr(x) for x in e]
        types = list({col._type for col in cols})
        if len(cols) == 0:
            raise ValueError("Don't support empty sets.")
        elif len(types) == 1:
            t = TArray(types[0])
        elif all([is_numeric(t) for t in types]):
            t = TArray(convert_numeric_typ(*types))
        else:
            raise ValueError("Don't support sets with multiple element types.")
        indices, aggregations, joins = unify_all(*cols)
        return Expression(ClassMethod('toSet', ArrayDeclaration([col._ast for col in cols])), t, indices, aggregations,
                          joins)
    elif isinstance(e, dict):
        key_cols = []
        value_cols = []
        keys = []
        values = []
        for k, v in e.items():
            key_cols.append(to_expr(k))
            keys.append(k)
            value_cols.append(to_expr(v))
            values.append(v)
        key_types = list({col._type for col in key_cols})
        value_types = list({col._type for col in value_cols})

        if len(key_types) == 0:
            raise ValueError("Don't support empty dictionaries.")
        elif len(key_types) == 1:
            key_type = key_types[0]
        elif all([is_numeric(t) for t in key_types]):
            key_type = convert_numeric_typ(*key_types)
        else:
            raise ValueError("Don't support dictionaries with multiple key types.")

        if len(value_types) == 1:
            value_type = value_types[0]
        elif all([is_numeric(t) for t in value_types]):
            value_type = convert_numeric_typ(*value_types)
        else:
            raise ValueError("Don't support dictionaries with multiple value types.")

        kc = to_expr(keys)
        vc = to_expr(values)

        indices, aggregations, joins = unify_all(kc, vc)

        assert key_type == kc._type.element_type
        assert value_type == vc._type.element_type

        ast = ApplyMethod('Dict',
                          ArrayDeclaration([k._ast for k in key_cols]),
                          ArrayDeclaration([v._ast for v in value_cols]))
        return Expression(ast, TDict(key_type, value_type), indices, aggregations, joins)
    else:
        raise ValueError("Cannot implicitly capture value `{}' with type `{}'.".format(e, e.__class__))


@decorator
def args_to_expr(func, *args):
    return func(*(convert_expr(to_expr(a)) for a in args))


def unify_all(*exprs):
    assert len(exprs) > 0
    new_indices = Indices.unify(*[e._indices for e in exprs])
    agg = list(exprs[0]._aggregations)
    joins = list(exprs[0]._joins)
    for e in exprs[1:]:
        agg.extend(e._aggregations)
        joins.extend(e._joins)
    return new_indices, tuple(agg), tuple(joins)


def convert_expr(x):
    if isinstance(x, Expression) and x._type.__class__ in typ_to_expr:
        x = typ_to_expr[x._type.__class__](x._ast, x._type, x._indices, x._aggregations, x._joins)

        if isinstance(x, ArrayExpression) and x._type.element_type.__class__ in elt_typ_to_array_expr:
            return elt_typ_to_array_expr[x._type.element_type.__class__](
                x._ast, x._type, x._indices, x._aggregations, x._joins)
        elif isinstance(x, SetExpression) and x._type.element_type.__class__ in elt_typ_to_set_expr:
            return elt_typ_to_set_expr[x._type.element_type.__class__](
                x._ast, x._type, x._indices, x._aggregations, x._joins)
        else:
            return x
    else:
        raise NotImplementedError("Can't convert column with type `" + str(x._type.__class__) + "'.")


__numeric_types = [TInt32, TInt64, TFloat32, TFloat64]


@typecheck(t=Type)
def is_numeric(t):
    return t.__class__ in __numeric_types


@typecheck(types=tupleof(Type))
def convert_numeric_typ(*types):
    priority_map = {t: p for t, p in zip(__numeric_types, range(len(__numeric_types)))}
    priority = 0
    for t in types:
        assert (is_numeric(t)), t
        t_priority = priority_map[t.__class__]
        if t_priority > priority:
            priority = t_priority
    return __numeric_types[priority]()


class Indices(object):
    @typecheck_method(source=anytype, axes=setof(strlike))
    def __init__(self, source=None, axes=set()):
        self.source = source
        self.axes = axes

    def __eq__(self, other):
        return isinstance(other, Indices) and self.source is other.source and self.axes == other.axes

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def unify(*indices):
        axes = set()
        src = None
        for ind in indices:
            if src is None:
                src = ind.source
            else:
                if ind.source is not None and ind.source is not src:
                    raise ExpressionException('Cannot unify_all operations between {} and {}'.format(
                        repr(src), repr(ind.source)))

            intersection = axes.intersection(ind.axes)
            left = axes - intersection
            right = ind.axes - intersection
            if right:
                for i in left:
                    info('broadcasting index {} along axes [{}]'.format(i, ', '.join(right)))
            if left:
                for i in right:
                    info('broadcasting index {} along axes [{}]'.format(i, ', '.join(left)))

            axes = axes.union(ind.axes)

        return Indices(src, axes)


class Aggregation(object):
    def __init__(self, indices):
        self.indices = indices


class Join(object):
    def __init__(self, join_function, temp_vars):
        self.join_function = join_function
        # self.right = right
        # self.index_exprs = index_exprs
        self.temp_vars = temp_vars


class Expression(object):
    @typecheck_method(ast=AST, type=Type, indices=Indices, aggregations=tupleof(Aggregation), joins=tupleof(Join))
    def __init__(self, ast, type, indices=Indices(), aggregations=(), joins=()):
        self._ast = ast
        self._type = type
        self._indices = indices
        self._aggregations = aggregations
        self._joins = joins

        self._init()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        s = "{super_repr}\n  Type: {type}".format(
            super_repr=super(Expression, self).__repr__(),
            type=str(self._type),
        )

        indices = self._indices
        if len(indices.axes) == 0:
            s += '\n  Index{agg}: None'.format(agg=' (aggregated)' if self._aggregations else '')
        else:
            s += '\n  {ind}{agg}:\n    {index_lines}'.format(ind=plural('Index', len(indices.axes), 'Indices'),
                                                             agg=' (aggregated)' if self._aggregations else '',
                                                             index_lines='\n    '.join('{} of {}'.format(
                                                                 axis, indices.source) for axis in indices.axes))
        if self._joins:
            s += '\n  Dependent on {} {}'.format(len(self._joins),
                                                 plural('broadcast/join', len(self._joins), 'broadcasts/joins'))
        return s

    def _init(self):
        pass

    def __nonzero__(self):
        raise NotImplementedError("The truth value of an expression is undefined\n  Hint: instead of if/else, use 'f.cond'")

    def _unary_op(self, name):
        return convert_expr(Expression(UnaryOperation(self._ast, name),
                                       self._type, self._indices, self._aggregations, self._joins))

    def _bin_op(self, name, other, ret_typ):
        other = to_expr(other)
        indices, aggregations, joins = unify_all(self, other)
        return convert_expr(
            Expression(BinaryOperation(self._ast, other._ast, name), ret_typ, indices, aggregations, joins))

    def _bin_op_reverse(self, name, other, ret_typ):
        other = to_expr(other)
        indices, aggregations, joins = unify_all(self, other)
        return convert_expr(
            Expression(BinaryOperation(other._ast, self._ast, name), ret_typ, indices, aggregations, joins))

    def _field(self, name, ret_typ):
        return convert_expr(
            Expression(Select(self._ast, name), ret_typ, self._indices, self._aggregations, self._joins))

    def _method(self, name, ret_typ, *args):
        args = (to_expr(arg) for arg in args)
        indices, aggregations, joins = unify_all(self, *args)
        return convert_expr(Expression(ClassMethod(name, self._ast, *(a._ast for a in args)),
                                       ret_typ, indices, aggregations, joins))

    def _index(self, ret_typ, key):
        key = to_expr(key)
        indices, aggregations, joins = unify_all(self, key)
        return convert_expr(Expression(Index(self._ast, key._ast),
                                       ret_typ, indices, aggregations, joins))

    def _slice(self, ret_typ, start=None, stop=None, step=None):
        if start is not None:
            start = to_expr(start)
            start_ast = start._ast
        else:
            start_ast = None
        if stop is not None:
            stop = to_expr(stop)
            stop_ast = stop._ast
        else:
            stop_ast = None
        if step is not None:
            raise NotImplementedError('Variable slice step size is not currently supported')

        non_null = [x for x in [start, stop] if x is not None]
        indices, aggregations, joins = unify_all(self, *non_null)
        return convert_expr(Expression(Index(self._ast, Slice(start_ast, stop_ast)),
                                       ret_typ, indices, aggregations, joins))

    def _bin_lambda_method(self, name, f, inp_typ, ret_typ_f, *args):
        args = (to_expr(arg) for arg in args)
        new_id = Env._get_uid()
        lambda_result = to_expr(
            f(convert_expr(Expression(Reference(new_id), inp_typ, self._indices, self._aggregations, self._joins))))
        indices, aggregations, joins = unify_all(self, lambda_result)
        ast = LambdaClassMethod(name, new_id, self._ast, lambda_result._ast, *(a._ast for a in args))
        return convert_expr(Expression(ast, ret_typ_f(lambda_result._type), indices, aggregations, joins))

    def __eq__(self, other):
        return self._bin_op("==", other, TBoolean())

    def __ne__(self, other):
        return self._bin_op("!=", other, TBoolean())

    @staticmethod
    def null(typ):
        return convert_expr(Expression("NA: {}".format(typ), typ))


class CollectionExpression(Expression):
    def contains(self, item):
        item = to_expr(item)
        return self.exists(lambda x: x == item)

    def exists(self, f):
        return self._bin_lambda_method("exists", f, self._type.element_type, lambda t: TBoolean())

    def filter(self, f):
        return self._bin_lambda_method("filter", f, self._type.element_type, lambda t: self._type)

    def find(self, f):
        return self._bin_lambda_method("find", f, self._type.element_type, lambda t: self._type.element_type)

    def flatmap(self, f):
        return self._bin_lambda_method("flatMap", f, self._type.element_type, lambda t: t)

    def forall(self, f):
        return self._bin_lambda_method("forall", f, self._type.element_type, lambda t: TBoolean())

    def group_by(self, f):
        return self._bin_lambda_method("groupBy", f, self._type.element_type, lambda t: TDict(t, self._type))

    def head(self):
        return self._method("head", self._type.element_type)

    def is_empty(self):
        return self._method("isEmpty", TBoolean())

    def map(self, f):
        return self._bin_lambda_method("map", f, self._type.element_type, lambda t: self._type.__class__(t))

    def size(self):
        return self._method("size", TInt32())

    def tail(self):
        return self._method("tail", self._type)

    def to_array(self):
        return self._method("toArray", TArray(self._type.element_type))

    def to_set(self):
        return self._method("toSet", TSet(self._type.element_type))


class CollectionNumericExpression(CollectionExpression):
    def max(self):
        return self._method("max", self._type.element_type)

    def mean(self):
        return self._method("mean", TFloat64())

    def median(self):
        return self._method("median", self._type.element_type)

    def min(self):
        return self._method("min", self._type.element_type)

    def product(self):
        return self._method("product", self._type.element_type)

    def sort(self, ascending=True):
        return self._method("sort", self._type, ascending)

    def sum(self):
        return self._method("sum", self._type.element_type)


class ArrayExpression(CollectionExpression):
    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._slice(self._type, item.start, item.stop, item.step)
        elif isinstance(item, int) or isinstance(item, Int32Expression):
            return self._index(self._type.element_type, item)
        else:
            raise NotImplementedError

    def append(self, x):
        return self._method("append", self._type, x)

    def extend(self, a):
        return self._method("extend", self._type, a)

    def length(self):
        return self._method("length", TInt32())

    def sort_by(self, f, ascending=True):
        return self._bin_lambda_method("sortBy", f, self._type.element_type, lambda t: self._type, ascending)


class ArrayBooleanExpression(ArrayExpression):
    def sort(self, ascending=True):
        return self._method("sort", self._type, ascending)


class ArrayNumericExpression(ArrayExpression, CollectionNumericExpression):
    def _bin_op_ret_typ(self, other):
        if isinstance(self, ArrayNumericExpression) and isinstance(other, float):
            return TArray(TFloat64())

        elif isinstance(self, ArrayNumericExpression) and isinstance(other, Float64Expression):
            return TArray(TFloat64())
        elif isinstance(self, Float64Expression) and isinstance(other, ArrayNumericExpression):
            return TArray(TFloat64())
        elif isinstance(self, ArrayFloat64Expression) and isinstance(other, ArrayNumericExpression):
            return TArray(TFloat64())
        elif isinstance(self, ArrayNumericExpression) and isinstance(other, ArrayFloat64Expression):
            return TArray(TFloat64())

        elif isinstance(self, Float32Expression) and isinstance(other, ArrayNumericExpression):
            return TArray(TFloat32())
        elif isinstance(self, ArrayNumericExpression) and isinstance(other, Float32Expression):
            return TArray(TFloat32())
        elif isinstance(self, ArrayFloat32Expression) and isinstance(other, ArrayNumericExpression):
            return TArray(TFloat32())
        elif isinstance(self, ArrayNumericExpression) and isinstance(other, ArrayFloat32Expression):
            return TArray(TFloat32())

        elif isinstance(self, ArrayInt64Expression) and isinstance(other, int):
            return TArray(TInt64())
        elif isinstance(self, Int64Expression) and isinstance(other, ArrayNumericExpression):
            return TArray(TInt64())
        elif isinstance(self, ArrayNumericExpression) and isinstance(other, Int64Expression):
            return TArray(TInt64())
        elif isinstance(self, ArrayInt64Expression) and isinstance(other, ArrayNumericExpression):
            return TArray(TInt64())
        elif isinstance(self, ArrayNumericExpression) and isinstance(other, ArrayInt64Expression):
            return TArray(TInt64())

        elif isinstance(self, ArrayInt32Expression) and isinstance(other, int):
            return TArray(TInt32())
        elif isinstance(self, Int32Expression) and isinstance(other, ArrayNumericExpression):
            return TArray(TInt32())
        elif isinstance(self, ArrayNumericExpression) and isinstance(other, Int32Expression):
            return TArray(TInt32())
        elif isinstance(self, ArrayInt32Expression) and isinstance(other, ArrayNumericExpression):
            return TArray(TInt32())
        elif isinstance(self, ArrayNumericExpression) and isinstance(other, ArrayInt32Expression):
            return TArray(TInt32())

        else:
            raise NotImplementedError('''Error in return type for numeric conversion.
                self = {}
                other = {}'''.format(type(self), type(other)))

    def _bin_op_numeric(self, name, other):
        other = convert_expr(to_expr(other))
        ret_typ = self._bin_op_ret_typ(other)
        return self._bin_op(name, other, ret_typ)

    def _bin_op_numeric_reverse(self, name, other):
        other = convert_expr(to_expr(other))
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


class ArrayFloat64Expression(ArrayNumericExpression):
    pass


class ArrayFloat32Expression(ArrayNumericExpression):
    pass


class ArrayInt32Expression(ArrayNumericExpression):
    pass


class ArrayInt64Expression(ArrayNumericExpression):
    pass


class ArrayStringExpression(ArrayExpression):
    def mkstring(self, delimiter):
        return self._method("mkString", TString(), delimiter)

    def sort(self, ascending=True):
        return self._method("sort", self._type, ascending)


class ArrayStructExpression(ArrayExpression):
    pass


class ArrayArrayExpression(ArrayExpression):
    def flatten(self):
        return self._method("flatten", self._type.element_type)


class SetExpression(CollectionExpression):
    def add(self, x):
        return self._method("add", self._type, x)

    def contains(self, x):
        return self._method("contains", TBoolean(), x)

    def difference(self, s):
        return self._method("difference", self._type, s)

    def intersection(self, s):
        return self._method("intersection", self._type, s)

    def is_subset(self, s):
        return self._method("isSubset", TBoolean(), s)

    def union(self, s):
        return self._method("union", self._type, s)


class SetFloat64Expression(SetExpression, CollectionNumericExpression):
    pass


class SetFloat32Expression(SetExpression, CollectionNumericExpression):
    pass


class SetInt32Expression(SetExpression, CollectionNumericExpression):
    pass


class SetInt64Expression(SetExpression, CollectionNumericExpression):
    pass


class SetStringExpression(SetExpression):
    def mkstring(self, delimiter):
        return self._method("mkString", TString(), delimiter)


class SetSetExpression(SetExpression):
    def flatten(self):
        return self._method("flatten", self._type.element_type)


class DictExpression(Expression):
    def _init(self):
        self._key_typ = self._type.key_type
        self._value_typ = self._type.value_type

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise NotImplementedError
        else:
            return self._index(self._value_typ, item)

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


class Aggregable(object):
    def __init__(self, ast, type, indices, aggregations, joins):
        self._ast = ast
        self._type = type
        self._indices = indices
        self._aggregations = aggregations
        self._joins = joins

    def __nonzero__(self):
        raise NotImplementedError('Truth value of an aggregable collection is undefined')

    def __eq__(self, other):
        raise NotImplementedError('Comparison of aggregable collections is undefined')

    def __ne__(self, other):
        raise NotImplementedError('Comparison of aggregable collections is undefined')


class StructExpression(Expression):
    def _init(self):
        self._fields = {}

        for fd in self._type.fields:
            expr = typ_to_expr[fd.typ.__class__](Select(self._ast, fd.name), fd.typ,
                                                 self._indices, self._aggregations, self._joins)
            self._set_field(fd.name, expr)

    def _set_field(self, key, value):
        self._fields[key] = value
        if key not in dir(self):
            self.__dict__[key] = value

    def __getitem__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise AttributeError("Could not find field `" + str(item) + "' in schema.")

    def __iter__(self):
        return iter([self._fields[f.name] for f in self._type.fields])


class AtomicExpression(Expression):
    def to_float64(self):
        return self._method("toFloat64", TFloat64())

    def to_float32(self):
        return self._method("toFloat32", TFloat32())

    def to_int64(self):
        return self._method("toInt64", TInt64())

    def to_int32(self):
        return self._method("toInt32", TInt32())


class BooleanExpression(AtomicExpression):
    def _bin_op_logical(self, name, other):
        other = to_expr(other)
        return self._bin_op(name, other, TBoolean())

    def __and__(self, other):
        return self._bin_op_logical("&&", other)

    def __or__(self, other):
        return self._bin_op_logical("||", other)


class NumericExpression(AtomicExpression):
    def _bin_op_ret_typ(self, other):
        if isinstance(self, NumericExpression) and isinstance(other, float):
            return TFloat64()
        elif isinstance(self, Float64Expression) and isinstance(other, NumericExpression):
            return TFloat64()
        elif isinstance(self, NumericExpression) and isinstance(other, Float64Expression):
            return TFloat64()
        elif isinstance(self, Float32Expression) and isinstance(other, NumericExpression):
            return TFloat32()
        elif isinstance(self, NumericExpression) and isinstance(other, Float32Expression):
            return TFloat32()
        elif isinstance(self, Int64Expression) and (
                        isinstance(other, NumericExpression) or isinstance(other, int) or isinstance(other, long)):
            return TInt64()
        elif isinstance(self, NumericExpression) and isinstance(other, Int64Expression):
            return TInt64()
        elif isinstance(self, Int32Expression) and (isinstance(other, NumericExpression) or isinstance(other, int)):
            return TInt32()
        elif isinstance(self, NumericExpression) or isinstance(other, Int32Expression):
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
        return self._bin_op("<", other, TBoolean())

    def __le__(self, other):
        return self._bin_op("<=", other, TBoolean())

    def __gt__(self, other):
        return self._bin_op(">", other, TBoolean())

    def __ge__(self, other):
        return self._bin_op(">=", other, TBoolean())

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

    def __mod__(self, other):
        return self._bin_op_numeric('%', other)

    def __rmod__(self, other):
        return self._bin_op_numeric_reverse('%', other)

    def __pow__(self, power, modulo=None):
        return self._bin_op('**', power, TFloat64())

    def __rpow__(self, other):
        return self._bin_op_reverse('**', other, TFloat64())

    def signum(self):
        return self._method("signum", TInt32())

    def abs(self):
        return self._method("abs", self._type)

    def max(self, other):
        assert (isinstance(other, self.__class__))
        return self._method("max", self._type, other)

    def min(self, other):
        assert (isinstance(other, self.__class__))
        return self._method("min", self._type, other)


class Float64Expression(NumericExpression):
    pass


class Float32Expression(NumericExpression):
    pass


class Int32Expression(NumericExpression):
    pass


class Int64Expression(NumericExpression):
    pass


class StringExpression(AtomicExpression):
    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._slice(TString(), item.start, item.stop, item.step)
        elif isinstance(item, int):
            return self._index(TString(), item)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        if not isinstance(other, StringExpression) or isinstance(other, str):
            raise TypeError("cannot concatenate 'TString' expression and '{}'".format(
                other._type.__class__ if isinstance(other, Expression) else other.__class__))
        return self._bin_op("+", other, TString())

    def __radd__(self, other):
        assert (isinstance(other, StringExpression) or isinstance(other, str))
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


class CallExpression(Expression):
    @staticmethod
    @args_to_expr
    def from_int32(i):
        return CallExpression(ApplyMethod('Call', i._ast), TCall(), i._indices, i._source, i._aggregations)

    @property
    def gt(self):
        return self._field("gt", TInt32())

    def gtj(self):
        return self._method("gtj", TInt32())

    def gtk(self):
        return self._method("gtk", TInt32())

    def is_non_ref(self):
        return self._method("isNonRef", TBoolean())

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

    def num_nonref_alleles(self):
        return self._method("nNonRefAlleles", TInt32())

    def one_hot_alleles(self, v):
        return self._method("oneHotAlleles", TArray(TInt32()), v)

    def one_hot_genotype(self, v):
        return self._method("oneHotGenotype", TArray(TInt32()), v)


class LocusExpression(Expression):
    @property
    def contig(self):
        return self._field("contig", TString())

    @property
    def position(self):
        return self._field("position", TInt32())


class IntervalExpression(Expression):
    @typecheck_method(locus=oneof(LocusExpression, hail.genetics.Locus))
    def contains(self, locus):
        locus = to_expr(locus)
        if not locus._type._rg == self._type._rg:
            raise TypeError('Reference genome mismatch: {}, {}'.format(self._type._rg, locus._type._rg))
        return self._method("contains", TBoolean(), locus)

    @property
    def end(self):
        return self._field("end", TLocus())

    @property
    def start(self):
        return self._field("start", TLocus())


class AltAlleleExpression(Expression):
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


class VariantExpression(Expression):
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


typ_to_expr = {
    TBoolean: BooleanExpression,
    TInt32: Int32Expression,
    TInt64: Int64Expression,
    TFloat64: Float64Expression,
    TFloat32: Float32Expression,
    TLocus: LocusExpression,
    TInterval: IntervalExpression,
    TVariant: VariantExpression,
    TCall: CallExpression,
    TAltAllele: AltAlleleExpression,
    TString: StringExpression,
    TDict: DictExpression,
    TArray: ArrayExpression,
    TSet: SetExpression,
    TStruct: StructExpression
}

elt_typ_to_array_expr = {
    TBoolean: ArrayBooleanExpression,
    TInt32: ArrayInt32Expression,
    TFloat64: ArrayFloat64Expression,
    TInt64: ArrayInt64Expression,
    TFloat32: ArrayFloat32Expression,
    TString: ArrayStringExpression,
    TStruct: ArrayStructExpression,
    TArray: ArrayArrayExpression
}

elt_typ_to_set_expr = {
    TInt32: SetInt32Expression,
    TFloat64: SetFloat64Expression,
    TFloat32: SetFloat32Expression,
    TInt64: SetFloat32Expression,
    TString: SetStringExpression,
    TSet: SetSetExpression
}


class ExpressionException(Exception):
    def __init__(self, msg=''):
        self.msg = msg
        super(ExpressionException, self).__init__(msg)


class ExpressionWarning(Warning):
    def __init__(self, msg=''):
        self.msg = msg
        super(ExpressionWarning, self).__init__(msg)


@typecheck(expr=Expression,
           expected_indices=Indices,
           aggregation_axes=setof(strlike),
           scoped_variables=setof(strlike))
def analyze(expr, expected_indices, aggregation_axes, scoped_variables=None):
    indices = expr._indices
    source = indices.source
    axes = indices.axes
    aggregations = expr._aggregations

    warnings = []
    errors = []

    expected_source = expected_indices.source
    expected_axes = expected_indices.axes

    if source is not None and source is not expected_source:
        errors.append(
            ExpressionException('Expected an expression from source {}, found expression derived from {}'.format(
                expected_source, source
            )))

    # TODO: use ast.search to find the references to bad-indexed exprs

    # check for stray indices
    unexpected_axes = axes - expected_axes
    if unexpected_axes:
        errors.append(ExpressionException(
            'out-of-scope expression: expected an expression indexed by {}, found indices {}'.format(
                list(expected_axes),
                list(indices.axes)
            )))

    if aggregations:
        if aggregation_axes:
            expected_agg_axes = expected_axes.union(aggregation_axes)
            for agg in aggregations:
                agg_indices = agg.indices
                agg_axes = agg_indices.axes
                if agg_indices.source is not None and agg_indices.source is not expected_source:
                    errors.append(
                        ExpressionException('Expected an expression from source {}, found expression derived from {}'.format(
                            expected_source, source
                        )))

                # check for stray indices
                unexpected_agg_axes = agg_axes - expected_agg_axes
                if unexpected_agg_axes:
                    errors.append(ExpressionException(
                        'out-of-scope expression: expected an expression indexed by {}, found indices {}'.format(
                            list(expected_agg_axes),
                            list(agg_axes)
                        )
                    ))

                # check for low-complexity aggregations
                missing_agg_axes = expected_agg_axes - agg_axes
                if missing_agg_axes:
                    pass
                    # warnings.append(ExpressionWarning(
                    #     'low complexity: expected aggregation expressions indexed by {} (aggregating along {}), '
                    #     'but found indices {}. The aggregated expression will be identical for all results along axis {}'.format(
                    #         list(expected_agg_axes),
                    #         list(expected_agg_axes - agg_axes),
                    #         list(agg_axes),
                    #         list(missing_agg_axes)
                    #     )
                    # ))
        else:
            errors.append(ExpressionException('invalid aggregation: no aggregations supported in this method'))
    else:
        # check for low-complexity operations
        missing_axes = expected_axes - axes
        if missing_axes:
            pass
            warnings.append(ExpressionWarning(
                'low complexity: expected an expression indexed by {}, found indices {}. Results will be broadcast along {}.'.format(
                    list(expected_axes),
                    list(axes),
                    list(missing_axes)
                )
            ))

    # this may already be checked by the above, but it seems like a good idea
    references = [r.name for r in expr._ast.search(lambda ast: isinstance(ast, Reference) and ast.top_level)]
    for r in references:
        if not r in scoped_variables:
            errors.append(ExpressionException("scope exception: referenced out-of-scope field '{}'".format(r)))

    for w in warnings:
        warn('Analysis warning: {}'.format(w.msg))
    if errors:
        for e in errors:
            error('Analysis exception: {}'.format(e.msg))
        raise errors[0]
