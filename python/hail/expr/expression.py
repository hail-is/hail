from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.expr.ast import *
from hail.expr.types import *
from hail.utils.java import *
from hail.genetics import Locus, Variant, Interval, Call


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
        self.temp_vars = temp_vars


@typecheck(ast=AST, type=Type, indices=Indices, aggregations=tupleof(Aggregation), joins=tupleof(Join))
def construct_expr(ast, type, indices=Indices(), aggregations=(), joins=()):
    if isinstance(type, TArray) and type.element_type.__class__ in elt_typ_to_array_expr:
        return elt_typ_to_array_expr[type.element_type.__class__](ast, type, indices, aggregations, joins)
    elif isinstance(type, TSet) and type.element_type.__class__ in elt_typ_to_set_expr:
        return elt_typ_to_set_expr[type.element_type.__class__](ast, type, indices, aggregations, joins)
    elif type.__class__ in typ_to_expr:
        return typ_to_expr[type.__class__](ast, type, indices, aggregations, joins)
    else:
        raise NotImplementedError(type)


def to_expr(e):
    if isinstance(e, Expression):
        return e
    elif isinstance(e, str) or isinstance(e, unicode):
        return construct_expr(Literal('"{}"'.format(e)), TString())
    elif isinstance(e, bool):
        return construct_expr(Literal("true" if e else "false"), TBoolean())
    elif isinstance(e, int):
        return construct_expr(Literal(str(e)), TInt32())
    elif isinstance(e, long):
        return construct_expr(ClassMethod('toInt64', Literal('"{}"'.format(e))), TInt64())
    elif isinstance(e, float):
        return construct_expr(Literal(str(e)), TFloat64())
    elif isinstance(e, Variant):
        return construct_expr(ApplyMethod('Variant', Literal('"{}"'.format(str(e)))), TVariant(e.reference_genome))
    elif isinstance(e, Locus):
        return construct_expr(ApplyMethod('Locus', Literal('"{}"'.format(str(e)))), TLocus(e.reference_genome))
    elif isinstance(e, Interval):
        return construct_expr(ApplyMethod('Interval', Literal('"{}"'.format(str(e)))), TInterval(e.reference_genome))
    elif isinstance(e, Struct):
        attrs = e._attrs.items()
        cols = [to_expr(x) for _, x in attrs]
        names = [k for k, _ in attrs]
        indices, aggregations, joins = unify_all(*cols)
        t = TStruct(names, [col._type for col in cols])
        return construct_expr(StructDeclaration(names, [c._ast for c in cols]),
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
        return construct_expr(ArrayDeclaration([col._ast for col in cols]),
                              t, indices, aggregations, joins)
    elif isinstance(e, set):
        cols = [to_expr(x) for x in e]
        types = list({col._type for col in cols})
        if len(cols) == 0:
            raise ValueError("Don't support empty sets.")
        elif len(types) == 1:
            t = TSet(types[0])
        elif all([is_numeric(t) for t in types]):
            t = TSet(convert_numeric_typ(*types))
        else:
            raise ValueError("Don't support sets with multiple element types.")
        indices, aggregations, joins = unify_all(*cols)
        return construct_expr(ClassMethod('toSet', ArrayDeclaration([col._ast for col in cols])), t, indices,
                              aggregations,
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
        return construct_expr(ast, TDict(key_type, value_type), indices, aggregations, joins)
    else:
        raise ValueError("Cannot implicitly capture value `{}' with type `{}'.".format(e, e.__class__))


@decorator
def args_to_expr(func, *args):
    return func(*(to_expr(a) for a in args))


def unify_all(*exprs):
    assert len(exprs) > 0
    new_indices = Indices.unify(*[e._indices for e in exprs])
    agg = list(exprs[0]._aggregations)
    joins = list(exprs[0]._joins)
    for e in exprs[1:]:
        agg.extend(e._aggregations)
        joins.extend(e._joins)
    return new_indices, tuple(agg), tuple(joins)


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


class Expression(object):
    """Base class for Hail expressions."""

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
        raise NotImplementedError(
            "The truth value of an expression is undefined\n  Hint: instead of if/else, use 'f.cond'")

    def _unary_op(self, name):
        return construct_expr(UnaryOperation(self._ast, name),
                              self._type, self._indices, self._aggregations, self._joins)

    def _bin_op(self, name, other, ret_typ):
        other = to_expr(other)
        indices, aggregations, joins = unify_all(self, other)
        return construct_expr(BinaryOperation(self._ast, other._ast, name), ret_typ, indices, aggregations, joins)

    def _bin_op_reverse(self, name, other, ret_typ):
        other = to_expr(other)
        indices, aggregations, joins = unify_all(self, other)
        return construct_expr(BinaryOperation(other._ast, self._ast, name), ret_typ, indices, aggregations, joins)

    def _field(self, name, ret_typ):
        return construct_expr(Select(self._ast, name), ret_typ, self._indices, self._aggregations, self._joins)

    def _method(self, name, ret_typ, *args):
        args = tuple(to_expr(arg) for arg in args)
        indices, aggregations, joins = unify_all(self, *args)
        return construct_expr(ClassMethod(name, self._ast, *(a._ast for a in args)),
                              ret_typ, indices, aggregations, joins)

    def _index(self, ret_typ, key):
        key = to_expr(key)
        indices, aggregations, joins = unify_all(self, key)
        return construct_expr(Index(self._ast, key._ast),
                              ret_typ, indices, aggregations, joins)

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
        return construct_expr(Index(self._ast, Slice(start_ast, stop_ast)),
                              ret_typ, indices, aggregations, joins)

    def _bin_lambda_method(self, name, f, inp_typ, ret_typ_f, *args):
        args = (to_expr(arg) for arg in args)
        new_id = Env._get_uid()
        lambda_result = to_expr(
            f(construct_expr(Reference(new_id), inp_typ, self._indices, self._aggregations, self._joins)))
        indices, aggregations, joins = unify_all(self, lambda_result)
        ast = LambdaClassMethod(name, new_id, self._ast, lambda_result._ast, *(a._ast for a in args))
        return construct_expr(ast, ret_typ_f(lambda_result._type), indices, aggregations, joins)

    def __eq__(self, other):
        return self._bin_op("==", other, TBoolean())

    def __ne__(self, other):
        return self._bin_op("!=", other, TBoolean())


class CollectionExpression(Expression):
    """Expression of type :py:class:`hail.expr.types.TArray` or :py:class:`hail.expr.types.TSet`"""

    def contains(self, item):
        """Returns a boolean indicating whether `item` is found in the collection.

        Parameters
        ----------
        item

        Returns
        -------
        :py:class:`BooleanExpression`
            `True` if the element is found in the collection, `False` otherwise.
        """
        item = to_expr(item)
        return self.exists(lambda x: x == item)

    def exists(self, f):
        """Returns `True` if `f` returns `True` for any element.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection. Must return a
            :py:class`BooleanExpression`.

        Returns
        -------
        :py:class:`BooleanExpression`.
            `True` if `f` returns `True` for any element, `False` otherwise.
        """
        return self._bin_lambda_method("exists", f, self._type.element_type, lambda t: TBoolean())

    def filter(self, f):
        """Returns a new collection containing elements where `f` returns `True`.

        Returns a same-type expression; evaluated on a :py:class`SetExpression`, returns a
        :py:class`SetExpression`. Evaluated on an :py:class`ArrayExpression`,
        returns an :py:class`ArrayExpression`.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection. Must return a
            :py:class`BooleanExpression`.

        Returns
        -------
        :py:class:`CollectionExpression`
            Expression of the same type as the callee.
        """
        return self._bin_lambda_method("filter", f, self._type.element_type, lambda t: self._type)

    def find(self, f):
        """Returns the first element where `f` returns `True`.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection. Must return a
            :py:class`BooleanExpression`.

        Returns
        -------
        :py:class:`Expression`
            Expression whose type is the element type of the collection.
        """
        return self._bin_lambda_method("find", f, self._type.element_type, lambda t: self._type.element_type)

    def flatmap(self, f):
        """Map each element of the collection to a new collection, and flatten the results.

        Parameters
        ----------
        f : callable
            Function from the element type of the collection to the type of the
            collection. For instance, `flatmap` on a ``Set[String]`` should take
            a ``String`` and return a ``Set``.

        Returns
        -------
        :py:class:`CollectionExpression`
        """
        return self._bin_lambda_method("flatMap", f, self._type.element_type, lambda t: t)

    def forall(self, f):
        """Returns `True` if `f` returns `True` for every element.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection. Must return a
            :py:class`BooleanExpression`.

        Returns
        -------
        :py:class:`BooleanExpression`.
            `True` if `f` returns `True` for every element, `False` otherwise.
        """
        return self._bin_lambda_method("forall", f, self._type.element_type, lambda t: TBoolean())

    def group_by(self, f):
        """Group elements into a dict according to a lambda function.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection to produce a key for the
            resulting dictionary.

        Returns
        -------
        :py:class:`DictExpression`.
        """
        return self._bin_lambda_method("groupBy", f, self._type.element_type, lambda t: TDict(t, self._type))

    def map(self, f):
        """Transform each element of a collection.

        Parameters
        ----------
        f : callable
            Function to transform each element of the collection.

        Returns
        -------
        :py:class:`CollectionExpression`.
            Collection where each element has been transformed according to `f`.
        """
        return self._bin_lambda_method("map", f, self._type.element_type, lambda t: self._type.__class__(t))

    def length(self):
        """Returns the size of a collection.

        Returns
        -------
        :py:class:`hail.expr.Int32Expression`
            The number of elements in the collection.
        """
        return self._method("size", TInt32())

    def size(self):
        """Returns the size of a collection.

        Returns
        -------
        :py:class:`hail.expr.Int32Expression`
            The number of elements in the collection.
        """
        return self._method("size", TInt32())



class CollectionNumericExpression(CollectionExpression):
    """Expression of type :class:`hail.expr.types.TArray` or :class:`hail.expr.types.TSet` with numeric element type."""
    def max(self):
        """Returns the maximum element.

        Returns
        -------
        :py:class:`NumericExpression`
            The maximum value in the collection.
        """
        return self._method("max", self._type.element_type)

    def min(self):
        """Returns the minimum element.

        Returns
        -------
        :py:class:`NumericExpression`
            The miniumum value in the collection.
        """
        return self._method("min", self._type.element_type)

    def mean(self):
        """Returns the mean of all values in the collection.

        Note
        ----
        Missing elements are ignored.

        Returns
        -------
        :py:class:`Float64Expression`
            The mean value of the collection.
        """
        return self._method("mean", TFloat64())

    def median(self):
        """Returns the median element.

        Note
        ----
        Missing elements are ignored.

        Returns
        -------
        :py:class:`NumericExpression`
            The median value in the collection.
        """
        return self._method("median", self._type.element_type)

    def product(self):
        """Returns the product of all elements in the collection.

        Note
        ----
        Missing elements are ignored.

        Returns
        -------
        :py:class:`Float64Expression`
            The product of the collection.
        """
        return self._method("product",
                            TInt64() if isinstance(self._type.element_type, TInt32) or
                                        isinstance(self._type.element_type, TInt64) else TFloat64)

    def sum(self):
        """Returns the sum of all elements in the collection.

        Note
        ----
        Missing elements are ignored.

        Returns
        -------
        :py:class:`Float64Expression`
            The sum of the collection.
        """
        return self._method("sum", self._type.element_type)


class ArrayExpression(CollectionExpression):
    """Expression of type :class:`hail.expr.types.TArray`."""
    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._slice(self._type, item.start, item.stop, item.step)
        elif isinstance(item, int) or isinstance(item, Int32Expression):
            return self._index(self._type.element_type, item)
        else:
            raise NotImplementedError

    def append(self, x):
        """Append an element to the array and return the result.

        Parameters
        ----------
        x
            Element to append, same type as the array element type.

        Returns
        -------
        :py:class:`ArrayExpression`
        """
        return self._method("append", self._type, x)

    def extend(self, a):
        """Concatenate two arrays and return the result.

        Parameters
        ----------
        a
            Array to concatenate, same type as the callee.

        Returns
        -------
        :py:class:`ArrayExpression`
        """
        return self._method("extend", self._type, a)

    def sort_by(self, f, ascending=True):
        """Sort the array according to a function.

        Parameters
        ----------
        f : callable
            Function to evaluate per element to obtain the sort key.
        ascending : bool or :py:class:`BooleanExpression`
            Sort in ascending order.

        Returns
        -------
        :py:class:`ArrayExpression`
        """
        return self._bin_lambda_method("sortBy", f, self._type.element_type, lambda t: self._type, ascending)

    def to_set(self):
        """Convert the array to a set.

        Returns
        -------
        :py:class:`SetExpression`
        """
        return self._method("toSet", TSet(self._type.element_type))


class ArrayNumericExpression(ArrayExpression, CollectionNumericExpression):
    """Expression of type :class:`hail.expr.types.TArray` with a numeric type.

    Numeric arrays support arithmetic both with scalar values and other arrays. Arithmetic
    between two numeric arrays requires that the length of each array is identical, and will
    apply the operation positionally (``a1 * a2`` will multiply the first element of ``a1`` by
    the first element of ``a2``, the second element of ``a1`` by the second element of ``a2``,
    and so on). Arithmetic with a scalar will apply the operation to each element of the array.
    """
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
        other = to_expr(other)
        ret_typ = self._bin_op_ret_typ(other)
        return self._bin_op(name, other, ret_typ)

    def _bin_op_numeric_reverse(self, name, other):
        other = to_expr(other)
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

    def sort(self, ascending=True):
        """Returns a sorted array.

        Parameters
        ----------
        ascending : bool or :py:class:`BooleanExpression`
            Sort in ascending order.

        Returns
        -------
        :py:class:`ArrayNumericExpression`
            Sorted array.
        """
        return self._method("sort", self._type, ascending)


class ArrayFloat64Expression(ArrayNumericExpression):
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TFloat64`."""
    pass


class ArrayFloat32Expression(ArrayNumericExpression):
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TFloat32`."""
    pass


class ArrayInt32Expression(ArrayNumericExpression):
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TInt32`"""
    pass


class ArrayInt64Expression(ArrayNumericExpression):
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TInt64`."""
    pass


class ArrayStringExpression(ArrayExpression):
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TString`."""
    def mkstring(self, delimiter):
        """Joins the elements of the array into a single string delimited by `delimiter`.

        Parameters
        ----------
        delimiter : str or :py:class:`StringExpression`
            Field delimiter.
        Returns
        -------
        :py:class:`StringExpression`
        """
        return self._method("mkString", TString(), delimiter)

    def sort(self, ascending=True):
        """Returns a sorted array.

        Parameters
        ----------
        ascending : bool or :py:class:`BooleanExpression`
            Sort in ascending order.

        Returns
        -------
        :py:class:`ArrayStringExpression`
            Sorted array.
        """
        return self._method("sort", self._type, ascending)


class ArrayStructExpression(ArrayExpression):
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TStruct`."""
    pass


class ArrayArrayExpression(ArrayExpression):
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TArray`."""
    def flatten(self):
        """Flatten the nested array by concatenating subarrays.

        Returns
        -------
        :py:class:`ArrayExpression`
            Array generated by concatenating all subarrays.
        """
        return self._method("flatten", self._type.element_type)


class SetExpression(CollectionExpression):
    """Expression of type :class:`hail.expr.types.TSet`."""
    def add(self, x):
        return self._method("add", self._type, x)

    def remove(self, x):
        """Returns the result of removing the argument from this Set.

        Parameters
        ----------
        x

        Returns
        -------
        :py:class:`SetExpression`
            This set with the element `x` removed.

        Examples
        --------
        .. doctest::

            >>> eval_expr(functions.capture({1,2,3}).remove(1))
            {2, 3}

        """
        return self._method("remove", self._type, x)

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

    def to_array(self):
        return self._method("toArray", TArray(self._type.element_type))


class SetFloat64Expression(SetExpression, CollectionNumericExpression):
    """Expression of type :class:`hail.expr.types.TSet` with element type :class:`hail.expr.types.TFloat64`."""
    pass


class SetFloat32Expression(SetExpression, CollectionNumericExpression):
    """Expression of type :class:`hail.expr.types.TSet` with element type :class:`hail.expr.types.TFloat32`."""
    pass


class SetInt32Expression(SetExpression, CollectionNumericExpression):
    """Expression of type :class:`hail.expr.types.TSet` with element type :class:`hail.expr.types.TInt32`."""
    pass


class SetInt64Expression(SetExpression, CollectionNumericExpression):
    """Expression of type :class:`hail.expr.types.TSet` with element type :class:`hail.expr.types.TInt64`."""
    pass


class SetStringExpression(SetExpression):
    """Expression of type :class:`hail.expr.types.TSet` with element type :class:`hail.expr.types.TString`."""
    def mkstring(self, delimiter):
        """Joins the elements of the set into a single string delimited by `delimiter`.

        Parameters
        ----------
        delimiter : str or :py:class:`StringExpression`
            Field delimiter.
        Returns
        -------
        :py:class:`StringExpression`
        """
        return self._method("mkString", TString(), delimiter)


class SetSetExpression(SetExpression):
    """Expression of type :class:`hail.expr.types.TSet` with element type :class:`hail.expr.types.TSet`."""
    def flatten(self):
        """Flatten the nested set by concatenating subsets.

        Returns
        -------
        :py:class:`SetExpression`
            Set generated by concatenating all subsets.
        """
        return self._method("flatten", self._type.element_type)


class DictExpression(Expression):
    """Expression of type :class:`hail.expr.types.TDict`."""
    def _init(self):
        self._key_typ = self._type.key_type
        self._value_typ = self._type.value_type

    def __getitem__(self, item):
        return self._index(self._value_typ, item)

    def contains(self, k):
        """Returns whether a given key is present in the dictionary.

        Parameters
        ----------
        k
            Key to test for inclusion.

        Returns
        -------
        :py:class:`BooleanExpression`
            `True` if `k` is a key of the dictionary, `False` otherwise.
        """
        return self._method("contains", TBoolean(), k)

    def get(self, k):
        """Returns the value corresponding to a given key, or missing if that key is not found.

        Parameters
        ----------
        k
            Key.

        Returns
        -------
        :py:class:`Expression`
            The value associated with `k`, or missing.
        """
        return self._method("get", self._value_typ, k)

    def key_set(self):
        """Returns the set of keys in the dictionary.

        Returns
        -------
        :py:class:`SetExpression`
            Set of all keys.
        """
        return self._method("keySet", TSet(self._key_typ))

    def keys(self):
        """Returns an array with all keys in the dictionary.

        Returns
        -------
        :py:class:`Expression`
            Array of all keys.
        """
        return self._method("keys", TArray(self._key_typ))

    def map_values(self, f):
        """Transform values of the dictionary according to a function.

        Parameters
        ----------
        f : callable
            Function to apply to each value.

        Returns
        -------
        :py:class:`DictExpression`
            Dictionary with transformed values.
        """
        return self._bin_lambda_method("mapValues", f, self._value_typ, lambda t: TDict(self._key_typ, t))

    def size(self):
        """Returns the size of the dictionary.

        Returns
        -------
        :py:class:`Int32Expression`
            Size of the dictionary.
        """
        return self._method("size", TInt32())

    def values(self):
        """Returns an array with all values in the dictionary.

        Returns
        -------
        :py:class:`ArrayExpression`
            All values in the dictionary.
        """
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
    """Expression of type :class:`hail.expr.types.TStruct`."""
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
    """Abstract base class for numeric and logical types."""
    def to_float64(self):
        """Convert to a 64-bit floating point expression.

        Returns
        -------
        :py:class:`Float64Expression`
        """
        return self._method("toFloat64", TFloat64())

    def to_float32(self):
        """Convert to a 32-bit floating point expression.

        Returns
        -------
        :py:class:`Float32Expression`
        """
        return self._method("toFloat32", TFloat32())

    def to_int64(self):
        """Convert to a 64-bit integer expression.

        Returns
        -------
        :py:class:`Int64Expression`
        """
        return self._method("toInt64", TInt64())

    def to_int32(self):
        """Convert to a 32-bit integer expression.

        Returns
        -------
        :py:class:`Int32Expression`
        """
        return self._method("toInt32", TInt32())


class BooleanExpression(AtomicExpression):
    """Expression of type :class:`hail.expr.types.TBoolean`.

    Use the bitwise operators ``|`` and ``&`` for boolean comparisons.
    """
    def _bin_op_logical(self, name, other):
        other = to_expr(other)
        return self._bin_op(name, other, TBoolean())

    def __and__(self, other):
        return self._bin_op_logical("&&", other)

    def __or__(self, other):
        return self._bin_op_logical("||", other)

    def __invert__(self):
        return self._unary_op("!")


class NumericExpression(AtomicExpression):
    """Expression of numeric type."""
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
        """Returns the sign of the callee, ``1`` or ``-1``.

        Returns
        -------
        :py:class:`Int32Expression`
            ``1`` or ``-1``.
        """
        return self._method("signum", TInt32())

    def abs(self):
        """Returns the absolute value of the callee.

        Returns
        -------
        :py:class:`.NumericExpression`
        """
        return self._method("abs", self._type)

    def max(self, other):
        """Returns the maximum value between the callee and `other`.

        Parameters
        ----------
        other : :py:class:`NumericExpression`
            Value to compare against.

        Returns
        -------
        :py:class:`.NumericExpression`
            Maximum value.
        """
        assert (isinstance(other, self.__class__))
        return self._method("max", self._type, other)

    def min(self, other):
        """Returns the minumum value between the callee and `other`.

        Parameters
        ----------
        other : :py:class:`NumericExpression`
            Value to compare against.

        Returns
        -------
        :py:class:`.NumericExpression`
            Minimum value.
        """
        assert (isinstance(other, self.__class__))
        return self._method("min", self._type, other)


class Float64Expression(NumericExpression):
    """Expression of type :class:`hail.expr.types.TFloat64`."""
    pass


class Float32Expression(NumericExpression):
    """Expression of type :class:`hail.expr.types.TFloat32`."""
    pass


class Int32Expression(NumericExpression):
    """Expression of type :class:`hail.expr.types.TInt32`."""
    pass


class Int64Expression(NumericExpression):
    """Expression of type :class:`hail.expr.types.TInt64`."""
    pass


class StringExpression(AtomicExpression):
    """Expression of type :class:`hail.expr.types.TString`."""
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
        """Returns the length of the string.

        Returns
        -------
        :class:`Int32Expression`
            Length of the string.
        """
        return self._method("length", TInt32())

    def replace(self, pattern1, pattern2):
        """Replace substrings matching `pattern1` with `pattern2` using regex.

        The regex expressions used should follow
        `Java regex syntax <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`_

        Parameters
        ----------
        pattern1 : str or :class:`StringExpression`
        pattern2 : str or :class:`StringExpression`

        Returns
        -------

        """
        return self._method("replace", TString(), pattern1, pattern2)

    def split(self, delim, n=None):
        """Returns an array of strings generated by splitting the string at `delim`.

        The delimiter is a regex using the
        `Java regex syntax <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`_ delimiter.
        To split on special characters, escape them with double backslash (``\\\\``).

        Parameters
        ----------
        delim : str or :class:`StringExpression`
            Delimiter regex.
        n : int or :class:`Int32Expression`, optional
            Maximum number of splits.

        Returns
        -------
        :class:`ArrayStringExpression`
        """
        if n is None:
            return self._method("split", TArray(TString()), delim)
        else:
            return self._method("split", TArray(TString()), delim, n)


class CallExpression(Expression):
    """Expression of type :class:`hail.expr.types.TCall`."""
    @property
    def gt(self):
        """Returns the triangle number of :py:meth:`Call.gtj` and :py:meth:`Call.gtk`.

        Returns
        -------
        :class:`Int32Expression`
            Triangle number of the two alleles.
        """
        return self._field("gt", TInt32())

    def gtj(self):
        """Returns the allele index of the first allele.

        Returns
        -------
        :class:`Int32Expression`
            First allele index.
        """
        return self._method("gtj", TInt32())

    def gtk(self):
        """Returns the allele index of the second allele.

        Returns
        -------
        :class:`Int32Expression`
            Second allele index.
        """
        return self._method("gtk", TInt32())

    def is_non_ref(self):
        """Evaluate whether the call includes one or more non-reference alleles.

        Returns
        -------
        :class:`BooleanExpression`
            `True` if at least one allele is non-reference, `False` otherwise.
        """
        return self._method("isNonRef", TBoolean())

    def is_het(self):
        """Evaluate whether the call includes two different alleles.

        Returns
        -------
        :class:`BooleanExpression`
            `True` if the two alleles are different, `False` if they are the same.
        """
        return self._method("isHet", TBoolean())

    def is_het_nonref(self):
        """Evaluate whether the call includes two different alleles, neither of which is reference.

        Returns
        -------
        :class:`BooleanExpression`
            `True` if the call includes two different alternate alleles, `False` otherwise.
        """
        return self._method("isHetNonRef", TBoolean())

    def is_het_ref(self):
        """Evaluate whether the call includes two different alleles, one of which is reference.

        Returns
        -------
        :class:`BooleanExpression`
            `True` if the call includes one reference and one alternate allele, `False` otherwise.
        """
        return self._method("isHetRef", TBoolean())

    def is_hom_ref(self):
        """Evaluate whether the call includes two reference alleles.

        Returns
        -------
        :class:`BooleanExpression`
            `True` if the call includes two reference alleles, `False` otherwise.
        """
        return self._method("isHomRef", TBoolean())

    def is_hom_var(self):
        """Evaluate whether the call includes two identical alternate alleles.

        Returns
        -------
        :class:`BooleanExpression`
            `True` if the call includes two identical alternate alleles, `False` otherwise.
        """
        return self._method("isHomVar", TBoolean())

    def num_alt_alleles(self):
        """Returns the number of non-reference alleles (0, 1, or 2).

        Returns
        -------
        :class:`Int32Expression`
            The number of non-reference alleles (0, 1, or 2).
        """
        return self._method("nNonRefAlleles", TInt32())

    def one_hot_alleles(self, v):
        """Returns an array containing the summed one-hot encoding of the two alleles.

        This one-hot representation is the positional sum of the one-hot encoding for
        each called allele.  For a biallelic variant, the one-hot encoding for a
        reference allele is ``[1, 0]`` and the one-hot encoding for an alternate allele
        is ``[0, 1]``. Diploid calls would produce the following arrays: ``[2, 0]`` for
        homozygous reference, ``[1, 1]`` for heterozygous, and ``[0, 2]`` for homozygous
        alternate.


        Parameters
        ----------
        v : :class:`hail.genetics.Variant` or :class:`VariantExpression`
            Variant, used to determine the number of possible alleles.

        Returns
        -------
        :class:`ArrayInt32Expression`
            An array of summed one-hot encodings of allele indices.
        """
        return self._method("oneHotAlleles", TArray(TInt32()), v)

    def one_hot_genotype(self, v):
        """Returns the triangle number of the genotype one-hot encoded into an integer array.

        A one-hot encoding uses a vector to represent categorical data, like a genotype call,
        by using an array with one ``1`` and many ``0`` s. In this case, the array will have a
        ``1`` at the position of the triangle number of the genotype (:py:meth:`Call.gt`).

        This is useful for manipulating counts of categorical variables.

        Parameters
        ----------
        v : :class:`hail.genetics.Variant` or :class:`VariantExpression`
            Variant, used to determine the number of possible alleles.
        Returns
        -------
        :class:`ArrayInt32Expression`
            An array with a one-hot encoding of the call.
        """
        return self._method("oneHotGenotype", TArray(TInt32()), v)


class LocusExpression(Expression):
    """Expression of type :class:`hail.expr.types.TLocus`."""
    @property
    def contig(self):
        """Returns the chromosome.

        Returns
        -------
        :class:`StringExpression`
            The chromosome for this locus.
        """
        return self._field("contig", TString())

    @property
    def position(self):
        """Returns the position along the chromosome.

        Returns
        -------
        :class:`Int32Expression`
            This locus's position along its chromosome.
        """
        return self._field("position", TInt32())


class IntervalExpression(Expression):
    """Expression of type :class:`hail.expr.types.TInterval`."""
    @typecheck_method(locus=oneof(LocusExpression, Locus))
    def contains(self, locus):
        """Tests whether a locus is contained in the interval.

        Parameters
        ----------
        locus : :class:`hail.genetics.Locus` or :class:`LocusExpression`
            Locus to test for interval membership.
        Returns
        -------
        :class:`BooleanExpression`
            `True` if `locus` is contained in the interval, `False` otherwise.
        """
        locus = to_expr(locus)
        if not locus._type._rg == self._type._rg:
            raise TypeError('Reference genome mismatch: {}, {}'.format(self._type._rg, locus._type._rg))
        return self._method("contains", TBoolean(), locus)

    @property
    def end(self):
        """Returns the end locus.

        Returns
        -------
        :class:`LocusExpression`
            End locus.
        """
        return self._field("end", TLocus())

    @property
    def start(self):
        """Returns the start locus.

        Returns
        -------
        :class:`LocusExpression`
            Start locus.
        """
        return self._field("start", TLocus())


class AltAlleleExpression(Expression):
    """Expression of type :class:`hail.expr.types.TAltAllele`."""
    @property
    def alt(self):
        """Returns the alternate allele string.

        Returns
        -------
        :class:`StringExpression`
        """
        return self._field("alt", TString())

    @property
    def ref(self):
        """Returns the reference allele string.

        Returns
        -------
        :class:`StringExpression`
        """
        return self._field("ref", TString())

    def category(self):
        """Returns the string representation of the alternate allele class.

        Returns
        -------
        :class:`StringExpression`
            One of ``"SNP"``, ``"MNP"``, ``"Insertion"``, ``"Deletion"``, ``"Star"``, ``"Complex"``.
        """
        return self._method("category", TString())

    def is_complex(self):
        """Returns true if the polymorphism is not a SNP, MNP, star, insertion, or deletion.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isComplex", TBoolean())

    def is_deletion(self):
        """Returns true if the polymorphism is a deletion.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isDeletion", TBoolean())

    def is_indel(self):
        """Returns true if the polymorphism is an insertion or deletion.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isIndel", TBoolean())

    def is_insertion(self):
        """Returns true if the polymorphism is an insertion.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isInsertion", TBoolean())

    def is_mnp(self):
        """Returns true if the polymorphism is a MNP.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isMNP", TBoolean())

    def is_snp(self):
        """Returns true if the polymorphism is a SNP.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isSNP", TBoolean())

    def is_star(self):
        """Returns true if the polymorphism is a star (upstream deletion) allele.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isStar", TBoolean())

    def is_transition(self):
        """Returns true if the polymorphism is a transition SNP.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isTransition", TBoolean())

    def is_transversion(self):
        """Returns true if the polymorphism is a transversion SNP.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isTransversion", TBoolean())


class VariantExpression(Expression):
    """Expression of type :class:`hail.expr.types.TVariant`."""
    def alt(self):
        """Returns the alternate allele string.

        Warning
        -------
        Assumes biallelic. If the variant is multiallelic, throws an error.

        Returns
        -------
        :class:`StringExpression`
        """
        return self._method("alt", TString())

    def alt_allele(self):
        """Returns the alternate allele.

        Warning
        -------
        Assumes biallelic. If the variant is multiallelic, throws an error.

        Returns
        -------
        :class:`AltAlleleExpression`
        """
        return self._method("altAllele", TAltAllele())

    @property
    def alt_alleles(self):
        """Returns the alternate alleles in the polymorphism.

        Returns
        -------
        :class:`ArrayExpression` with element type :class:`AltAlleleExpression`
        """
        return self._field("altAlleles", TArray(TAltAllele()))

    @property
    def contig(self):
        """Returns the chromosome.

        Returns
        -------
        :class:`StringExpression`
        """
        return self._field("contig", TString())

    def in_x_nonpar(self):
        """Returns true if the polymorphism is in a non-pseudoautosomal region of chromosome X.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("inXNonPar", TBoolean())

    def in_x_par(self):
        """Returns true if the polymorphism is in the pseudoautosomal region of chromosome X.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("inXPar", TBoolean())

    def in_y_nonpar(self):
        """Returns true if the polymorphism is in a non-pseudoautosomal region of chromosome Y.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("inYNonPar", TBoolean())

    def in_y_par(self):
        """Returns true if the polymorphism is in a pseudoautosomal region of chromosome Y.

        Note
        ----
        Most variant callers only generate variants on chromosome X for the pseudoautosomal region.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("inYPar", TBoolean())

    def is_autosomal(self):
        """Returns true if the polymorphism is found on an autosome.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isAutosomal", TBoolean())

    def is_biallelic(self):
        """Returns true if there is only one alternate allele.

        Returns
        -------
        :class:`BooleanExpression`
        """
        return self._method("isBiallelic", TBoolean())

    def locus(self):
        """Returns the locus at which the polymorphism occurs.

        Returns
        -------
        :class:`LocusExpression`
        """
        return self._method("locus", TLocus())

    def num_alleles(self):
        """Returns the number of alleles in the polymorphism, including the reference.

        Returns
        -------
        :class:`Int32Expression`
        """
        return self._method("nAlleles", TInt32())

    def num_alt_alleles(self):
        """Returns the number of alleles in the polymorphism, excluding the reference.

        Returns
        -------
        :class:`Int32Expression`
        """
        return self._method("nAltAlleles", TInt32())

    def num_genotypes(self):
        """Returns the number of possible genotypes given the number of total alleles.

        Returns
        -------
        :class:`Int32Expression`
        """
        return self._method("nGenotypes", TInt32())

    @property
    def ref(self):
        """Returns the reference allele string.

        Returns
        -------
        :class:`StringExpression`
        """
        return self._field("ref", TString())

    @property
    def start(self):
        """Returns the chromosomal position of the polymorphism.

        Returns
        -------
        :class:`Int32Expression`
        """
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
                        ExpressionException(
                            'Expected an expression from source {}, found expression derived from {}'.format(
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


@args_to_expr
def eval_expr(expression):
    """Evaluate a Hail expression, returning the result.

    This method is extremely useful for learning about Hail expressions and understanding
    how to compose them.

    Expressions that refer to fields of :class:`hail.api2.Table` or :class:`hail.api.MatrixTable`
    objects cannot be evaluated.

    Examples
    --------
    Evaluate a conditional:

    .. doctest::

        >>> x = 6
        >>> eval_expr(functions.cond(x % 2 == 0, 'Even', 'Odd'))
        'Even'

    Parameters
    ----------
    expression : :class:`hail.expr.expression.Expression`
        Any expression, or a Python value that can be implicitly interpreted as an expression.

    Returns
    -------
    any
        Result of evaluating `expression`.
    """
    return eval_expr_typed(expression)[0]


@args_to_expr
def eval_expr_typed(expression):
    """Evaluate a Hail expression, returning the result and the type of the result.

    This method is extremely useful for learning about Hail expressions and understanding
    how to compose them.

    Expressions that refer to fields of :class:`hail.api2.Table` or :class:`hail.api.MatrixTable`
    objects cannot be evaluated.

    Examples
    --------
    Evaluate a conditional:

    .. doctest::

        >>> x = 6
        >>> eval_expr_typed(functions.cond(x % 2 == 0, 'Even', 'Odd'))
        ('Odd', TString())

    Parameters
    ----------
    expression : :class:`hail.expr.expression.Expression`
        Any expression, or a Python value that can be implicitly interpreted as an expression.

    Returns
    -------
    (any, :class:`hail.expr.Type`)
        Result of evaluating `expression`, and its type.
    """
    analyze(expression, Indices(), set(), set())
    if len(expression._joins) > 0:
        raise ExpressionException("'eval_expr' methods do not support joins or broadcasts")
    r, t = Env.hc().eval_expr_typed(expression._ast.to_hql())
    return r, t

expr_int32 = oneof(Int32Expression, int)
expr_numeric = oneof(Float32Expression, Float64Expression, Int64Expression, float, expr_int32)
expr_list = oneof(list, ArrayExpression)
expr_set = oneof(set, SetExpression)
expr_bool = oneof(bool, BooleanExpression)
expr_struct = oneof(Struct, StructExpression)
expr_str = oneof(strlike, StringExpression)
expr_variant = oneof(Variant, VariantExpression)
expr_locus = oneof(Locus, LocusExpression)
expr_call = oneof(Call, CallExpression)
