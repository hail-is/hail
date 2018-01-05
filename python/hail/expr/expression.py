from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.expr.ast import *
from hail.expr.types import *
from hail.utils.java import *
from hail.utils.queue import Queue
from hail.genetics import Locus, Variant, Interval, Call, AltAllele
from hail.typecheck import *

def to_expr(e):
    if isinstance(e, Expression):
        return e
    elif isinstance(e, str) or isinstance(e, unicode):
        return construct_expr(Literal('"{}"'.format(Env.jutils().escapePyString(e))), TString())
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
    elif isinstance(e, Call):
        return construct_expr(ApplyMethod('Call', Literal(str(e.gt))), TCall())
    elif isinstance(e, AltAllele):
        return construct_expr(ApplyMethod('AltAllele', to_expr(e.ref)._ast, to_expr(e.alt)._ast), TAltAllele())
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


_lazy_int32 = lazy()
_lazy_numeric = lazy()
_lazy_array = lazy()
_lazy_set = lazy()
_lazy_bool = lazy()
_lazy_struct = lazy()
_lazy_string = lazy()
_lazy_variant = lazy()
_lazy_locus = lazy()
_lazy_altallele = lazy()
_lazy_interval = lazy()
_lazy_call = lazy()
_lazy_expr = lazy()

expr_int32 = transformed((_lazy_int32, identity),
                         (integral, to_expr))
expr_numeric = transformed((_lazy_numeric, identity),
                           (integral, to_expr),
                           (float, to_expr))
expr_list = transformed((list, to_expr),
                        (_lazy_array, identity))
expr_set = transformed((set, to_expr),
                       (_lazy_set, identity))
expr_bool = transformed((bool, to_expr),
                        (_lazy_bool, identity))
expr_struct = transformed((Struct, to_expr),
                          (_lazy_struct, identity))
expr_str = transformed((strlike, to_expr),
                       (_lazy_string, identity))
expr_variant = transformed((Variant, to_expr),
                           (_lazy_variant, identity))
expr_locus = transformed((Locus, to_expr),
                         (_lazy_locus, identity))
expr_altallele = transformed((AltAllele, to_expr),
                             (_lazy_altallele, identity))
expr_interval = transformed((Interval, to_expr),
                            (_lazy_interval, identity))
expr_call = transformed((Call, to_expr),
                        (_lazy_call, identity))
expr_any = transformed((_lazy_expr, identity),
                       (anytype, to_expr))


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


def unify_all(*exprs):
    assert len(exprs) > 0
    new_indices = Indices.unify(*[e._indices for e in exprs])
    first, rest = exprs[0], exprs[1:]
    agg = first._aggregations
    joins = first._joins
    for e in rest:
        agg = agg.push(*e._aggregations)
        joins = joins.push(*e._joins)
    return new_indices, agg, joins


__numeric_types = [TInt32, TInt64, TFloat32, TFloat64]


@typecheck(t=Type)
def is_numeric(t):
    return t.__class__ in __numeric_types


@typecheck(types=Type)
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

    @typecheck_method(ast=AST, type=Type, indices=Indices, aggregations=Queue, joins=Queue)
    def __init__(self, ast, type, indices=Indices(), aggregations=Queue(), joins=Queue()):
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
        """Returns ``True`` if the two epressions are equal.

        Examples
        --------

        .. doctest::

            >>> x = functions.capture(5)
            >>> y = functions.capture(5)
            >>> z = functions.capture(1)

            >>> eval_expr(x == y)
            True

            >>> eval_expr(x == z)
            False

        Notes
        -----
        This method will fail with an error if the two expressions are not
        of comparable types.

        Parameters
        ----------
        other : :class:`Expression`
            Expression for equality comparison.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the two expressions are equal.
        """
        return self._bin_op("==", other, TBoolean())

    def __ne__(self, other):
        """Returns ``True`` if the two expressions are not equal.

        Examples
        --------

        .. doctest::

            >>> x = functions.capture(5)
            >>> y = functions.capture(5)
            >>> z = functions.capture(1)

            >>> eval_expr(x != y)
            False

            >>> eval_expr(x != z)
            True

        Notes
        -----
        This method will fail with an error if the two expressions are not
        of comparable types.

        Parameters
        ----------
        other : :class:`Expression`
            Expression for inequality comparison.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the two expressions are not equal.
        """
        return self._bin_op("!=", other, TBoolean())


class CollectionExpression(Expression):
    """Expression of type :py:class:`hail.expr.types.TArray` or :py:class:`hail.expr.types.TSet`

    >>> a = functions.capture([1, 2, 3, 4, 5])

    >>> s = functions.capture({'Alice', 'Bob', 'Charlie'})
    """

    def exists(self, f):
        """Returns ``True`` if `f` returns ``True`` for any element.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.exists(lambda x: x % 2 == 0))
            True

            >>> eval_expr(s.exists(lambda x: x[0] == 'D'))
            False

        Notes
        -----
        This method always returns ``False`` for empty collections.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection. Must return a
            :py:class:`BooleanExpression`.

        Returns
        -------
        :py:class:`BooleanExpression`.
            ``True`` if `f` returns ``True`` for any element, ``False`` otherwise.
        """
        return self._bin_lambda_method("exists", f, self._type.element_type, lambda t: TBoolean())

    def filter(self, f):
        """Returns a new collection containing elements where `f` returns ``True``.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.filter(lambda x: x % 2 == 0))
            [2, 4]

            >>> eval_expr(s.filter(lambda x: ~(x[-1] == 'e')))
            {'Bob'}

        Notes
        -----
        Returns a same-type expression; evaluated on a :py:class:`SetExpression`, returns a
        :py:class:`SetExpression`. Evaluated on an :py:class:`ArrayExpression`,
        returns an :py:class:`ArrayExpression`.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection. Must return a
            :py:class:`BooleanExpression`.

        Returns
        -------
        :py:class:`CollectionExpression`
            Expression of the same type as the callee.
        """
        return self._bin_lambda_method("filter", f, self._type.element_type, lambda t: self._type)

    def find(self, f):
        """Returns the first element where `f` returns ``True``.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.find(lambda x: x ** 2 > 20))
            5

            >>> eval_expr(s.find(lambda x: x[0] == 'D'))
            None

        Notes
        -----
        If `f` returns ``False`` for every element, then the result is missing.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection. Must return a
            :py:class:`BooleanExpression`.

        Returns
        -------
        :py:class:`Expression`
            Expression whose type is the element type of the collection.
        """
        return self._bin_lambda_method("find", f, self._type.element_type, lambda t: self._type.element_type)

    def flatmap(self, f):
        """Map each element of the collection to a new collection, and flatten the results.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.flatmap(lambda x: functions.range(0, x)))
            [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]

            >>> eval_expr(s.flatmap(lambda x: functions.range(0, x.length()).map(lambda i: x[i]).to_set()))
            {'A', 'B', 'C', 'a', 'b', 'c', 'e', 'h', 'i', 'l', 'o', 'r'}

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
        """Returns ``True`` if `f` returns ``True`` for every element.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.forall(lambda x: x < 10))
            True

        Notes
        -----
        This method returns ``True`` if the collection is empty.

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection. Must return a
            :py:class:`BooleanExpression`.

        Returns
        -------
        :py:class:`BooleanExpression`.
            ``True`` if `f` returns ``True`` for every element, ``False`` otherwise.
        """
        return self._bin_lambda_method("forall", f, self._type.element_type, lambda t: TBoolean())

    def group_by(self, f):
        """Group elements into a dict according to a lambda function.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.group_by(lambda x: x % 2 == 0))
            {False: [1, 3, 5], True: [2, 4]}

            >>> eval_expr(s.group_by(lambda x: x.length()))
            {3: {'Bob'}, 5: {'Alice'}, 7: {'Charlie'}}

        Parameters
        ----------
        f : callable
            Function to evaluate for each element of the collection to produce a key for the
            resulting dictionary.

        Returns
        -------
        :py:class:`DictExpression`.
            Dictionary keyed by results of `f`.
        """
        return self._bin_lambda_method("groupBy", f, self._type.element_type, lambda t: TDict(t, self._type))

    def map(self, f):
        """Transform each element of a collection.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.map(lambda x: x ** 3))
            [1.0, 8.0, 27.0, 64.0, 125.0]

            >>> eval_expr(s.map(lambda x: x.length()))
            {3, 5, 7}

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

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.length())
            5

            >>> eval_expr(s.length())
            3

        Returns
        -------
        :py:class:`hail.expr.Int32Expression`
            The number of elements in the collection.
        """
        return self._method("size", TInt32())

    def size(self):
        """Returns the size of a collection.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.size())
            5

            >>> eval_expr(s.size())
            3

        Returns
        -------
        :py:class:`hail.expr.Int32Expression`
            The number of elements in the collection.
        """
        return self._method("size", TInt32())


class CollectionNumericExpression(CollectionExpression):
    """Expression of type :class:`hail.expr.types.TArray` or :class:`hail.expr.types.TSet` with numeric element type.

    >>> a = functions.capture([1, 2, 3, 4, 5])
    """

    def max(self):
        """Returns the maximum element.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.max())
            5

        Returns
        -------
        :py:class:`NumericExpression`
            The maximum value in the collection.
        """
        return self._method("max", self._type.element_type)

    def min(self):
        """Returns the minimum element.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.min())
            1

        Returns
        -------
        :py:class:`NumericExpression`
            The miniumum value in the collection.
        """
        return self._method("min", self._type.element_type)

    def mean(self):
        """Returns the mean of all values in the collection.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.mean())
            3.0

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

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.median())
            3

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

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.product())
            120

        Note
        ----
        Missing elements are ignored.

        Returns
        -------
        :py:class:`NumericExpression`
            The product of the collection.
        """
        return self._method("product",
                            TInt64() if isinstance(self._type.element_type, TInt32) or
                                        isinstance(self._type.element_type, TInt64) else TFloat64())

    def sum(self):
        """Returns the sum of all elements in the collection.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.sum())
            15

        Note
        ----
        Missing elements are ignored.

        Returns
        -------
        :py:class:`NumericExpression`
            The sum of the collection.
        """
        return self._method("sum", self._type.element_type)


class ArrayExpression(CollectionExpression):
    """Expression of type :class:`hail.expr.types.TArray`.

    >>> a = functions.capture(['Alice', 'Bob', 'Charlie'])
    """
    def __getitem__(self, item):
        """Index into or slice the array.

        Examples
        --------

        Index with a single integer:

        .. doctest::

            >>> eval_expr(a[1])
            'Bob'

            >>> eval_expr(a[-1])
            'Charlie'

        Slicing is also supported:

        .. doctest::

            >>> eval_expr(a[1:])
            ['Bob', 'Charlie']

        Parameters
        ----------
        item : slice or :class:`Int32Expression`
            Index or slice.

        Returns
        -------
        :class:`Expression`
            Element or array slice.
        """
        if isinstance(item, slice):
            return self._slice(self._type, item.start, item.stop, item.step)
        elif isinstance(item, int) or isinstance(item, Int32Expression):
            return self._index(self._type.element_type, item)
        else:
            raise NotImplementedError

    @typecheck_method(item=expr_any)
    def contains(self, item):
        """Returns a boolean indicating whether `item` is found in the array.

        Examples
        --------

        .. doctest::

            >>> eval_expr(a.contains('Charlie'))
            True

            >>> eval_expr(a.contains('Helen'))
            False

        Parameters
        ----------
        item : :class:`Expression`
            Item for inclusion test.

        Warning
        -------
        This method takes time proportional to the length of the array. If a
        pipeline uses this method on the same array several times, it may be
        more efficient to convert the array to a set first
        (:meth:`ArrayExpression.to_set`).

        Returns
        -------
        :py:class:`BooleanExpression`
            ``True`` if the element is found in the array, ``False`` otherwise.
        """
        return self.exists(lambda x: x == item)

    @typecheck_method(x=expr_any)
    def append(self, x):
        """Append an element to the array and return the result.

        Examples
        --------

        .. doctest::

            >>> eval_expr(a.append('Dan'))
            ['Alice', 'Bob', 'Charlie', 'Dan']

        Parameters
        ----------
        x : :class:`Expression`
            Element to append, same type as the array element type.

        Returns
        -------
        :py:class:`ArrayExpression`
        """
        return self._method("append", self._type, x)

    @typecheck_method(a=expr_list)
    def extend(self, a):
        """Concatenate two arrays and return the result.

        Examples
        --------

        .. doctest::

            >>> eval_expr(a.extend(['Dan', 'Edith']))
            ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith']

        Parameters
        ----------
        a : :class:`ArrayExpression`
            Array to concatenate, same type as the callee.

        Returns
        -------
        :py:class:`ArrayExpression`
        """
        return self._method("extend", self._type, a)

    def sort_by(self, f, ascending=True):
        """Sort the array according to a function.

        Examples
        --------

        .. doctest::

            >>> eval_expr(a.sort_by(lambda x: x, ascending=False))
            ['Charlie', 'Bob', 'Alice']

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

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.to_set())
            {'Alice', 'Bob', 'Charlie'}

        Returns
        -------
        :py:class:`SetExpression`
            Set of all unique elements.
        """
        return self._method("toSet", TSet(self._type.element_type))


class ArrayNumericExpression(ArrayExpression, CollectionNumericExpression):
    """Expression of type :class:`hail.expr.types.TArray` with a numeric type.

    Numeric arrays support arithmetic both with scalar values and other arrays.
    Arithmetic between two numeric arrays requires that the length of each array
    is identical, and will apply the operation positionally (``a1 * a2`` will
    multiply the first element of ``a1`` by the first element of ``a2``, the
    second element of ``a1`` by the second element of ``a2``, and so on).
    Arithmetic with a scalar will apply the operation to each element of the
    array.

    >>> a1 = functions.capture([0, 1, 2, 3, 4, 5])

    >>> a2 = functions.capture([1, -1, 1, -1, 1, -1])

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
        """Positionally add an array or a scalar.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a1 + 5)
            [5, 6, 7, 8, 9, 10]

            >>> eval_expr(a1 + a2)
            [1, 0, 3, 2, 5, 4]

        Parameters
        ----------
        other : :class:`NumericExpression` or :class:`ArrayNumericExpression`
            Value or array to add.

        Returns
        -------
        :class:`ArrayNumericExpression`
            Array of positional sums.
        """
        return self._bin_op_numeric("+", other)

    def __radd__(self, other):
        return self._bin_op_numeric_reverse("+", other)

    def __sub__(self, other):
        """Positionally subtract an array or a scalar.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a2 - 1)
            [0, -2, 0, -2, 0, -2]

            >>> eval_expr(a1 - a2)
            [-1, 2, 1, 4, 3, 6]

        Parameters
        ----------
        other : :class:`NumericExpression` or :class:`ArrayNumericExpression`
            Value or array to subtract.

        Returns
        -------
        :class:`ArrayNumericExpression`
            Array of positional differences.
        """
        return self._bin_op_numeric("-", other)

    def __rsub__(self, other):
        return self._bin_op_numeric_reverse("-", other)

    def __mul__(self, other):
        """Positionally multiply by an array or a scalar.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a2 * 5)
            [5, -5, 5, -5, 5, -5]

            >>> eval_expr(a1 * a2)
            [0, -1, 2, -3, 4, -5]

        Parameters
        ----------
        other : :class:`NumericExpression` or :class:`ArrayNumericExpression`
            Value or array to multiply by.

        Returns
        -------
        :class:`ArrayNumericExpression`
            Array of positional products.
        """
        return self._bin_op_numeric("*", other)

    def __rmul__(self, other):
        return self._bin_op_numeric_reverse("*", other)

    def __div__(self, other):
        """Positionally divide by an array or a scalar.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a1 / 10)
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

            >>> eval_expr(a2 / a1)
            [inf, -1.0, 0.5, -0.3333333333333333, 0.25, -0.2]

        Parameters
        ----------
        other : :class:`NumericExpression` or :class:`ArrayNumericExpression`
            Value or array to divide by.

        Returns
        -------
        :class:`ArrayNumericExpression`
            Array of positional quotients.
        """
        return self._bin_op("/", other, TArray(TFloat64()))

    def __rdiv__(self, other):
        return self._bin_op_reverse("/", other, TArray(TFloat64()))

    def __pow__(self, other):
        """Positionally raise to the power of an array or a scalar.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a1 ** 2)
            [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]

            >>> eval_expr(a1 ** a2)
            [0.0, 1.0, 2.0, 0.3333333333333333, 4.0, 0.2]

        Parameters
        ----------
        other : :class:`NumericExpression` or :class:`ArrayNumericExpression`
            Value or array to exponentiate by.

        Returns
        -------
        :class:`ArrayNumericExpression`
            Array of positional exponentiations.
        """
        return self._bin_op('**', other, TArray(TFloat64()))

    def __rpow__(self, other):
        return self._bin_op_reverse('**', other, TArray(TFloat64()))

    def sort(self, ascending=True):
        """Returns a sorted array.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a1.sort())
            [-1, -1, -1, 1, 1, 1]

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
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TString`.

    >>> a = functions.capture(['Alice', 'Bob', 'Charles'])
    """
    def mkstring(self, delimiter):
        """Joins the elements of the array into a single string delimited by `delimiter`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.mkstring(','))
            'Alice,Bob,Charles'

        Parameters
        ----------
        delimiter : str or :py:class:`StringExpression`
            Field delimiter.

        Returns
        -------
        :py:class:`StringExpression`
            Joined string expression.
        """
        return self._method("mkString", TString(), delimiter)

    def sort(self, ascending=True):
        """Returns a sorted array.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.sort(ascending=False))
            ['Charles', 'Bob', 'Alice']

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
    """Expression of type :class:`hail.expr.types.TArray` with element type :class:`hail.expr.types.TArray`.

    >>> a = functions.capture([[1, 2], [3, 4]])
    """
    def flatten(self):
        """Flatten the nested array by concatenating subarrays.

        Examples
        --------
        .. doctest::

            >>> eval_expr(a.flatten())
            [1, 2, 3, 4]

        Returns
        -------
        :py:class:`ArrayExpression`
            Array generated by concatenating all subarrays.
        """
        return self._method("flatten", self._type.element_type)


class SetExpression(CollectionExpression):
    """Expression of type :class:`hail.expr.types.TSet`.

    >>> s1 = functions.capture({1, 2, 3})
    >>> s2 = functions.capture({1, 3, 5})
    """
    @typecheck_method(x=expr_any)
    def add(self, x):
        """Returns a new set including `x`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s1.add(10))
            {1, 2, 3, 10}

        Parameters
        ----------
        x : :class:`Expression`
            Value to add.

        Returns
        -------
        :class:`SetExpression`
            Set with `x` added.
        """
        return self._method("add", self._type, x)

    @typecheck_method(x=expr_any)
    def remove(self, x):
        """Returns a new set excluding `x`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s1.remove(1))
            {2, 3}

        Parameters
        ----------
        x : :class:`Expression`
            Value to remove.

        Returns
        -------
        :py:class:`SetExpression`
            Set with `x` removed.
        """
        return self._method("remove", self._type, x)

    @typecheck_method(x=expr_any)
    def contains(self, x):
        """Returns ``True`` if `x` is in the set.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s1.contains(1))
            True

            >>> eval_expr(s1.contains(10))
            False

        Parameters
        ----------
        x : :class:`Expression`
            Value for inclusion test..

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if `x` is in the set.
        """
        return self._method("contains", TBoolean(), x)

    @typecheck_method(s=expr_set)
    def difference(self, s):
        """Return the set of elements in the set that are not present in set `s`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s1.difference(s2))
            {2}

            >>> eval_expr(s2.difference(s1))
            {5}

        Parameters
        ----------
        s : :class:`SetExpression`
            Set expression of the same type.

        Returns
        -------
        :class:`SetExpression`
            Set of elements not in `s`.
        """
        return self._method("difference", self._type, s)

    @typecheck_method(s=expr_set)
    def intersection(self, s):
        """Return the intersection of the set and set `s`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s1.intersection(s2))
            {1, 3}

        Parameters
        ----------
        s : :class:`SetExpression`
            Set expression of the same type.

        Returns
        -------
        :class:`SetExpression`
            Set of elements present in `s`.
        """
        return self._method("intersection", self._type, s)

    @typecheck_method(s=expr_set)
    def is_subset(self, s):
        """Returns ``True`` if every element is contained in set `s`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s1.is_subset(s2))
            False

            >>> eval_expr(s1.remove(2).is_subset(s2))
            True

        Parameters
        ----------
        s : :class:`SetExpression`
            Set expression of the same type.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if every element is contained in set `s`.
        """
        return self._method("isSubset", TBoolean(), s)

    @typecheck_method(s=expr_set)
    def union(self, s):
        """Return the union of the set and set `s`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s1.union(s2))
            {1, 2, 3, 5}

        Parameters
        ----------
        s : :class:`SetExpression`
            Set expression of the same type.

        Returns
        -------
        :class:`SetExpression`
            Set of elements present in either set.
        """
        return self._method("union", self._type, s)

    def to_array(self):
        """Convert the set to an array.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s1.to_array())
            [1, 2, 3]

        Notes
        -----
        The order of elements in the array is not guaranteed.

        Returns
        -------
        :class:`ArrayExpression`
            Array of all elements in the set.
        """
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
    """Expression of type :class:`hail.expr.types.TSet` with element type :class:`hail.expr.types.TString`.

    >>> s = functions.capture({'Alice', 'Bob', 'Charles'})
    """
    @typecheck_method(delimiter=expr_str)
    def mkstring(self, delimiter):
        """Joins the elements of the set into a single string delimited by `delimiter`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s.mkstring(','))
            'Bob,Charles,Alice'

        Notes
        -----
        The order of the elements in the string is not guaranteed.

        Parameters
        ----------
        delimiter : str or :py:class:`StringExpression`
            Field delimiter.

        Returns
        -------
        :py:class:`StringExpression`
            Joined string expression.
        """
        return self._method("mkString", TString(), delimiter)


class SetSetExpression(SetExpression):
    """Expression of type :class:`hail.expr.types.TSet` with element type :class:`hail.expr.types.TSet`.

    >>> s = functions.capture({1, 2, 3}).map(lambda s: {s, 2 * s})
    """
    def flatten(self):
        """Flatten the nested set by concatenating subsets.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s.flatten())
            {1, 2, 3, 4, 6}

        Returns
        -------
        :py:class:`SetExpression`
            Set generated by concatenating all subsets.
        """
        return self._method("flatten", self._type.element_type)


class DictExpression(Expression):
    """Expression of type :class:`hail.expr.types.TDict`.

    >>> d = functions.capture({'Alice': 43, 'Bob': 33, 'Charles': 44})
    """
    def _init(self):
        self._key_typ = self._type.key_type
        self._value_typ = self._type.value_type

    @typecheck_method(item=expr_any)
    def __getitem__(self, item):
        """Get the value associated with key `item`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(d['Alice'])
            43

        Notes
        -----
        Raises an error if `item` is not a key of the dictionary. Use
        :meth:`DictExpression.get` to return missing instead of an error.

        Parameters
        ----------
        item : :class:`Expression`
            Key expression.

        Returns
        -------
        :class:`Expression`
            Value associated with key `item`.
        """
        return self._index(self._value_typ, item)

    @typecheck_method(k=expr_any)
    def contains(self, k):
        """Returns whether a given key is present in the dictionary.

        Examples
        --------
        .. doctest::

            >>> eval_expr(d.contains('Alice'))
            True

            >>> eval_expr(d.contains('Anne'))
            False

        Parameters
        ----------
        k : :class:`Expression`
            Key to test for inclusion.

        Returns
        -------
        :py:class:`BooleanExpression`
            ``True`` if `k` is a key of the dictionary, ``False`` otherwise.
        """
        return self._method("contains", TBoolean(), k)

    @typecheck_method(k=expr_any)
    def get(self, k):
        """Returns the value associated with key `k`, or missing if that key is not present.

        Examples
        --------
        .. doctest::

            >>> eval_expr(d.get('Alice'))
            43

            >>> eval_expr(d.get('Anne'))
            None

        Parameters
        ----------
        k : :class:`Expression`
            Key.

        Returns
        -------
        :py:class:`Expression`
            The value associated with `k`, or missing.
        """
        return self._method("get", self._value_typ, k)

    def key_set(self):
        """Returns the set of keys in the dictionary.

        Examples
        --------
        .. doctest::

            >>> eval_expr(d.key_set())
            {'Alice', 'Bob', 'Charles'}

        Returns
        -------
        :py:class:`SetExpression`
            Set of all keys.
        """
        return self._method("keySet", TSet(self._key_typ))

    def keys(self):
        """Returns an array with all keys in the dictionary.

        Examples
        --------
        .. doctest::

            >>> eval_expr(d.keys())
            ['Bob', 'Charles', 'Alice']

        Returns
        -------
        :py:class:`ArrayExpression`
            Array of all keys.
        """
        return self._method("keys", TArray(self._key_typ))

    def map_values(self, f):
        """Transform values of the dictionary according to a function.

        Examples
        --------
        .. doctest::

            >>> eval_expr(d.map_values(lambda x: x * 10))
            {'Alice': 430, 'Bob': 330, 'Charles': 440}

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

        Examples
        --------
        .. doctest::

            >>> eval_expr(d.size())
            3

        Returns
        -------
        :py:class:`Int32Expression`
            Size of the dictionary.
        """
        return self._method("size", TInt32())

    def values(self):
        """Returns an array with all values in the dictionary.

        Examples
        --------
        .. doctest::

            >>> eval_expr(d.values())
            [33, 44, 43]

        Returns
        -------
        :py:class:`ArrayExpression`
            All values in the dictionary.
        """
        return self._method("values", TArray(self._value_typ))


class Aggregable(object):
    """Expression that can only be aggregated.

    An :class:`Aggregable` is produced by the :meth:`explode` or :meth:`filter`
    methods. These objects can be aggregated using aggregator functions, but
    cannot otherwise be used in expressions.
    """
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
    """Expression of type :class:`hail.expr.types.TStruct`.

    >>> s = functions.capture(Struct(a=5, b='Foo'))

    Struct fields are accessible as attributes and keys. It is therefore
    possible to access field `a` of struct `s` with dot syntax:

    .. doctest::

        >>> eval_expr(s.a)
        5

    It is possible to have fields whose names violate Python identifier syntax.
    For these, use the :meth:`StructExpression.__getitem__` syntax.
    """
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

    @typecheck_method(item=strlike)
    def __getitem__(self, item):
        """Access a field of the struct.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s['a'])
            5

        Parameters
        ----------
        item : :obj:`str`
            Field name.

        Returns
        -------
        :class:`Expression`
            Struct field.
        """

        if item in self._fields:
            return self._fields[item]
        else:
            raise AttributeError("No field '{}' in schema. Available fields: {}".format(
                item, [f.name for f in self._type.fields]))

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

    >>> t = functions.capture(True)
    >>> f = functions.capture(False)
    >>> na = functions.null(TBoolean())

    .. doctest::

        >>> eval_expr(t)
        True

        >>> eval_expr(f)
        False

        >>> eval_expr(na)
        None

    """

    def _bin_op_logical(self, name, other):
        other = to_expr(other)
        return self._bin_op(name, other, TBoolean())

    @typecheck_method(other=expr_bool)
    def __and__(self, other):
        """Return ``True`` if the left and right arguments are ``True``.

        Examples
        --------
        .. doctest::

            >>> eval_expr(t & f)
            False

            >>> eval_expr(t & na)
            None

            >>> eval_expr(f & na)
            False

        The ``&`` and ``|`` operators have higher priority than comparison
        operators like ``==``, ``<``, or ``>``. Parentheses are often
        necessary:

        .. doctest::

            >>> x = functions.capture(5)

            >>> eval_expr((x < 10) & (x > 2))
            True

        Parameters
        ----------
        other : :class:`BooleanExpression`
            Right-side operation.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if both left and right are ``True``.
        """
        return self._bin_op_logical("&&", other)

    def __or__(self, other):
        """Return ``True`` if at least one of the left and right arguments is ``True``.

        Examples
        --------
        .. doctest::

            >>> eval_expr(t | f)
            True

            >>> eval_expr(t | na)
            True

            >>> eval_expr(f | na)
            None

        The ``&`` and ``|`` operators have higher priority than comparison
        operators like ``==``, ``<``, or ``>``. Parentheses are often
        necessary:

        .. doctest::

            >>> x = functions.capture(5)

            >>> eval_expr((x < 10) | (x > 20))
            True

        Parameters
        ----------
        other : :class:`BooleanExpression`
            Right-side operation.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if either left or right is ``True``.
        """
        return self._bin_op_logical("||", other)

    def __invert__(self):
        """Return the boolean inverse.

        Examples
        --------
        .. doctest::

            >>> eval_expr(~t)
            False

            >>> eval_expr(~f)
            True

            >>> eval_expr(~na)
            None

        Returns
        -------
        :class:`BooleanExpression`
            Boolean inverse.
        """
        return self._unary_op("!")


class NumericExpression(AtomicExpression):
    """Expression of numeric type.

    >>> x = functions.capture(3)

    >>> y = functions.capture(4.5)
    """
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

    @typecheck_method(other=expr_numeric)
    def __lt__(self, other):
        """Less-than comparison.

        Examples
        --------
        .. doctest::

            >>> eval_expr(x < 5)
            True

        Parameters
        ----------
        other : :class:`NumericExpression`
            Right side for comparison.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the left side is smaller than the right side.
        """
        return self._bin_op("<", other, TBoolean())

    @typecheck_method(other=expr_numeric)
    def __le__(self, other):
        """Less-than-or-equals comparison.

        Examples
        --------
        .. doctest::

            >>> eval_expr(x <= 3)
            True

        Parameters
        ----------
        other : :class:`NumericExpression`
            Right side for comparison.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the left side is smaller than or equal to the right side.
        """
        return self._bin_op("<=", other, TBoolean())

    @typecheck_method(other=expr_numeric)
    def __gt__(self, other):
        """Greater-than comparison.

        Examples
        --------
        .. doctest::

            >>> eval_expr(y > 4)
            True

        Parameters
        ----------
        other : :class:`NumericExpression`
            Right side for comparison.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the left side is greater than the right side.
        """
        return self._bin_op(">", other, TBoolean())

    @typecheck_method(other=expr_numeric)
    def __ge__(self, other):
        """Greater-than-or-equals comparison.

        Examples
        --------
        .. doctest::

            >>> eval_expr(y >= 4)
            True

        Parameters
        ----------
        other : :class:`NumericExpression`
            Right side for comparison.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the left side is greater than or equal to the right side.
        """
        return self._bin_op(">=", other, TBoolean())


    def __pos__(self):
        return self

    def __neg__(self):
        """Negate the number (multiply by -1).

        Examples
        --------
        .. doctest::

            >>> eval_expr(-x)
            -3

        Returns
        -------
        :class:`NumericExpression`
            Negated number.
        """
        return self._unary_op("-")

    def __add__(self, other):
        """Add two numbers.

        Examples
        --------
        .. doctest::

            >>> eval_expr(x + 2)
            5

            >>> eval_expr(x + y)
            7.5

        Parameters
        ----------
        other : :class:`NumericExpression`
            Number to add.

        Returns
        -------
        :class:`NumericExpression`
            Sum of the two numbers.
        """
        return self._bin_op_numeric("+", other)

    def __radd__(self, other):
        return self._bin_op_numeric_reverse("+", other)

    def __sub__(self, other):
        """Subtract the right number from the left.

        Examples
        --------
        .. doctest::

            >>> eval_expr(x - 2)
            1

            >>> eval_expr(x - y)
            -1.5

        Parameters
        ----------
        other : :class:`NumericExpression`
            Number to subtract.

        Returns
        -------
        :class:`NumericExpression`
            Difference of the two numbers.
        """
        return self._bin_op_numeric("-", other)

    def __rsub__(self, other):
        return self._bin_op_numeric_reverse("-", other)

    def __mul__(self, other):
        """Multiply two numbers.

        Examples
        --------
        .. doctest::

            >>> eval_expr(x * 2)
            6

            >>> eval_expr(x * y)
            9.0

        Parameters
        ----------
        other : :class:`NumericExpression`
            Number to multiply.

        Returns
        -------
        :class:`NumericExpression`
            Product of the two numbers.
        """
        return self._bin_op_numeric("*", other)

    def __rmul__(self, other):
        return self._bin_op_numeric_reverse("*", other)

    def __div__(self, other):
        """Divide two numbers.

        Examples
        --------
        .. doctest::

            >>> eval_expr(x / 2)
            1.5

            >>> eval_expr(y / 0.1)
            45.0

        Parameters
        ----------
        other : :class:`NumericExpression`
            Dividend.

        Returns
        -------
        :class:`NumericExpression`
            The left number divided by the left.
        """
        return self._bin_op_numeric("/", other)

    def __rdiv__(self, other):
        return self._bin_op_numeric_reverse("/", other)

    def __mod__(self, other):
        """Compute the left modulo the right number.

        Examples
        --------
        .. doctest::

            >>> eval_expr(32 % x)
            2

            >>> eval_expr(7 % y)
            2.5

        Parameters
        ----------
        other : :class:`NumericExpression`
            Dividend.

        Returns
        -------
        :class:`NumericExpression`
            Remainder after dividing the left by the right.
        """
        return self._bin_op_numeric('%', other)

    def __rmod__(self, other):
        return self._bin_op_numeric_reverse('%', other)

    def __pow__(self, power, modulo=None):
        """Raise the left to the right power.

        Examples
        --------
        .. doctest::

            >>> eval_expr(x ** 2)
            9.0

            >>> eval_expr(x ** -2)
            0.1111111111111111

            >>> eval_expr(y ** 1.5)
            9.545941546018392

        Parameters
        ----------
        power : :class:`NumericExpression`
        modulo
            Unsupported argument.

        Returns
        -------
        :class:`Float64Expression`
            Result of raising left to the right power.
        """
        return self._bin_op('**', power, TFloat64())

    def __rpow__(self, other):
        return self._bin_op_reverse('**', other, TFloat64())

    def signum(self):
        """Returns the sign of the callee, ``1`` or ``-1``.

        Examples
        --------
        .. doctest::

            >>> eval_expr(y.signum())
            1

        Returns
        -------
        :py:class:`Int32Expression`
            ``1`` or ``-1``.
        """
        return self._method("signum", TInt32())

    def abs(self):
        """Returns the absolute value of the callee.

        Examples
        --------
        .. doctest::

            >>> eval_expr(y.abs())
            4.5

        Returns
        -------
        :py:class:`.NumericExpression`
            Absolute value of the number.
        """
        return self._method("abs", self._type)

    @typecheck_method(other=expr_numeric)
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

    @typecheck_method(other=expr_numeric)
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
    """Expression of type :class:`hail.expr.types.TString`.

    >>> s = functions.capture('The quick brown fox')
    """
    @typecheck_method(item=oneof(slice, expr_int32))
    def __getitem__(self, item):
        """Slice or index into the string.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s[:15])
            'The quick brown'

            >>> eval_expr(s[0])
            'T'

        Parameters
        ----------
        item : slice or :class:`Int32Expression`
            Slice or character index.

        Returns
        -------
        :class:`StringExpression`
            Substring or character at index `item`.
        """
        if isinstance(item, slice):
            return self._slice(TString(), item.start, item.stop, item.step)
        elif isinstance(item, int) or isinstance(item, Int32Expression):
            return self._index(TString(), item)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        """Concatenate strings.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s + ' jumped over the lazy dog')
            'The quick brown fox jumped over the lazy dog'

        Parameters
        ----------
        other : :class:`StringExpression`
            String to concatenate.

        Returns
        -------
        :class:`StringExpression`
            Concatenated string.
        """
        other = to_expr(other)
        if not isinstance(other, StringExpression):
            raise TypeError("cannot concatenate 'TString' expression and '{}'".format(other._type.__class__))
        return self._bin_op("+", other, TString())

    def __radd__(self, other):
        assert (isinstance(other, StringExpression) or isinstance(other, str))
        return self._bin_op_reverse("+", other, TString())

    def length(self):
        """Returns the length of the string.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s.length())
            19

        Returns
        -------
        :class:`Int32Expression`
            Length of the string.
        """
        return self._method("length", TInt32())

    @typecheck_method(pattern1=expr_str, pattern2=expr_str)
    def replace(self, pattern1, pattern2):
        """Replace substrings matching `pattern1` with `pattern2` using regex.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s.replace(' ', '_'))
            'The_quick_brown_fox'

        Notes
        -----
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

    @typecheck_method(delim=expr_str, n=nullable(expr_int32))
    def split(self, delim, n=None):
        """Returns an array of strings generated by splitting the string at `delim`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(s.split('\\s+'))
            ['The', 'quick', 'brown', 'fox']

            >>> eval_expr(s.split('\\s+', 2))
            ['The', 'quick brown fox']

        Notes
        -----
        The delimiter is a regex using the
        `Java regex syntax <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`_
        delimiter. To split on special characters, escape them with double
        backslash (``\\\\``).

        Parameters
        ----------
        delim : str or :class:`StringExpression`
            Delimiter regex.
        n : :class:`Int32Expression`, optional
            Maximum number of splits.

        Returns
        -------
        :class:`ArrayExpression`
            Array of split strings.
        """
        if n is None:
            return self._method("split", TArray(TString()), delim)
        else:
            return self._method("split", TArray(TString()), delim, n)

    @typecheck_method(regex=strlike)
    def matches(self, regex):
        """Returns ``True`` if the string contains any match for the given regex.

        Examples
        --------

        >>> string = functions.capture('NA12878')

        The `regex` parameter does not need to match the entire string:

        .. doctest::

            >>> eval_expr(string.matches('12'))
            True

        Regex motifs can be used to match sequences of characters:

        .. doctest::

            >>> eval_expr(string.matches(r'NA\\\\d+'))
            True

        Notes
        -----
        The `regex` argument is a
        `regular expression <https://en.wikipedia.org/wiki/Regular_expression>`__,
        and uses
        `Java regex syntax <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`__.

        Parameters
        ----------
        regex: :obj:`str`
            Pattern to match.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the string contains any match for the regex, otherwise ``False``.
        """
        return construct_expr(RegexMatch(self._ast, regex), TBoolean(),
                              self._indices, self._aggregations, self._joins)


class CallExpression(Expression):
    """Expression of type :class:`hail.expr.types.TCall`.

    >>> call = functions.capture(Call(1))
    """
    @property
    def gt(self):
        """Returns the triangle number of :py:meth:`Call.gtj` and :py:meth:`Call.gtk`.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.gt)
            1

        Returns
        -------
        :class:`Int32Expression`
            Triangle number of the two alleles.
        """
        return self._field("gt", TInt32())

    def gtj(self):
        """Returns the allele index of the first allele.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.gtj())
            0

        Returns
        -------
        :class:`Int32Expression`
            First allele index.
        """
        return self._method("gtj", TInt32())

    def gtk(self):
        """Returns the allele index of the second allele.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.gtk())
            1

        Returns
        -------
        :class:`Int32Expression`
            Second allele index.
        """
        return self._method("gtk", TInt32())

    def is_non_ref(self):
        """Evaluate whether the call includes one or more non-reference alleles.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.is_non_ref())
            True

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if at least one allele is non-reference, ``False`` otherwise.
        """
        return self._method("isNonRef", TBoolean())

    def is_het(self):
        """Evaluate whether the call includes two different alleles.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.is_het())
            True

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the two alleles are different, ``False`` if they are the same.
        """
        return self._method("isHet", TBoolean())

    def is_het_nonref(self):
        """Evaluate whether the call includes two different alleles, neither of which is reference.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.is_het_nonref())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the call includes two different alternate alleles, ``False`` otherwise.
        """
        return self._method("isHetNonRef", TBoolean())

    def is_het_ref(self):
        """Evaluate whether the call includes two different alleles, one of which is reference.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.is_het_ref())
            True

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the call includes one reference and one alternate allele, ``False`` otherwise.
        """
        return self._method("isHetRef", TBoolean())

    def is_hom_ref(self):
        """Evaluate whether the call includes two reference alleles.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.is_hom_ref())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the call includes two reference alleles, ``False`` otherwise.
        """
        return self._method("isHomRef", TBoolean())

    def is_hom_var(self):
        """Evaluate whether the call includes two identical alternate alleles.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.is_hom_var())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the call includes two identical alternate alleles, ``False`` otherwise.
        """
        return self._method("isHomVar", TBoolean())

    def num_alt_alleles(self):
        """Returns the number of non-reference alleles (0, 1, or 2).

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.num_alt_alleles())
            1

        Returns
        -------
        :class:`Int32Expression`
            The number of non-reference alleles (0, 1, or 2).
        """
        return self._method("nNonRefAlleles", TInt32())

    @typecheck_method(v=expr_variant)
    def one_hot_alleles(self, v):
        """Returns an array containing the summed one-hot encoding of the two alleles.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.one_hot_alleles(Variant('1', 1, 'A', 'T')))
            [1, 1]

        This one-hot representation is the positional sum of the one-hot
        encoding for each called allele. For a biallelic variant, the one-hot
        encoding for a reference allele is ``[1, 0]`` and the one-hot encoding
        for an alternate allele is ``[0, 1]``. Diploid calls would produce the
        following arrays: ``[2, 0]`` for homozygous reference, ``[1, 1]`` for
        heterozygous, and ``[0, 2]`` for homozygous alternate.

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

    @typecheck_method(v=expr_variant)
    def one_hot_genotype(self, v):
        """Returns the triangle number of the genotype one-hot encoded into an integer array.

        Examples
        --------
        .. doctest::

            >>> eval_expr(call.one_hot_genotype(Variant('1', 1, 'A', 'T')))
            [0, 1, 0]

        A one-hot encoding uses a vector to represent categorical data, like a
        genotype call, by using an array with one ``1`` and many ``0`` s. In
        this case, the array will have a ``1`` at the position of the triangle
        number of the genotype (:py:meth:`Call.gt`).

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
    """Expression of type :class:`hail.expr.types.TLocus`.

    >>> locus = functions.capture(Locus('1', 100))
    """
    @property
    def contig(self):
        """Returns the chromosome.

        Examples
        --------
        .. doctest::

            >>> eval_expr(locus.contig)
            '1'

        Returns
        -------
        :class:`StringExpression`
            The chromosome for this locus.
        """
        return self._field("contig", TString())

    @property
    def position(self):
        """Returns the position along the chromosome.

        Examples
        --------
        .. doctest::

            >>> eval_expr(locus.position)
            100

        Returns
        -------
        :class:`Int32Expression`
            This locus's position along its chromosome.
        """
        return self._field("position", TInt32())


class IntervalExpression(Expression):
    """Expression of type :class:`hail.expr.types.TInterval`.

    >>> interval = functions.capture(Interval.parse('X:1M-2M'))
    """
    @typecheck_method(locus=expr_locus)
    def contains(self, locus):
        """Tests whether a locus is contained in the interval.

        Examples
        --------
        .. doctest::

            >>> eval_expr(interval.contains(Locus('X', 3000000)))
            False

            >>> eval_expr(interval.contains(Locus('X', 1500000)))
            True

        Parameters
        ----------
        locus : :class:`hail.genetics.Locus` or :class:`LocusExpression`
            Locus to test for interval membership.
        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if `locus` is contained in the interval, ``False`` otherwise.
        """
        locus = to_expr(locus)
        if not locus._type._rg == self._type._rg:
            raise TypeError('Reference genome mismatch: {}, {}'.format(self._type._rg, locus._type._rg))
        return self._method("contains", TBoolean(), locus)

    @property
    def end(self):
        """Returns the end locus.

        Examples
        --------
        .. doctest::

            >>> eval_expr(interval.end)
            Locus(contig=X, position=2000000, reference_genome=GRCh37)

        Returns
        -------
        :class:`LocusExpression`
            End locus.
        """
        return self._field("end", TLocus())

    @property
    def start(self):
        """Returns the start locus.

        Examples
        --------
        .. doctest::

            >>> eval_expr(interval.start)
            Locus(contig=X, position=1000000, reference_genome=GRCh37)

        Returns
        -------
        :class:`LocusExpression`
            Start locus.
        """
        return self._field("start", TLocus())


class AltAlleleExpression(Expression):
    """Expression of type :class:`hail.expr.types.TAltAllele`.

    >>> altallele = functions.capture(AltAllele('A', 'AAA'))
    """
    @property
    def alt(self):
        """Returns the alternate allele string.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.alt)
            'AAA'

        Returns
        -------
        :class:`StringExpression`
            Alternate base string.
        """
        return self._field("alt", TString())

    @property
    def ref(self):
        """Returns the reference allele string.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.ref)
            'A'

        Returns
        -------
        :class:`StringExpression`
            Reference base string.
        """
        return self._field("ref", TString())

    def category(self):
        """Returns the string representation of the alternate allele class.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.category())
            'Insertion'

        Returns
        -------
        :class:`StringExpression`
            One of ``"SNP"``, ``"MNP"``, ``"Insertion"``, ``"Deletion"``, ``"Star"``, ``"Complex"``.
        """
        return self._method("category", TString())

    def is_complex(self):
        """Returns ``True`` if the polymorphism is not a SNP, MNP, star, insertion, or deletion.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_complex())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the allele is a complex event.
        """
        return self._method("isComplex", TBoolean())

    def is_deletion(self):
        """Returns ``True`` if the polymorphism is a deletion.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_deletion())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the event is a a deletion.
        """
        return self._method("isDeletion", TBoolean())

    def is_indel(self):
        """Returns ``True`` if the polymorphism is an insertion or deletion.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_insertion())
            True

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the event is an insertion or deletion.
        """
        return self._method("isIndel", TBoolean())

    def is_insertion(self):
        """Returns ``True`` if the polymorphism is an insertion.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_insertion())
            True

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the event is an insertion.
        """
        return self._method("isInsertion", TBoolean())

    def is_mnp(self):
        """Returns ``True`` if the polymorphism is a MNP.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_mnp())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the event is a multiple nucleotide polymorphism.
        """
        return self._method("isMNP", TBoolean())

    def is_snp(self):
        """Returns ``True`` if the polymorphism is a SNP.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_snp())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the event is a single nucleotide polymorphism.
        """
        return self._method("isSNP", TBoolean())

    def is_star(self):
        """Returns ``True`` if the polymorphism is a star (upstream deletion) allele.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_star())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the event is a star (upstream deletion) allele.
        """
        return self._method("isStar", TBoolean())

    def is_transition(self):
        """Returns ``True`` if the polymorphism is a transition SNP.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_transition())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the event is a SNP and a transition.
        """
        return self._method("isTransition", TBoolean())

    def is_transversion(self):
        """Returns ``True`` if the polymorphism is a transversion SNP.

        Examples
        --------
        .. doctest::

            >>> eval_expr(altallele.is_transversion())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the event is a SNP and a transversion.
        """
        return self._method("isTransversion", TBoolean())


class VariantExpression(Expression):
    """Expression of type :class:`hail.expr.types.TVariant`.

    >>> variant = functions.capture(Variant('16', 123055, 'A', 'C'))
    """
    def alt(self):
        """Returns the alternate allele string.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.alt())
            'C'

        Warning
        -------
        Assumes biallelic. If the variant is multiallelic, throws an error.

        Returns
        -------
        :class:`StringExpression`
            Alternate allele base string.
        """
        return self._method("alt", TString())

    def alt_allele(self):
        """Returns the alternate allele.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.alt_allele())
            AltAllele(ref='A', alt='C')

        Warning
        -------
        Assumes biallelic. If the variant is multiallelic, throws an error.

        Returns
        -------
        :class:`AltAlleleExpression`
            Alternate allele.
        """
        return self._method("altAllele", TAltAllele())

    @property
    def alt_alleles(self):
        """Returns the alternate alleles in the polymorphism.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.alt_alleles)
            [AltAllele(ref='A', alt='C')]

        Returns
        -------
        :class:`ArrayExpression` with element type :class:`AltAlleleExpression`
            Alternate alleles.
        """
        return self._field("altAlleles", TArray(TAltAllele()))

    @property
    def contig(self):
        """Returns the chromosome.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.contig)
            '16'

        Returns
        -------
        :class:`StringExpression`
            Contig.
        """
        return self._field("contig", TString())

    def in_x_nonpar(self):
        """Returns ``True`` if the polymorphism is in a non-pseudoautosomal region of chromosome X.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.in_x_nonpar())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the variant is found in a non-pseudoautosomal region of chromosome X.
        """
        return self._method("inXNonPar", TBoolean())

    def in_x_par(self):
        """Returns ``True`` if the polymorphism is in the pseudoautosomal region of chromosome X.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.in_x_par())
            False

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the variant is found in a pseudoautosomal region of chromosome X.
        """
        return self._method("inXPar", TBoolean())

    def in_y_nonpar(self):
        """Returns ``True`` if the polymorphism is in a non-pseudoautosomal region of chromosome Y.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.in_y_nonpar())
            False

        Note
        ----
        Many variant callers only generate variants on chromosome X for the
        pseudoautosomal region. In this case, all variants mapped to chromosome
        Y are non-pseudoautosomal.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the variant is found in a non-pseudoautosomal region of chromosome Y.
        """
        return self._method("inYNonPar", TBoolean())

    def in_y_par(self):
        """Returns ``True`` if the polymorphism is in a pseudoautosomal region of chromosome Y.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.in_y_par())
            False

        Note
        ----
        Many variant callers only generate variants on chromosome X for the
        pseudoautosomal region. In this case, all variants mapped to chromosome
        Y are non-pseudoautosomal.

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the variant is found in a pseudoautosomal region of chromosome Y.
        """
        return self._method("inYPar", TBoolean())

    def is_autosomal(self):
        """Returns ``True`` if the polymorphism is found on an autosome.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.is_autosomal())
            True

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the variant is found on an autosome.
        """
        return self._method("isAutosomal", TBoolean())

    def is_biallelic(self):
        """Returns ``True`` if there is only one alternate allele.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.is_biallelic())
            True

        Returns
        -------
        :class:`BooleanExpression`
            ``True`` if the variant has only one alternate allele.
        """
        return self._method("isBiallelic", TBoolean())

    def locus(self):
        """Returns the locus at which the polymorphism occurs.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.locus())
            Locus(contig=16, position=123055, reference_genome=GRCh37)

        Returns
        -------
        :class:`LocusExpression`
            Locus associated with this variant.
        """
        return self._method("locus", TLocus())

    def num_alleles(self):
        """Returns the number of alleles in the polymorphism, including the reference.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.num_alleles())
            2

        Returns
        -------
        :class:`Int32Expression`
            Total number of alleles, including the reference.
        """
        return self._method("nAlleles", TInt32())

    def num_alt_alleles(self):
        """Returns the number of alleles in the polymorphism, excluding the reference.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.num_alt_alleles())
            1

        Returns
        -------
        :class:`Int32Expression`
            Total number of non-reference alleles.
        """
        return self._method("nAltAlleles", TInt32())

    def num_genotypes(self):
        """Returns the number of possible genotypes given the number of total alleles.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.num_genotypes())
            3

        Returns
        -------
        :class:`Int32Expression`
            Total number of possible diploid genotype configurations.
        """
        return self._method("nGenotypes", TInt32())

    @property
    def ref(self):
        """Returns the reference allele string.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.ref)
            'A'

        Returns
        -------
        :class:`StringExpression`
            Reference base string.
        """
        return self._field("ref", TString())

    @property
    def start(self):
        """Returns the chromosomal position of the polymorphism.

        Examples
        --------
        .. doctest::

            >>> eval_expr(variant.start)
            123055

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
           aggregation_axes=setof(strlike))
def analyze(expr, expected_indices, aggregation_axes=set()):
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

    for w in warnings:
        warn('Analysis warning: {}'.format(w.msg))
    if errors:
        for e in errors:
            error('Analysis exception: {}'.format(e.msg))
        raise errors[0]


@typecheck(expression=expr_any)
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


@typecheck(expression=expr_any)
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
    analyze(expression, Indices())
    if len(expression._joins) > 0:
        raise ExpressionException("'eval_expr' methods do not support joins or broadcasts")
    r, t = Env.hc().eval_expr_typed(expression._ast.to_hql())
    return r, t


_lazy_int32.set(Int32Expression)
_lazy_numeric.set(NumericExpression)
_lazy_array.set(ArrayExpression)
_lazy_set.set(SetExpression)
_lazy_bool.set(BooleanExpression)
_lazy_struct.set(StructExpression)
_lazy_string.set(StringExpression)
_lazy_variant.set(VariantExpression)
_lazy_locus.set(LocusExpression)
_lazy_altallele.set(AltAlleleExpression)
_lazy_interval.set(IntervalExpression)
_lazy_call.set(CallExpression)
_lazy_expr.set(Expression)
