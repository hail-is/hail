import hail as hl
from hail.expr.expr_ast import *
from hail.expr.expressions import Expression, to_expr, ExpressionException, \
    unify_all, Indices, Aggregation
from hail.expr.expressions.expression_typecheck import *
from hail.expr.types import *
from hail.typecheck import *
from hail.utils.java import *
from hail.utils.linkedlist import LinkedList
from hail.utils.misc import get_nice_field_error, get_nice_attr_error
from hail.genetics.reference_genome import reference_genome_type


class CollectionExpression(Expression):
    """Expression of type :class:`.tarray` or :class:`.tset`

    >>> a = hl.literal([1, 2, 3, 4, 5])

    >>> s = hl.literal({'Alice', 'Bob', 'Charlie'})
    """

    @typecheck_method(f=func_spec(1, expr_bool))
    def any(self, f):
        """Returns ``True`` if `f` returns ``True`` for any element.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.any(lambda x: x % 2 == 0))
            True

            >>> hl.eval_expr(s.any(lambda x: x[0] == 'D'))
            False

        Notes
        -----
        This method always returns ``False`` for empty collections.

        Parameters
        ----------
        f : function ( (arg) -> :class:`.BooleanExpression`)
            Function to evaluate for each element of the collection. Must return a
            :class:`.BooleanExpression`.

        Returns
        -------
        :class:`.BooleanExpression`.
            ``True`` if `f` returns ``True`` for any element, ``False`` otherwise.
        """

        def unify_ret(t):
            if t != tbool:
                raise TypeError("'exists' expects 'f' to return an expression of type 'bool', found '{}'".format(t))
            return t

        return self._bin_lambda_method("exists", f, self._type.element_type, unify_ret)

    @typecheck_method(f=func_spec(1, expr_bool))
    def filter(self, f):
        """Returns a new collection containing elements where `f` returns ``True``.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.filter(lambda x: x % 2 == 0))
            [2, 4]

            >>> hl.eval_expr(s.filter(lambda x: ~(x[-1] == 'e')))
            {'Bob'}

        Notes
        -----
        Returns a same-type expression; evaluated on a :class:`.SetExpression`, returns a
        :class:`.SetExpression`. Evaluated on an :class:`.ArrayExpression`,
        returns an :class:`.ArrayExpression`.

        Parameters
        ----------
        f : function ( (arg) -> :class:`.BooleanExpression`)
            Function to evaluate for each element of the collection. Must return a
            :class:`.BooleanExpression`.

        Returns
        -------
        :class:`.CollectionExpression`
            Expression of the same type as the callee.
        """

        def unify_ret(t):
            if t != tbool:
                raise TypeError("'filter' expects 'f' to return an expression of type 'bool', found '{}'".format(t))
            return self._type

        return self._bin_lambda_method("filter", f, self._type.element_type, unify_ret)

    @typecheck_method(f=func_spec(1, expr_bool))
    def find(self, f):
        """Returns the first element where `f` returns ``True``.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.find(lambda x: x ** 2 > 20))
            5

            >>> hl.eval_expr(s.find(lambda x: x[0] == 'D'))
            None

        Notes
        -----
        If `f` returns ``False`` for every element, then the result is missing.

        Parameters
        ----------
        f : function ( (arg) -> :class:`.BooleanExpression`)
            Function to evaluate for each element of the collection. Must return a
            :class:`.BooleanExpression`.

        Returns
        -------
        :class:`.Expression`
            Expression whose type is the element type of the collection.
        """

        def unify_ret(t):
            if t != tbool:
                raise TypeError("'find' expects 'f' to return an expression of type 'bool', found '{}'".format(t))
            return self._type.element_type

        return self._bin_lambda_method("find", f, self._type.element_type, unify_ret)

    @typecheck_method(f=func_spec(1, expr_any))
    def flatmap(self, f):
        """Map each element of the collection to a new collection, and flatten the results.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.flatmap(lambda x: hl.range(0, x)))
            [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]

            >>> hl.eval_expr(s.flatmap(lambda x: hl.set(hl.range(0, x.length()).map(lambda i: x[i]))))
            {'A', 'B', 'C', 'a', 'b', 'c', 'e', 'h', 'i', 'l', 'o', 'r'}

        Parameters
        ----------
        f : function ( (arg) -> :class:`.CollectionExpression`)
            Function from the element type of the collection to the type of the
            collection. For instance, `flatmap` on a ``set<str>`` should take
            a ``str`` and return a ``set``.

        Returns
        -------
        :class:`.CollectionExpression`
        """
        expected_type, s = (tarray, 'array') if isinstance(self._type, tarray) else (tset, 'set')

        def unify_ret(t):
            if not isinstance(t, expected_type):
                raise TypeError("'flatmap' expects 'f' to return an expression of type '{}', found '{}'".format(s, t))
            return t

        return self._bin_lambda_method("flatMap", f, self._type.element_type, unify_ret)

    @typecheck_method(f=func_spec(1, expr_bool))
    def all(self, f):
        """Returns ``True`` if `f` returns ``True`` for every element.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.all(lambda x: x < 10))
            True

        Notes
        -----
        This method returns ``True`` if the collection is empty.

        Parameters
        ----------
        f : function ( (arg) -> :class:`.BooleanExpression`)
            Function to evaluate for each element of the collection. Must return a
            :class:`.BooleanExpression`.

        Returns
        -------
        :class:`.BooleanExpression`.
            ``True`` if `f` returns ``True`` for every element, ``False`` otherwise.
        """

        def unify_ret(t):
            if t != tbool:
                raise TypeError("'forall' expects 'f' to return an expression of type 'bool', found '{}'".format(t))
            return t

        return self._bin_lambda_method("forall", f, self._type.element_type, unify_ret)

    @typecheck_method(f=func_spec(1, expr_any))
    def group_by(self, f):
        """Group elements into a dict according to a lambda function.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.group_by(lambda x: x % 2 == 0))
            {False: [1, 3, 5], True: [2, 4]}

            >>> hl.eval_expr(s.group_by(lambda x: x.length()))
            {3: {'Bob'}, 5: {'Alice'}, 7: {'Charlie'}}

        Parameters
        ----------
        f : function ( (arg) -> :class:`.Expression`)
            Function to evaluate for each element of the collection to produce a key for the
            resulting dictionary.

        Returns
        -------
        :class:`.DictExpression`.
            Dictionary keyed by results of `f`.
        """
        return self._bin_lambda_method("groupBy", f, self._type.element_type, lambda t: tdict(t, self._type))

    @typecheck_method(f=func_spec(1, expr_any))
    def map(self, f):
        """Transform each element of a collection.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.map(lambda x: x ** 3))
            [1.0, 8.0, 27.0, 64.0, 125.0]

            >>> hl.eval_expr(s.map(lambda x: x.length()))
            {3, 5, 7}

        Parameters
        ----------
        f : function ( (arg) -> :class:`.Expression`)
            Function to transform each element of the collection.

        Returns
        -------
        :class:`.CollectionExpression`.
            Collection where each element has been transformed according to `f`.
        """
        return self._bin_lambda_method("map", f, self._type.element_type, lambda t: self._type.__class__(t))

    def length(self):
        """Returns the size of a collection.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.length())
            5

            >>> hl.eval_expr(s.length())
            3

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
            The number of elements in the collection.
        """
        return self._method("size", tint32)

    def size(self):
        """Returns the size of a collection.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a.size())
            5

            >>> hl.eval_expr(s.size())
            3

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
            The number of elements in the collection.
        """
        return self._method("size", tint32)


class ArrayExpression(CollectionExpression):
    """Expression of type :class:`.tarray`.

    >>> a = hl.literal(['Alice', 'Bob', 'Charlie'])

    See Also
    --------
    :class:`.CollectionExpression`
    """

    def __getitem__(self, item):
        """Index into or slice the array.

        Examples
        --------

        Index with a single integer:

        .. doctest::

            >>> hl.eval_expr(a[1])
            'Bob'

            >>> hl.eval_expr(a[-1])
            'Charlie'

        Slicing is also supported:

        .. doctest::

            >>> hl.eval_expr(a[1:])
            ['Bob', 'Charlie']

        Parameters
        ----------
        item : slice or :class:`.Expression` of type :py:data:`.tint32`
            Index or slice.

        Returns
        -------
        :class:`.Expression`
            Element or array slice.
        """
        if isinstance(item, slice):
            return self._slice(self.dtype, item.start, item.stop, item.step)
        else:
            item = to_expr(item)
            if not item.dtype == tint32:
                raise TypeError("array expects key to be type 'slice' or expression of type 'int32', "
                                "found expression of type '{}'".format(item._type))
            return self._index(self.dtype.element_type, item)

    @typecheck_method(item=expr_any)
    def contains(self, item):
        """Returns a boolean indicating whether `item` is found in the array.

        Examples
        --------

        .. doctest::

            >>> hl.eval_expr(a.contains('Charlie'))
            True

            >>> hl.eval_expr(a.contains('Helen'))
            False

        Parameters
        ----------
        item : :class:`.Expression`
            Item for inclusion test.

        Warning
        -------
        This method takes time proportional to the length of the array. If a
        pipeline uses this method on the same array several times, it may be
        more efficient to convert the array to a set first
        (:func:`~hail.expr.functions.set`).

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the element is found in the array, ``False`` otherwise.
        """
        import hail as hl
        return hl.any(lambda x: x == item, self)

    @typecheck_method(item=expr_any)
    def append(self, item):
        """Append an element to the array and return the result.

        Examples
        --------

        .. doctest::

            >>> hl.eval_expr(a.append('Dan'))
            ['Alice', 'Bob', 'Charlie', 'Dan']

        Note
        ----
        This method does not mutate the caller, but instead returns a new
        array by copying the caller and adding `item`.

        Parameters
        ----------
        item : :class:`.Expression`
            Element to append, same type as the array element type.

        Returns
        -------
        :class:`.ArrayExpression`
        """
        if not item._type == self._type.element_type:
            raise TypeError("'ArrayExpression.append' expects 'item' to be the same type as its elements\n"
                            "    array element type: '{}'\n"
                            "    type of arg 'item': '{}'".format(self._type._element_type, item._type))
        return self._method("append", self._type, item)

    @typecheck_method(a=expr_array())
    def extend(self, a):
        """Concatenate two arrays and return the result.

        Examples
        --------

        .. doctest::

            >>> hl.eval_expr(a.extend(['Dan', 'Edith']))
            ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith']

        Parameters
        ----------
        a : :class:`.ArrayExpression`
            Array to concatenate, same type as the callee.

        Returns
        -------
        :class:`.ArrayExpression`
        """
        if not a._type == self._type:
            raise TypeError("'ArrayExpression.extend' expects 'a' to be the same type as the caller\n"
                            "    caller type: '{}'\n"
                            "    type of 'a': '{}'".format(self._type, a._type))
        return self._method("extend", self._type, a)


class ArrayNumericExpression(ArrayExpression):
    """Expression of type :class:`.tarray` with a numeric type.

    Numeric arrays support arithmetic both with scalar values and other arrays.
    Arithmetic between two numeric arrays requires that the length of each array
    is identical, and will apply the operation positionally (``a1 * a2`` will
    multiply the first element of ``a1`` by the first element of ``a2``, the
    second element of ``a1`` by the second element of ``a2``, and so on).
    Arithmetic with a scalar will apply the operation to each element of the
    array.

    >>> a1 = hl.literal([0, 1, 2, 3, 4, 5])

    >>> a2 = hl.literal([1, -1, 1, -1, 1, -1])

    """

    def __neg__(self):
        """Negate elements of the array.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(-a1)
            [0, -1, -2, -3, -4, -5]

        Returns
        -------
        :class:`.ArrayNumericExpression`
            Array expression of the same type.
        """
        return self * -1

    def __add__(self, other):
        """Positionally add an array or a scalar.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a1 + 5)
            [5, 6, 7, 8, 9, 10]

            >>> hl.eval_expr(a1 + a2)
            [1, 0, 3, 2, 5, 4]

        Parameters
        ----------
        other : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`
            Value or array to add.

        Returns
        -------
        :class:`.ArrayNumericExpression`
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

            >>> hl.eval_expr(a2 - 1)
            [0, -2, 0, -2, 0, -2]

            >>> hl.eval_expr(a1 - a2)
            [-1, 2, 1, 4, 3, 6]

        Parameters
        ----------
        other : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`
            Value or array to subtract.

        Returns
        -------
        :class:`.ArrayNumericExpression`
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

            >>> hl.eval_expr(a2 * 5)
            [5, -5, 5, -5, 5, -5]

            >>> hl.eval_expr(a1 * a2)
            [0, -1, 2, -3, 4, -5]

        Parameters
        ----------
        other : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`
            Value or array to multiply by.

        Returns
        -------
        :class:`.ArrayNumericExpression`
            Array of positional products.
        """
        return self._bin_op_numeric("*", other)

    def __rmul__(self, other):
        return self._bin_op_numeric_reverse("*", other)

    def __truediv__(self, other):
        """Positionally divide by an array or a scalar.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a1 / 10)
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

            >>> hl.eval_expr(a2 / a1)
            [inf, -1.0, 0.5, -0.3333333333333333, 0.25, -0.2]

        Parameters
        ----------
        other : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`
            Value or array to divide by.

        Returns
        -------
        :class:`.ArrayNumericExpression`
            Array of positional quotients.
        """

        def ret_type_f(t):
            assert is_numeric(t)
            if t == tint32 or t == tint64:
                return tfloat32
            else:
                # Float64 or Float32
                return t

        return self._bin_op_numeric("/", other, ret_type_f)

    def __rtruediv__(self, other):
        def ret_type_f(t):
            assert is_numeric(t)
            if t == tint32 or t == tint64:
                return tfloat32
            else:
                # Float64 or Float32
                return t

        return self._bin_op_numeric_reverse("/", other, ret_type_f)

    def __floordiv__(self, other):
        """Positionally divide by an array or a scalar using floor division.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a1 // 2)
            [0, 0, 1, 1, 2, 2]

        Parameters
        ----------
        other : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`

        Returns
        -------
        :class:`.ArrayNumericExpression`
        """
        return self._bin_op_numeric('//', other)

    def __rfloordiv__(self, other):
        return self._bin_op_numeric_reverse('//', other)

    def __mod__(self, other):
        """Positionally compute the left modulo the right.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a1 % 2)
            [0, 1, 0, 1, 0, 1]

        Parameters
        ----------
        other : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`

        Returns
        -------
        :class:`.ArrayNumericExpression`
        """
        return self._bin_op_numeric('%', other)

    def __rmod__(self, other):
        return self._bin_op_numeric_reverse('%', other)

    def __pow__(self, other):
        """Positionally raise to the power of an array or a scalar.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(a1 ** 2)
            [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]

            >>> hl.eval_expr(a1 ** a2)
            [0.0, 1.0, 2.0, 0.3333333333333333, 4.0, 0.2]

        Parameters
        ----------
        other : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`

        Returns
        -------
        :class:`.ArrayNumericExpression`
        """
        return self._bin_op_numeric('**', other, lambda _: tfloat64)

    def __rpow__(self, other):
        return self._bin_op_numeric_reverse('**', other, lambda _: tfloat64)


class SetExpression(CollectionExpression):
    """Expression of type :class:`.tset`.

    >>> s1 = hl.literal({1, 2, 3})
    >>> s2 = hl.literal({1, 3, 5})

    See Also
    --------
    :class:`.CollectionExpression`
    """

    @typecheck_method(item=expr_any)
    def add(self, item):
        """Returns a new set including `item`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s1.add(10))
            {1, 2, 3, 10}

        Parameters
        ----------
        item : :class:`.Expression`
            Value to add.

        Returns
        -------
        :class:`.SetExpression`
            Set with `item` added.
        """
        if not item._type == self._type.element_type:
            raise TypeError("'SetExpression.add' expects 'item' to be the same type as its elements\n"
                            "    set element type:   '{}'\n"
                            "    type of arg 'item': '{}'".format(self._type._element_type, item._type))
        return self._method("add", self._type, item)

    @typecheck_method(item=expr_any)
    def remove(self, item):
        """Returns a new set excluding `item`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s1.remove(1))
            {2, 3}

        Parameters
        ----------
        item : :class:`.Expression`
            Value to remove.

        Returns
        -------
        :class:`.SetExpression`
            Set with `item` removed.
        """
        if not item._type == self._type.element_type:
            raise TypeError("'SetExpression.remove' expects 'item' to be the same type as its elements\n"
                            "    set element type:   '{}'\n"
                            "    type of arg 'item': '{}'".format(self._type._element_type, item._type))
        return self._method("remove", self._type, item)

    @typecheck_method(item=expr_any)
    def contains(self, item):
        """Returns ``True`` if `item` is in the set.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s1.contains(1))
            True

            >>> hl.eval_expr(s1.contains(10))
            False

        Parameters
        ----------
        item : :class:`.Expression`
            Value for inclusion test.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if `item` is in the set.
        """
        if not item._type == self._type.element_type:
            raise TypeError("'SetExpression.contains' expects 'item' to be the same type as its elements\n"
                            "    set element type:   '{}'\n"
                            "    type of arg 'item': '{}'".format(self._type._element_type, item._type))
        return self._method("contains", tbool, item)

    @typecheck_method(s=expr_set())
    def difference(self, s):
        """Return the set of elements in the set that are not present in set `s`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s1.difference(s2))
            {2}

            >>> hl.eval_expr(s2.difference(s1))
            {5}

        Parameters
        ----------
        s : :class:`.SetExpression`
            Set expression of the same type.

        Returns
        -------
        :class:`.SetExpression`
            Set of elements not in `s`.
        """
        if not s._type.element_type == self._type.element_type:
            raise TypeError("'SetExpression.difference' expects 's' to be the same type\n"
                            "    set type:    '{}'\n"
                            "    type of 's': '{}'".format(self._type, s._type))
        return self._method("difference", self._type, s)

    @typecheck_method(s=expr_set())
    def intersection(self, s):
        """Return the intersection of the set and set `s`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s1.intersection(s2))
            {1, 3}

        Parameters
        ----------
        s : :class:`.SetExpression`
            Set expression of the same type.

        Returns
        -------
        :class:`.SetExpression`
            Set of elements present in `s`.
        """
        if not s._type.element_type == self._type.element_type:
            raise TypeError("'SetExpression.intersection' expects 's' to be the same type\n"
                            "    set type:    '{}'\n"
                            "    type of 's': '{}'".format(self._type, s._type))
        return self._method("intersection", self._type, s)

    @typecheck_method(s=expr_set())
    def is_subset(self, s):
        """Returns ``True`` if every element is contained in set `s`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s1.is_subset(s2))
            False

            >>> hl.eval_expr(s1.remove(2).is_subset(s2))
            True

        Parameters
        ----------
        s : :class:`.SetExpression`
            Set expression of the same type.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if every element is contained in set `s`.
        """
        if not s._type.element_type == self._type.element_type:
            raise TypeError("'SetExpression.is_subset' expects 's' to be the same type\n"
                            "    set type:    '{}'\n"
                            "    type of 's': '{}'".format(self._type, s._type))
        return self._method("isSubset", tbool, s)

    @typecheck_method(s=expr_set())
    def union(self, s):
        """Return the union of the set and set `s`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s1.union(s2))
            {1, 2, 3, 5}

        Parameters
        ----------
        s : :class:`.SetExpression`
            Set expression of the same type.

        Returns
        -------
        :class:`.SetExpression`
            Set of elements present in either set.
        """
        if not s._type.element_type == self._type.element_type:
            raise TypeError("'SetExpression.union' expects 's' to be the same type\n"
                            "    set type:    '{}'\n"
                            "    type of 's': '{}'".format(self._type, s._type))
        return self._method("union", self._type, s)


class DictExpression(Expression):
    """Expression of type :class:`.tdict`.

    >>> d = hl.literal({'Alice': 43, 'Bob': 33, 'Charles': 44})
    """

    @typecheck_method(item=expr_any)
    def __getitem__(self, item):
        """Get the value associated with key `item`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(d['Alice'])
            43

        Notes
        -----
        Raises an error if `item` is not a key of the dictionary. Use
        :meth:`.DictExpression.get` to return missing instead of an error.

        Parameters
        ----------
        item : :class:`.Expression`
            Key expression.

        Returns
        -------
        :class:`.Expression`
            Value associated with key `item`.
        """
        if not item._type == self._type.key_type:
            raise TypeError("dict encountered an invalid key type\n"
                            "    dict key type:  '{}'\n"
                            "    type of 'item': '{}'".format(self._type.key_type, item._type))
        return self._index(self.dtype.value_type, item)

    @typecheck_method(item=expr_any)
    def contains(self, item):
        """Returns whether a given key is present in the dictionary.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(d.contains('Alice'))
            True

            >>> hl.eval_expr(d.contains('Anne'))
            False

        Parameters
        ----------
        item : :class:`.Expression`
            Key to test for inclusion.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if `item` is a key of the dictionary, ``False`` otherwise.
        """
        if not item._type == self._type.key_type:
            raise TypeError("'DictExpression.contains' encountered an invalid key type\n"
                            "    dict key type:  '{}'\n"
                            "    type of 'item': '{}'".format(self._type.key_type, item._type))
        return self._method("contains", tbool, item)

    @typecheck_method(item=expr_any, default=nullable(expr_any))
    def get(self, item, default=None):
        """Returns the value associated with key `k` or a default value if that key is not present.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(d.get('Alice'))
            43

            >>> hl.eval_expr(d.get('Anne'))
            None

            >>> hl.eval_expr(d.get('Anne', 0))
            0

        Parameters
        ----------
        item : :class:`.Expression`
            Key.
        default : :class:`.Expression`
            Default value. Must be same type as dictionary values.

        Returns
        -------
        :class:`.Expression`
            The value associated with `item`, or `default`.
        """
        if not item.dtype == self.dtype.key_type:
            raise TypeError("'DictExpression.get' encountered an invalid key type\n"
                            "    dict key type:  '{}'\n"
                            "    type of 'item': '{}'".format(self.dtype.key_type, item._type))
        if default is not None:
            if not self.dtype.value_type == default.dtype:
                raise TypeError("'get' expects parameter 'default' to have the "
                                "same type as the dictionary value type, found '{}' and '{}'"
                                .format(self.dtype, default.dtype))
            return self._method("get", self.dtype.value_type, item, default)
        else:
            return self._method("get", self.dtype.value_type, item)

    def key_set(self):
        """Returns the set of keys in the dictionary.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(d.key_set())
            {'Alice', 'Bob', 'Charles'}

        Returns
        -------
        :class:`.SetExpression`
            Set of all keys.
        """
        return self._method("keySet", tset(self.dtype.key_type))

    def keys(self):
        """Returns an array with all keys in the dictionary.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(d.keys())
            ['Bob', 'Charles', 'Alice']

        Returns
        -------
        :class:`.ArrayExpression`
            Array of all keys.
        """
        return self._method("keys", tarray(self.dtype.key_type))

    @typecheck_method(f=func_spec(1, expr_any))
    def map_values(self, f):
        """Transform values of the dictionary according to a function.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(d.map_values(lambda x: x * 10))
            {'Alice': 430, 'Bob': 330, 'Charles': 440}

        Parameters
        ----------
        f : function ( (arg) -> :class:`.Expression`)
            Function to apply to each value.

        Returns
        -------
        :class:`.DictExpression`
            Dictionary with transformed values.
        """
        return self._bin_lambda_method("mapValues", f, self.dtype.value_type, lambda t: tdict(self.dtype.key_type, t))

    def size(self):
        """Returns the size of the dictionary.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(d.size())
            3

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
            Size of the dictionary.
        """
        return self._method("size", tint32)

    def values(self):
        """Returns an array with all values in the dictionary.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(d.values())
            [33, 44, 43]

        Returns
        -------
        :class:`.ArrayExpression`
            All values in the dictionary.
        """
        return self._method("values", tarray(self.dtype.value_type))


class StructExpression(Mapping, Expression):
    """Expression of type :class:`.tstruct`.

    >>> s = hl.struct(a=5, b='Foo')

    Struct fields are accessible as attributes and keys. It is therefore
    possible to access field `a` of struct `s` with dot syntax:

    .. doctest::

        >>> hl.eval_expr(s.a)
        5

    However, it is recommended to use square brackets to select fields:

    .. doctest::

        >>> hl.eval_expr(s['a'])
        5

    The latter syntax is safer, because fields that share their name with
    an existing attribute of :class:`.StructExpression` (`keys`, `values`,
    `annotate`, `drop`, etc.) will only be accessible using the
    :meth:`.StructExpression.__getitem__` syntax. This is also the only way
    to access fields that are not valid Python identifiers, like fields with
    spaces or symbols.
    """

    @classmethod
    def _from_fields(cls, fields: Dict[str, Expression]):
        t = tstruct(**{k: v.dtype for k, v in fields.items()})
        ast = StructDeclaration(list(fields), list(expr._ast for expr in fields.values()))
        indices, aggregations = unify_all(*fields.values())
        s = StructExpression.__new__(cls)
        s._fields = {}
        for k, v in fields.items():
            s._set_field(k, v)
        super(StructExpression, s).__init__(ast, t, indices, aggregations)
        return s

    @typecheck_method(ast=AST, type=HailType, indices=Indices, aggregations=LinkedList)
    def __init__(self, ast, type, indices=Indices(), aggregations=LinkedList(Aggregation)):
        super(StructExpression, self).__init__(ast, type, indices, aggregations)
        self._fields: Dict[str, Expression] = {}

        for i, (f, t) in enumerate(self.dtype.items()):
            if isinstance(self._ast, StructDeclaration):
                expr = construct_expr(self._ast.values[i], t, self._indices,
                                      self._aggregations)
            else:
                expr = construct_expr(Select(self._ast, f), t, self._indices,
                                      self._aggregations)
            self._set_field(f, expr)

    def _set_field(self, key, value):
        self._fields[key] = value
        if key not in self.__dict__:
            self.__dict__[key] = value

    def _get_field(self, item):
        if item in self._fields:
            return self._fields[item]
        else:
            raise KeyError(get_nice_field_error(self, item))

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise AttributeError(get_nice_attr_error(self, item))

    def __len__(self):
        return len(self._fields)

    @typecheck_method(item=oneof(str, int))
    def __getitem__(self, item):
        """Access a field of the struct by name or index.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s['a'])
            5

            >>> hl.eval_expr(s[1])
            'Foo'

        Parameters
        ----------
        item : :obj:`str`
            Field name.

        Returns
        -------
        :class:`.Expression`
            Struct field.
        """
        if isinstance(item, str):
            return self._get_field(item)
        else:
            return self._get_field(self.dtype.fields[item])

    def __iter__(self):
        return iter(self._fields)

    def __hash__(self):
        return object.__hash__(self)

    def __eq__(self, other):
        return Expression.__eq__(self, other)

    def __ne__(self, other):
        return Expression.__ne__(self, other)

    def __nonzero__(self):
        return Expression.__nonzero__(self)

    @typecheck_method(named_exprs=expr_any)
    def annotate(self, **named_exprs):
        """Add new fields or recompute existing fields.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s.annotate(a=10, c=2*2*2))
            Struct(a=10, b='Foo', c=8)

        Notes
        -----
        If an expression in `named_exprs` shares a name with a field of the
        struct, then that field will be replaced but keep its position in
        the struct. New fields will be appended to the end of the struct.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Fields to add.

        Returns
        -------
        :class:`.StructExpression`
            Struct with new or updated fields.
        """
        names = []
        types = []
        for f, t in self.dtype.items():
            names.append(f)
            types.append(t)
        kwargs_struct = hl.struct(**named_exprs)

        for f, t in kwargs_struct.dtype.items():
            if not f in self._fields:
                names.append(f)
                types.append(t)

        result_type = tstruct(**dict(zip(names, types)))
        indices, aggregations = unify_all(kwargs_struct)

        return construct_expr(ApplyMethod('annotate', self._ast, kwargs_struct._ast), result_type,
                              indices, aggregations)

    @typecheck_method(fields=str, named_exprs=expr_any)
    def select(self, *fields, **named_exprs):
        """Select existing fields and compute new ones.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s.select('a', c=['bar', 'baz']))
            Struct(a=5, c=[u'bar', u'baz'])

        Notes
        -----
        The `fields` argument is a list of field names to keep. These fields
        will appear in the resulting struct in the order they appear in
        `fields`.

        The `named_exprs` arguments are new field expressions.

        Parameters
        ----------
        fields : varargs of :obj:`str`
            Field names to keep.
        named_exprs : keyword args of :class:`.Expression`
            New field expressions.

        Returns
        -------
        :class:`.StructExpression`
            Struct containing specified existing fields and computed fields.
        """
        names = []
        name_set = set()
        types = []
        for a in fields:
            if not a in self._fields:
                raise KeyError("Struct has no field '{}'\n"
                               "    Fields: [ {} ]".format(a, ', '.join("'{}'".format(x) for x in self._fields)))
            if a in name_set:
                raise ExpressionException("'StructExpression.select' does not support duplicate identifiers.\n"
                                          "    Identifier '{}' appeared more than once".format(a))
            names.append(a)
            name_set.add(a)
            types.append(self[a].dtype)
        select_names = names[:]
        select_name_set = set(select_names)

        kwargs_struct = hl.struct(**named_exprs)
        for f, t in kwargs_struct.dtype.items():
            if f in select_name_set:
                raise ExpressionException("Cannot select and assign '{}' in the same 'select' call".format(f))
            names.append(f)
            types.append(t)
        result_type = tstruct(**dict(zip(names, types)))

        indices, aggregations = unify_all(self, kwargs_struct)

        return construct_expr(ApplyMethod('merge', StructOp('select', self._ast, *select_names), kwargs_struct._ast),
                              result_type, indices, aggregations)

    @typecheck_method(fields=str)
    def drop(self, *fields):
        """Drop fields from the struct.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s.drop('b'))
            Struct(a=5)

        Parameters
        ----------
        fields: varargs of :obj:`str`
            Fields to drop.

        Returns
        -------
        :class:`.StructExpression`
            Struct without certain fields.
        """
        to_drop = set()
        for a in fields:
            if not a in self._fields:
                raise KeyError("Struct has no field '{}'\n"
                               "    Fields: [ {} ]".format(a, ', '.join("'{}'".format(x) for x in self._fields)))
            if a in to_drop:
                warn("Found duplicate field name in 'StructExpression.drop': '{}'".format(a))
            to_drop.add(a)

        names = []
        types = []
        for f, t in self.dtype.items():
            if not f in to_drop:
                names.append(f)
                types.append(t)
        result_type = tstruct(**dict(zip(names, types)))
        return construct_expr(StructOp('drop', self._ast, *to_drop), result_type,
                              self._indices, self._aggregations)


class TupleExpression(Expression, Sequence):
    """Expression of type :class:`.ttuple`.

    >>> t = hl.literal(("a", 1, [1, 2, 3]))
    """

    @typecheck_method(item=int)
    def __getitem__(self, item):
        """Index into the tuple.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(t[1])
            1

        Parameters
        ----------
        item : :obj:`int`
            Element index.

        Returns
        -------
        :class:`.Expression`
        """
        if not 0 <= item < len(self):
            raise IndexError("Out of bounds index. Tuple length is {}.".format(len(self)))
        return self._index(self.dtype.types[item], item)

    def __len__(self):
        """Returns the length of the tuple.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(len(t))
            3

        Returns
        -------
        :obj:`int`
        """
        return len(self.dtype.types)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class NumericExpression(Expression):
    """Expression of numeric type.

    >>> x = hl.literal(3)

    >>> y = hl.literal(4.5)
    """

    @typecheck_method(other=expr_numeric)
    def __lt__(self, other):
        """Less-than comparison.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(x < 5)
            True

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Right side for comparison.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the left side is smaller than the right side.
        """
        return self._bin_op_numeric("<", other, lambda _: tbool)

    @typecheck_method(other=expr_numeric)
    def __le__(self, other):
        """Less-than-or-equals comparison.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(x <= 3)
            True

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Right side for comparison.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the left side is smaller than or equal to the right side.
        """
        return self._bin_op_numeric("<=", other, lambda _: tbool)

    @typecheck_method(other=expr_numeric)
    def __gt__(self, other):
        """Greater-than comparison.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(y > 4)
            True

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Right side for comparison.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the left side is greater than the right side.
        """
        return self._bin_op_numeric(">", other, lambda _: tbool)

    @typecheck_method(other=expr_numeric)
    def __ge__(self, other):
        """Greater-than-or-equals comparison.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(y >= 4)
            True

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Right side for comparison.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the left side is greater than or equal to the right side.
        """
        return self._bin_op_numeric(">=", other, lambda _: tbool)

    def __pos__(self):
        return self

    def __neg__(self):
        """Negate the number (multiply by -1).

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(-x)
            -3

        Returns
        -------
        :class:`.NumericExpression`
            Negated number.
        """

        return expr_numeric.coerce(self)._unary_op("-")

    def __add__(self, other):
        """Add two numbers.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(x + 2)
            5

            >>> hl.eval_expr(x + y)
            7.5

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Number to add.

        Returns
        -------
        :class:`.NumericExpression`
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

            >>> hl.eval_expr(x - 2)
            1

            >>> hl.eval_expr(x - y)
            -1.5

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Number to subtract.

        Returns
        -------
        :class:`.NumericExpression`
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

            >>> hl.eval_expr(x * 2)
            6

            >>> hl.eval_expr(x * y)
            9.0

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Number to multiply.

        Returns
        -------
        :class:`.NumericExpression`
            Product of the two numbers.
        """
        return self._bin_op_numeric("*", other)

    def __rmul__(self, other):
        return self._bin_op_numeric_reverse("*", other)

    def __truediv__(self, other):
        """Divide two numbers.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(x / 2)
            1.5

            >>> hl.eval_expr(y / 0.1)
            45.0

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Dividend.

        Returns
        -------
        :class:`.NumericExpression`
            The left number divided by the left.
        """

        def ret_type_f(t):
            assert is_numeric(t)
            if t == tint32 or t == tint64:
                return tfloat32
            else:
                # Float64 or Float32
                return t

        return self._bin_op_numeric("/", other, ret_type_f)

    def __rtruediv__(self, other):
        def ret_type_f(t):
            assert is_numeric(t)
            if t == tint32 or t == tint64:
                return tfloat32
            else:
                # float64 or float32
                return t

        return self._bin_op_numeric_reverse("/", other, ret_type_f)

    def __floordiv__(self, other):
        """Divide two numbers with floor division.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(x // 2)
            1

            >>> hl.eval_expr(y // 2)
            2.0

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Dividend.

        Returns
        -------
        :class:`.NumericExpression`
            The floor of the left number divided by the right.
        """
        return self._bin_op_numeric('//', other)

    def __rfloordiv__(self, other):
        return self._bin_op_numeric_reverse('//', other)

    def __mod__(self, other):
        """Compute the left modulo the right number.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(32 % x)
            2

            >>> hl.eval_expr(7 % y)
            2.5

        Parameters
        ----------
        other : :class:`.NumericExpression`
            Dividend.

        Returns
        -------
        :class:`.NumericExpression`
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

            >>> hl.eval_expr(x ** 2)
            9.0

            >>> hl.eval_expr(x ** -2)
            0.1111111111111111

            >>> hl.eval_expr(y ** 1.5)
            9.545941546018392

        Parameters
        ----------
        power : :class:`.NumericExpression`
        modulo
            Unsupported argument.

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tfloat64`
            Result of raising left to the right power.
        """
        return self._bin_op_numeric('**', power, lambda _: tfloat64)

    def __rpow__(self, other):
        return self._bin_op_numeric_reverse('**', other, lambda _: tfloat64)


class BooleanExpression(NumericExpression):
    """Expression of type :py:data:`.tbool`.

    >>> t = hl.literal(True)
    >>> f = hl.literal(False)
    >>> na = hl.null(hl.tbool)

    .. doctest::

        >>> hl.eval_expr(t)
        True

        >>> hl.eval_expr(f)
        False

        >>> hl.eval_expr(na)
        None

    """

    def _bin_op_logical(self, name, other):
        other = to_expr(other)
        return self._bin_op(name, other, tbool)

    @typecheck_method(other=expr_bool)
    def __rand__(self, other):
        return self.__and__(other)

    @typecheck_method(other=expr_bool)
    def __ror__(self, other):
        return self.__or__(other)

    @typecheck_method(other=expr_bool)
    def __and__(self, other):
        """Return ``True`` if the left and right arguments are ``True``.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(t & f)
            False

            >>> hl.eval_expr(t & na)
            None

            >>> hl.eval_expr(f & na)
            False

        The ``&`` and ``|`` operators have higher priority than comparison
        operators like ``==``, ``<``, or ``>``. Parentheses are often
        necessary:

        .. doctest::

            >>> x = hl.literal(5)

            >>> hl.eval_expr((x < 10) & (x > 2))
            True

        Parameters
        ----------
        other : :class:`.BooleanExpression`
            Right-side operand.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if both left and right are ``True``.
        """
        return self._bin_op_logical("&&", other)

    @typecheck_method(other=expr_bool)
    def __or__(self, other):
        """Return ``True`` if at least one of the left and right arguments is ``True``.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(t | f)
            True

            >>> hl.eval_expr(t | na)
            True

            >>> hl.eval_expr(f | na)
            None

        The ``&`` and ``|`` operators have higher priority than comparison
        operators like ``==``, ``<``, or ``>``. Parentheses are often
        necessary:

        .. doctest::

            >>> x = hl.literal(5)

            >>> hl.eval_expr((x < 10) | (x > 20))
            True

        Parameters
        ----------
        other : :class:`.BooleanExpression`
            Right-side operand.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if either left or right is ``True``.
        """
        return self._bin_op_logical("||", other)

    def __invert__(self):
        """Return the boolean negation.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(~t)
            False

            >>> hl.eval_expr(~f)
            True

            >>> hl.eval_expr(~na)
            None

        Returns
        -------
        :class:`.BooleanExpression`
            Boolean negation.
        """
        return self._unary_op("!")


class Float64Expression(NumericExpression):
    """Expression of type :py:data:`.tfloat64`."""
    pass


class Float32Expression(NumericExpression):
    """Expression of type :py:data:`.tfloat32`."""
    pass


class Int32Expression(NumericExpression):
    """Expression of type :py:data:`.tint32`."""
    pass


class Int64Expression(NumericExpression):
    """Expression of type :py:data:`.tint64`."""
    pass


class StringExpression(Expression):
    """Expression of type :py:data:`.tstr`.

    >>> s = hl.literal('The quick brown fox')
    """

    def __getitem__(self, item):
        """Slice or index into the string.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s[:15])
            'The quick brown'

            >>> hl.eval_expr(s[0])
            'T'

        Parameters
        ----------
        item : slice or :class:`.Expression` of type :py:data:`.tint32`
            Slice or character index.

        Returns
        -------
        :class:`.StringExpression`
            Substring or character at index `item`.
        """
        if isinstance(item, slice):
            return self._slice(tstr, item.start, item.stop, item.step)
        else:
            item = to_expr(item)
            if not item.dtype == tint32:
                raise TypeError("String expects index to be type 'slice' or expression of type 'int32', "
                                "found expression of type '{}'".format(item.dtype))
            return self._index(tstr, item)

    def __add__(self, other):
        """Concatenate strings.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s + ' jumped over the lazy dog')
            'The quick brown fox jumped over the lazy dog'

        Parameters
        ----------
        other : :class:`.StringExpression`
            String to concatenate.

        Returns
        -------
        :class:`.StringExpression`
            Concatenated string.
        """
        other = to_expr(other)
        if not other.dtype == tstr:
            raise NotImplementedError("'{}' + '{}'".format(self.dtype, other.dtype))
        return self._bin_op("+", other, self.dtype)

    def __radd__(self, other):
        other = to_expr(other)
        if not other.dtype == tstr:
            raise NotImplementedError("'{}' + '{}'".format(other.dtype, self.dtype))
        return self._bin_op_reverse("+", other, self.dtype)

    def length(self):
        """Returns the length of the string.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s.length())
            19

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
            Length of the string.
        """
        return self._method("length", tint32)

    @typecheck_method(pattern1=expr_str, pattern2=expr_str)
    def replace(self, pattern1, pattern2):
        """Replace substrings matching `pattern1` with `pattern2` using regex.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s.replace(' ', '_'))
            'The_quick_brown_fox'

        Notes
        -----
        The regex expressions used should follow
        `Java regex syntax <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`_

        Parameters
        ----------
        pattern1 : str or :class:`.StringExpression`
        pattern2 : str or :class:`.StringExpression`

        Returns
        -------

        """
        return self._method("replace", tstr, pattern1, pattern2)

    @typecheck_method(delim=expr_str, n=nullable(expr_int32))
    def split(self, delim, n=None):
        """Returns an array of strings generated by splitting the string at `delim`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(s.split('\\s+'))
            ['The', 'quick', 'brown', 'fox']

            >>> hl.eval_expr(s.split('\\s+', 2))
            ['The', 'quick brown fox']

        Notes
        -----
        The delimiter is a regex using the
        `Java regex syntax <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`_
        delimiter. To split on special characters, escape them with double
        backslash (``\\\\``).

        Parameters
        ----------
        delim : str or :class:`.StringExpression`
            Delimiter regex.
        n : :class:`.Expression` of type :py:data:`.tint32`, optional
            Maximum number of splits.

        Returns
        -------
        :class:`.ArrayExpression`
            Array of split strings.
        """
        if n is None:
            return self._method("split", tarray(tstr), delim)
        else:
            return self._method("split", tarray(tstr), delim, n)

    def lower(self):
        """Returns a copy of the string, but with upper case letters converted
        to lower case.

        Examples
        --------
        >>> s.lower().value
        'the quick brown fox'

        Returns
        -------
        :class:`.StringExpression`
        """
        return self._method("lower", tstr)

    def upper(self):
        """Returns a copy of the string, but with lower case letters converted
        to upper case.

        Examples
        --------
        >>> s.upper().value
        'THE QUICK BROWN FOX'

        Returns
        -------
        :class:`.StringExpression`
        """
        return self._method("upper", tstr)

    def strip(self):
        r"""Returns a copy of the string with whitespace removed from the start
        and end.

        Examples
        --------
        >>> s2 = hl.str('  once upon a time\n')
        >>> s2.strip().value
        'once upon a time'

        Returns
        -------
        :class:`.StringExpression`
        """
        return self._method("strip", tstr)

    @typecheck_method(substr=expr_str)
    def contains(self, substr):
        """Returns whether `substr` is contained in the string.

        Examples
        --------
        >>> s.contains('fox').value
        True

        >>> s.contains('dog').value
        False

        Note
        ----
        This method is case-sensitive.

        Parameters
        ----------
        substr : :class:`.StringExpression`

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("contains", tbool, substr)

    @typecheck_method(substr=expr_str)
    def startswith(self, substr):
        """Returns whether `substr` is a prefix of the string.

        Examples
        --------
        >>> s.startswith('The').value
        True

        >>> s.startswith('the').value
        False

        Note
        ----
        This method is case-sensitive.

        Parameters
        ----------
        substr : :class:`.StringExpression`

        Returns
        -------
        :class:`.StringExpression`
        """
        return self._method('startswith', tbool, substr)


    @typecheck_method(substr=expr_str)
    def endswith(self, substr):
        """Returns whether `substr` is a suffix of the string.

        Examples
        --------
        >>> s.endswith('dog').value
        True

        Note
        ----
        This method is case-sensitive.

        Parameters
        ----------
        substr : :class:`.StringExpression`

        Returns
        -------
        :class:`.StringExpression`
        """
        return self._method('endswith', tbool, substr)

    @typecheck_method(regex=str)
    def first_match_in(self, regex):
        """Returns an array containing the capture groups of the first match of
        `regex` in the given character sequence.

        Examples
        --------
        >>> s.first_match_in("The quick (\\w+) fox").value
        ["brown"]

        >>> s.first_match_in("The (\\w+) (\\w+) (\\w+)").value
        ["quick", "brown", "fox"]

        >>> s.first_match_in("(\\w+) (\\w+)").value
        None

        Parameters
        ----------
        regex : :class:`.StringExpression`

        Returns
        -------
        :class:`.ArrayExpression` with element type :py:data:`.tstr`
        """
        return self._method('firstMatchIn', tarray(tstr), regex)

    @typecheck_method(regex=str)
    def matches(self, regex):
        """Returns ``True`` if the string contains any match for the given regex.

        Examples
        --------

        >>> string = hl.literal('NA12878')

        The `regex` parameter does not need to match the entire string:

        .. doctest::

            >>> hl.eval_expr(string.matches('12'))
            True

        Regex motifs can be used to match sequences of characters:

        .. doctest::

            >>> hl.eval_expr(string.matches(r'NA\\\\d+'))
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
        :class:`.BooleanExpression`
            ``True`` if the string contains any match for the regex, otherwise ``False``.
        """
        return construct_expr(RegexMatch(self._ast, regex), tbool,
                              self._indices, self._aggregations)


class CallExpression(Expression):
    """Expression of type :py:data:`.tcall`.

    >>> call = hl.call(0, 1, phased=False)
    """

    def __getitem__(self, item):
        """Get the i*th* allele.

        Examples
        --------

        Index with a single integer:

        .. doctest::

            >>> hl.eval_expr(call[0])
            0

            >>> hl.eval_expr(call[1])
            1

        Parameters
        ----------
        item : int or :class:`.Expression` of type :py:data:`.tint32`
            Allele index.

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
        """
        if isinstance(item, slice):
            raise NotImplementedError("CallExpression does not support indexing with a slice.")
        else:
            item = to_expr(item)
            if not item.dtype == tint32:
                raise TypeError("Call expects allele index to be an expression of type 'int32', "
                                "found expression of type '{}'".format(item.dtype))
            return self._index(tint32, item)

    @property
    def ploidy(self):
        """Return the number of alleles of this call.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.ploidy)
            2

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
        """
        return self._method("ploidy", tint32)

    @property
    def phased(self):
        """True if the call is phased.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.phased)
            False

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("isPhased", tbool)

    def is_haploid(self):
        """True if the call has ploidy equal to 1.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.is_haploid())
            False

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self.ploidy == 1

    def is_diploid(self):
        """True if the call has ploidy equal to 2.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.is_diploid())
            True

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self.ploidy == 2

    def is_non_ref(self):
        """Evaluate whether the call includes one or more non-reference alleles.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.is_non_ref())
            True

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if at least one allele is non-reference, ``False`` otherwise.
        """
        return self._method("isNonRef", tbool)

    def is_het(self):
        """Evaluate whether the call includes two different alleles.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.is_het())
            True

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the two alleles are different, ``False`` if they are the same.
        """
        return self._method("isHet", tbool)

    def is_het_nonref(self):
        """Evaluate whether the call includes two different alleles, neither of which is reference.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.is_het_nonref())
            False

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the call includes two different alternate alleles, ``False`` otherwise.
        """
        return self._method("isHetNonRef", tbool)

    def is_het_ref(self):
        """Evaluate whether the call includes two different alleles, one of which is reference.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.is_het_ref())
            True

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the call includes one reference and one alternate allele, ``False`` otherwise.
        """
        return self._method("isHetRef", tbool)

    def is_hom_ref(self):
        """Evaluate whether the call includes two reference alleles.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.is_hom_ref())
            False

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the call includes two reference alleles, ``False`` otherwise.
        """
        return self._method("isHomRef", tbool)

    def is_hom_var(self):
        """Evaluate whether the call includes two identical alternate alleles.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.is_hom_var())
            False

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the call includes two identical alternate alleles, ``False`` otherwise.
        """
        return self._method("isHomVar", tbool)

    def n_alt_alleles(self):
        """Returns the number of non-reference alleles.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.n_alt_alleles())
            1

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
            The number of non-reference alleles.
        """
        return self._method("nNonRefAlleles", tint32)

    @typecheck_method(alleles=expr_array(expr_str))
    def one_hot_alleles(self, alleles):
        """Returns an array containing the summed one-hot encoding of the
        alleles.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.one_hot_alleles(['A', 'T']))
            [1, 1]

        This one-hot representation is the positional sum of the one-hot
        encoding for each called allele. For a biallelic variant, the one-hot
        encoding for a reference allele is ``[1, 0]`` and the one-hot encoding
        for an alternate allele is ``[0, 1]``. Diploid calls would produce the
        following arrays: ``[2, 0]`` for homozygous reference, ``[1, 1]`` for
        heterozygous, and ``[0, 2]`` for homozygous alternate.

        Parameters
        ----------
        alleles: :class:`.ArrayStringExpression`
            Variant alleles.

        Returns
        -------
        :class:`.ArrayInt32Expression`
            An array of summed one-hot encodings of allele indices.
        """
        return self._method("oneHotAlleles", tarray(tint32), alleles)

    def unphased_diploid_gt_index(self):
        """Return the genotype index for unphased, diploid calls.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(call.unphased_diploid_gt_index())
            1

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
        """
        return self._method("unphasedDiploidGtIndex", tint32)


class LocusExpression(Expression):
    """Expression of type :class:`.tlocus`.

    >>> locus = hl.locus('1', 1034245)
    """

    @property
    def contig(self):
        """Returns the chromosome.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.contig)
            '1'

        Returns
        -------
        :class:`.StringExpression`
            The chromosome for this locus.
        """
        return self._field("contig", tstr)

    @property
    def position(self):
        """Returns the position along the chromosome.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.position)
            1034245

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
            This locus's position along its chromosome.
        """
        return self._field("position", tint32)

    def in_x_nonpar(self):
        """Returns ``True`` if the locus is in a non-pseudoautosomal
        region of chromosome X.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.in_x_nonpar())
            False

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("inXNonPar", tbool)

    def in_x_par(self):
        """Returns ``True`` if the locus is in a pseudoautosomal region
        of chromosome X.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.in_x_par())
            False

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("inXPar", tbool)

    def in_y_nonpar(self):
        """Returns ``True`` if the locus is in a non-pseudoautosomal
        region of chromosome Y.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.in_y_nonpar())
            False

        Note
        ----
        Many variant callers only generate variants on chromosome X for the
        pseudoautosomal region. In this case, all loci mapped to chromosome
        Y are non-pseudoautosomal.

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("inYNonPar", tbool)

    def in_y_par(self):
        """Returns ``True`` if the locus is in a pseudoautosomal region
        of chromosome Y.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.in_y_par())
            False

        Note
        ----
        Many variant callers only generate variants on chromosome X for the
        pseudoautosomal region. In this case, all loci mapped to chromosome
        Y are non-pseudoautosomal.

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("inYPar", tbool)

    def in_autosome(self):
        """Returns ``True`` if the locus is on an autosome.

        Notes
        -----
        All contigs are considered autosomal except those
        designated as X, Y, or MT by :class:`.ReferenceGenome`.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.in_autosome())
            True

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("isAutosomal", tbool)

    def in_autosome_or_par(self):
        """Returns ``True`` if the locus is on an autosome or
        a pseudoautosomal region of chromosome X or Y.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.in_autosome_or_par())
            True

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("isAutosomalOrPseudoAutosomal", tbool)

    def in_mito(self):
        """Returns ``True`` if the locus is on mitochondrial DNA.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(locus.in_mito())
            True

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._method("isMitochondrial", tbool)

    @typecheck_method(before=expr_int32, after=expr_int32)
    def sequence_context(self, before=0, after=0):
        """Return the reference genome sequence at the locus.

        Examples
        --------

        Get the reference allele at a locus:

        .. doctest::
            :options: +SKIP

            >>> locus.sequence_context().value
            "G"

        Get the reference sequence at a locus including the previous 5 bases:

        .. doctest::
            :options: +SKIP

            >>> locus.sequence_context(before=5).value
            "ACTCGG"

        Notes
        -----
        This function requires that this locus' reference genome has an attached
        reference sequence. Use :meth:`.ReferenceGenome.add_sequence` to
        load and attach a reference sequence to a reference genome.

        Parameters
        ----------
        before : :class:`.Expression` of type :py:data:`.tint32`, optional
            Number of bases to include before the locus. Truncates at
            contig boundary.
        after : :class:`.Expression` of type :py:data:`.tint32`, optional
            Number of bases to include after the locus. Truncates at
            contig boundary.

        Returns
        -------
        :class:`.StringExpression`
        """

        rg = self.dtype.reference_genome
        if not rg.has_sequence():
            raise TypeError("Reference genome '{}' does not have a sequence loaded. Use 'add_sequence' to load the sequence from a FASTA file.".format(rg.name))
        return hl.get_sequence(self.contig, self.position, before, after, rg)


class IntervalExpression(Expression):
    """Expression of type :class:`.tinterval`.

    >>> interval = hl.interval(3, 11)
    >>> locus_interval = hl.parse_locus_interval("1:53242-90543")
    """

    @typecheck_method(value=expr_any)
    def contains(self, value):
        """Tests whether a value is contained in the interval.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(interval.contains(3))
            True

            >>> hl.eval_expr(interval.contains(11))
            False

        Parameters
        ----------
        value :
            Object with type matching the interval point type.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if `value` is contained in the interval, ``False`` otherwise.
        """
        if self.dtype.point_type != value.dtype:
            raise TypeError("expected '{}', found: '{}'".format(self.dtype.point_type, value.dtype))
        return self._method("contains", tbool, value)

    @typecheck_method(interval=expr_interval(expr_any))
    def overlaps(self, interval):
        """True if the the supplied interval contains any value in common with this one.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(interval.overlaps(hl.interval(5, 9)))
            True

            >>> hl.eval_expr(interval.overlaps(hl.interval(11, 20)))
            False

        Parameters
        ----------
        interval : :class:`.Expression` with type :py:data:`.tinterval`
            Interval object with the same point type.

        Returns
        -------
        :class:`.BooleanExpression`
        """
        if self.dtype.point_type != interval.dtype.point_type:
            raise TypeError("expected '{}', found: '{}'".format(self.dtype.point_type, interval.dtype.point_type))
        return self._method("overlaps", tbool, interval)

    @property
    def end(self):
        """Returns the end point.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(interval.end)
            11

        Returns
        -------
        :class:`.Expression`
        """
        return self._field("end", self.dtype.point_type)

    @property
    def start(self):
        """Returns the start point.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(interval.start)
            3

        Returns
        -------
        :class:`.Expression`
        """
        return self._field("start", self.dtype.point_type)

    @property
    def includes_start(self):
        """True if the interval includes the start point.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(interval.includes_start)
            True

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._field("includesStart", tbool)

    @property
    def includes_end(self):
        """True if the interval includes the end point.

        Examples
        --------
        .. doctest::

            >>> hl.eval_expr(interval.includes_end)
            False

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return self._field("includesEnd", tbool)


scalars = {tbool: BooleanExpression,
           tint32: Int32Expression,
           tint64: Int64Expression,
           tfloat32: Float32Expression,
           tfloat64: Float64Expression,
           tstr: StringExpression,
           tcall: CallExpression}

typ_to_expr = {
    tlocus: LocusExpression,
    tinterval: IntervalExpression,
    tcall: CallExpression,
    tdict: DictExpression,
    tarray: ArrayExpression,
    tset: SetExpression,
    tstruct: StructExpression,
    ttuple: TupleExpression
}


@typecheck(ast=AST, type=HailType, indices=Indices, aggregations=LinkedList)
def construct_expr(ast: AST,
                   type: HailType,
                   indices: Indices = Indices(),
                   aggregations: LinkedList = LinkedList(Aggregation)):
    if isinstance(type, tarray) and is_numeric(type.element_type):
        return ArrayNumericExpression(ast, type, indices, aggregations)
    elif type in scalars:
        return scalars[type](ast, type, indices, aggregations)
    elif type.__class__ in typ_to_expr:
        return typ_to_expr[type.__class__](ast, type, indices, aggregations)
    else:
        raise NotImplementedError(type)


@typecheck(name=str, type=HailType, indices=Indices, prefix=nullable(str))
def construct_reference(name, type, indices, prefix=None):
    if prefix is not None:
        ast = Select(TopLevelReference(prefix, indices), name)
    else:
        ast = TopLevelReference(name, indices)
    return construct_expr(ast, type, indices)
