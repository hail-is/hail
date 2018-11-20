from typing import *

from hail.expr import expressions
from hail.expr.types import *
from hail.genetics import Locus, Call
from hail.ir import *
from hail.typecheck import linked_list
from hail.utils import Interval, Struct
from hail.utils.java import *
from hail.utils.linkedlist import LinkedList
from .indices import *


class ExpressionException(Exception):
    def __init__(self, msg=''):
        self.msg = msg
        super(ExpressionException, self).__init__(msg)


class ExpressionWarning(Warning):
    def __init__(self, msg=''):
        self.msg = msg
        super(ExpressionWarning, self).__init__(msg)


def impute_type(x):
    if isinstance(x, Expression):
        return x.dtype
    elif isinstance(x, bool):
        return tbool
    elif isinstance(x, int):
        if hl.tint32.min_value <= x <= hl.tint32.max_value:
            return tint32
        elif hl.tint64.min_value <= x <= hl.tint64.max_value:
            return tint64
        else:
            raise ValueError("Hail has no integer data type large enough to store {}".format(x))
    elif isinstance(x, float):
        return tfloat64
    elif isinstance(x, str):
        return tstr
    elif isinstance(x, Locus):
        return tlocus(x.reference_genome)
    elif isinstance(x, Interval):
        return tinterval(x.point_type)
    elif isinstance(x, Call):
        return tcall
    elif isinstance(x, Struct):
        return tstruct(**{k: impute_type(x[k]) for k in x})
    elif isinstance(x, tuple):
        return ttuple(*(impute_type(element) for element in x))
    elif isinstance(x, list):
        if len(x) == 0:
            raise ExpressionException("Cannot impute type of empty list. Use 'hl.empty_array' to create an empty array.")
        ts = {impute_type(element) for element in x}
        unified_type = unify_types_limited(*ts)
        if not unified_type:
            raise ExpressionException("Hail does not support heterogeneous arrays: "
                                      "found list with elements of types {} ".format(list(ts)))
        return tarray(unified_type)
    elif isinstance(x, set):
        if len(x) == 0:
            raise ExpressionException("Cannot impute type of empty set. Use 'hl.empty_set' to create an empty set.")
        ts = {impute_type(element) for element in x}
        unified_type = unify_types_limited(*ts)
        if not unified_type:
            raise ExpressionException("Hail does not support heterogeneous sets: "
                                      "found set with elements of types {} ".format(list(ts)))
        return tset(unified_type)
    elif isinstance(x, dict):
        if len(x) == 0:
            raise ExpressionException("Cannot impute type of empty dict. Use 'hl.empty_dict' to create an empty dict.")
        kts = {impute_type(element) for element in x.keys()}
        vts = {impute_type(element) for element in x.values()}
        unified_key_type = unify_types_limited(*kts)
        unified_value_type = unify_types_limited(*vts)
        if not unified_key_type:
            raise ExpressionException("Hail does not support heterogeneous dicts: "
                                      "found dict with keys of types {} ".format(list(kts)))
        if not unified_value_type:
            raise ExpressionException("Hail does not support heterogeneous dicts: "
                                      "found dict with values of types {} ".format(list(vts)))
        return tdict(unified_key_type, unified_value_type)
    elif x is None:
        raise ExpressionException("Hail cannot impute the type of 'None'")
    elif isinstance(x, (hl.expr.builders.CaseBuilder, hl.expr.builders.SwitchBuilder)):
        raise ExpressionException("'switch' and 'case' expressions must end with a call to either"
                                  "'default' or 'or_missing'")
    else:
        raise ExpressionException("Hail cannot automatically impute type of {}: {}".format(type(x), x))


def to_expr(e, dtype=None) -> 'Expression':
    if isinstance(e, Expression):
        if dtype and not dtype == e.dtype:
            raise TypeError("expected expression of type '{}', found expression of type '{}'".format(dtype, e.dtype))
        return e
    if not dtype:
        dtype = impute_type(e)
    x = _to_expr(e, dtype)
    if isinstance(x, Expression):
        return x
    else:
        return hl.literal(x, dtype)


def _to_expr(e, dtype):
    if e is None:
        return hl.null(dtype)
    elif isinstance(e, Expression):
        if e.dtype != dtype:
            assert is_numeric(dtype), 'expected {}, got {}'.format(dtype, e.dtype)
            if dtype == tfloat64:
                return hl.float64(e)
            elif dtype == tfloat32:
                return hl.float32(e)
            elif dtype == tint64:
                return hl.int64(e)
            else:
                assert dtype == tint32
                return hl.int32(e)
        return e
    elif not is_compound(dtype):
        # these are not container types and cannot contain expressions if we got here
        return e
    elif isinstance(dtype, tstruct):
        new_fields = []
        found_expr = False
        for f, t in dtype.items():
            value = _to_expr(e[f], t)
            found_expr = found_expr or isinstance(value, Expression)
            new_fields.append(value)

        if not found_expr:
            return e
        else:
            exprs = [new_fields[i] if isinstance(new_fields[i], Expression)
                     else hl.literal(new_fields[i], dtype[i])
                     for i in range(len(new_fields))]
            fields = {name: expr for name, expr in zip(dtype.keys(), exprs)}
            from .typed_expressions import StructExpression
            return StructExpression._from_fields(fields)

    elif isinstance(dtype, tarray):
        elements = []
        found_expr = False
        for element in e:
            value = _to_expr(element, dtype.element_type)
            found_expr = found_expr or isinstance(value, Expression)
            elements.append(value)
        if not found_expr:
            return e
        else:
            assert (len(elements) > 0)
            exprs = [element if isinstance(element, Expression)
                     else hl.literal(element, dtype.element_type)
                     for element in elements]
            indices, aggregations = unify_all(*exprs)
        ir = MakeArray([e._ir for e in exprs], None)
        return expressions.construct_expr(ir, dtype, indices, aggregations)
    elif isinstance(dtype, tset):
        elements = []
        found_expr = False
        for element in e:
            value = _to_expr(element, dtype.element_type)
            found_expr = found_expr or isinstance(value, Expression)
            elements.append(value)
        if not found_expr:
            return e
        else:
            assert (len(elements) > 0)
            exprs = [element if isinstance(element, Expression)
                     else hl.literal(element, dtype.element_type)
                     for element in elements]
            indices, aggregations = unify_all(*exprs)
            ir = ToSet(MakeArray([e._ir for e in exprs], None))
            return expressions.construct_expr(ir, dtype, indices, aggregations)
    elif isinstance(dtype, ttuple):
        elements = []
        found_expr = False
        assert len(e) == len(dtype.types)
        for i in range(len(e)):
            value = _to_expr(e[i], dtype.types[i])
            found_expr = found_expr or isinstance(value, Expression)
            elements.append(value)
        if not found_expr:
            return e
        else:
            exprs = [elements[i] if isinstance(elements[i], Expression)
                     else hl.literal(elements[i], dtype.types[i])
                     for i in range(len(elements))]
            indices, aggregations = unify_all(*exprs)
            ir = MakeTuple([expr._ir for expr in exprs])
            return expressions.construct_expr(ir, dtype, indices, aggregations)
    elif isinstance(dtype, tdict):
        keys = []
        values = []
        found_expr = False
        for k, v in e.items():
            k_ = _to_expr(k, dtype.key_type)
            v_ = _to_expr(v, dtype.value_type)
            found_expr = found_expr or isinstance(k_, Expression)
            found_expr = found_expr or isinstance(v_, Expression)
            keys.append(k_)
            values.append(v_)
        if not found_expr:
            return e
        else:
            assert len(keys) > 0
            # Here I use `to_expr` to call `lit` the keys and values separately.
            # I anticipate a common mode is statically-known keys and Expression
            # values.
            key_array = to_expr(keys, tarray(dtype.key_type))
            value_array = to_expr(values, tarray(dtype.value_type))
            return hl.dict(hl.zip(key_array, value_array))
    else:
        raise NotImplementedError(dtype)


def unify_all(*exprs) -> Tuple[Indices, LinkedList]:
    if len(exprs) == 0:
        return Indices(), LinkedList(Aggregation)
    try:
        new_indices = Indices.unify(*[e._indices for e in exprs])
    except ExpressionException:
        # source mismatch
        from collections import defaultdict
        sources = defaultdict(lambda: [])
        for e in exprs:
            from .expression_utils import get_refs
            for name, inds in get_refs(e, *[e for a in e._aggregations for e in a.exprs]).items():
                sources[inds.source].append(str(name))
        raise ExpressionException("Cannot combine expressions from different source objects."
                                  "\n    Found fields from {n} objects:{fields}".format(
            n=len(sources),
            fields=''.join("\n        {}: {}".format(src, fds) for src, fds in sources.items())
        )) from None
    first, rest = exprs[0], exprs[1:]
    aggregations = first._aggregations
    for e in rest:
        aggregations = aggregations.push(*e._aggregations)
    return new_indices, aggregations


def unify_types_limited(*ts):
    type_set = set(ts)
    if len(type_set) == 1:
        # only one distinct class
        return next(iter(type_set))
    elif all(is_numeric(t) for t in ts):
        # assert there are at least 2 numeric types
        assert len(type_set) > 1
        if tfloat64 in type_set:
            return tfloat64
        elif tfloat32 in type_set:
            return tfloat32
        elif tint64 in type_set:
            return tint64
        else:
            assert type_set == {tint32, tbool}
            return tint32
    else:
        return None


def unify_types(*ts):
    limited_unify = unify_types_limited(*ts)
    if limited_unify is not None:
        return limited_unify
    elif all(isinstance(t, tarray) for t in ts):
        et = unify_types_limited(*(t.element_type for t in ts))
        if et is not None:
            return tarray(et)
        else:
            return None
    else:
        return None

def unify_exprs(*exprs: 'Expression') -> Tuple:
    assert len(exprs) > 0
    types = {e.dtype for e in exprs}

    # all types are the same
    if len(types) == 1:
        return exprs + (True,)

    for t in types:
        c = expressions.coercer_from_dtype(t)
        if all(c.can_coerce(e.dtype) for e in exprs):
            return tuple([c.coerce(e) for e in exprs]) + (True,)

    # cannot coerce all types to the same type
    return exprs + (False,)

class Expression(object):
    """Base class for Hail expressions."""

    @typecheck_method(ir=IR, type=nullable(HailType), indices=Indices, aggregations=linked_list(Aggregation))
    def __init__(self,
                 ir: IR,
                 type: HailType,
                 indices: Indices = Indices(),
                 aggregations: LinkedList = LinkedList(Aggregation)):

        self._ir: IR = ir
        self._type = type
        self._indices = indices
        self._aggregations = aggregations

    def describe(self, handler=print):
        """Print information about type, index, and dependencies."""
        if self._aggregations:
            agg_indices = set()
            for a in self._aggregations:
                agg_indices = agg_indices.union(a.indices.axes)
            agg_tag = ' (aggregated)'
            agg_str = f'Includes aggregation with index {list(agg_indices)}\n' \
                      f'    (Aggregation index may be promoted based on context)'
        else:
            agg_tag = ''
            agg_str = ''

        bar = '--------------------------------------------------------'
        s = '{bar}\n' \
            'Type:\n' \
            '    {t}\n' \
            '{bar}\n' \
            'Source:\n' \
            '    {src}\n' \
            'Index:\n' \
            '    {inds}{agg_tag}{maybe_bar}{agg}\n' \
            '{bar}'.format(bar=bar,
                           t=self.dtype.pretty(indent=4),
                           src=self._indices.source,
                           inds=list(self._indices.axes),
                           maybe_bar='\n' + bar + '\n' if agg_str else '',
                           agg_tag=agg_tag,
                           agg=agg_str)
        handler(s)

    def __lt__(self, other):
        raise NotImplementedError("'<' comparison with expression of type {}".format(str(self._type)))

    def __le__(self, other):
        raise NotImplementedError("'<=' comparison with expression of type {}".format(str(self._type)))

    def __gt__(self, other):
        raise NotImplementedError("'>' comparison with expression of type {}".format(str(self._type)))

    def __ge__(self, other):
        raise NotImplementedError("'>=' comparison with expression of type {}".format(str(self._type)))

    def __nonzero__(self):
        raise ExpressionException(
            "The truth value of an expression is undefined\n"
            "    Hint: instead of 'if x', use 'hl.cond(x, ...)'\n"
            "    Hint: instead of 'x and y' or 'x or y', use 'x & y' or 'x | y'\n"
            "    Hint: instead of 'not x', use '~x'")

    def __iter__(self):
        raise ExpressionException(f"{repr(self)} object is not iterable")

    def _is_scalar(self):
        return self._indices.source is None

    def _promote_scalar(self, typ):
        if typ == tint32:
            return hail.int32(self)
        elif typ == tint64:
            return hail.int64(self)
        elif typ == tfloat32:
            return hail.float32(self)
        else:
            assert typ == tfloat64
            return hail.float64(self)

    def _promote_numeric(self, typ):
        coercer = expressions.coercer_from_dtype(typ)
        if isinstance(typ, tarray) and not isinstance(self.dtype, tarray):
            return coercer.ec.coerce(self)
        else:
            return coercer.coerce(self)

    def _bin_op_numeric_unify_types(self, name, other):
        def numeric_proxy(t):
            if t == tbool:
                return tint32
            else:
                return t

        def scalar_type(t):
            if isinstance(t, tarray):
                return numeric_proxy(t.element_type)
            else:
                return numeric_proxy(t)

        t = unify_types(scalar_type(self.dtype), scalar_type(other.dtype))
        if t is None:
            raise NotImplementedError("'{}' {} '{}'".format(
                self.dtype, name, other.dtype))
        if isinstance(self.dtype, tarray) or isinstance(other.dtype, tarray):
            t = tarray(t)
        return t

    def _bin_op_numeric(self, name, other, ret_type_f=None):
        other = to_expr(other)
        unified_type = self._bin_op_numeric_unify_types(name, other)
        me = self._promote_numeric(unified_type)
        other = other._promote_numeric(unified_type)
        if ret_type_f:
            if isinstance(unified_type, tarray):
                ret_type = tarray(ret_type_f(unified_type.element_type))
            else:
                ret_type = ret_type_f(unified_type)
        else:
            ret_type = unified_type
        return me._bin_op(name, other, ret_type)

    def _bin_op_numeric_reverse(self, name, other, ret_type_f=None):
        return to_expr(other)._bin_op_numeric(name, self, ret_type_f)

    def _unary_op(self, name):
        return expressions.construct_expr(ApplyUnaryOp(name, self._ir), self._type, self._indices, self._aggregations)

    def _bin_op(self, name, other, ret_type):
        other = to_expr(other)
        indices, aggregations = unify_all(self, other)
        if (name in {'+', '-', '*', '/', '//'}) and (ret_type in {tint32, tint64, tfloat32, tfloat64}):
            op = ApplyBinaryOp(name, self._ir, other._ir)
        elif name in {"==", "!=", "<", "<=", ">", ">="}:
            op = ApplyComparisonOp(name, self._ir, other._ir)
        else:
            op = Apply(name, self._ir, other._ir)
        return expressions.construct_expr(op, ret_type, indices, aggregations)

    def _bin_op_reverse(self, name, other, ret_type):
        return to_expr(other)._bin_op(name, self, ret_type)

    def _field(self, name, ret_type):
        return expressions.construct_expr(GetField(self._ir, name),
                                          ret_type, self._indices, self._aggregations)

    def _method(self, name, ret_type, *args):
        args = tuple(to_expr(arg) for arg in args)
        indices, aggregations = unify_all(self, *args)
        ir = Apply(name, self._ir, *(a._ir for a in args))
        return expressions.construct_expr(ir, ret_type, indices, aggregations)

    def _index(self, ret_type, key):
        key = to_expr(key)
        return self._method("[]", ret_type, key)

    def _slice(self, ret_type, start=None, stop=None, step=None):
        ir_args = []
        if start is not None:
            start = to_expr(start)
            ir_args.append(start)
            start_str = "*"
        else:
            start_str = ""
        if stop is not None:
            stop = to_expr(stop)
            ir_args.append(stop)
            stop_str = "*"
        else:
            stop_str = ""
        if step is not None:
            raise NotImplementedError('Variable slice step size is not currently supported')

        mname = "[{}:{}]".format(start_str, stop_str)
        return self._method(mname, ret_type, *ir_args)

    def _ir_lambda_method(self, irf, f, input_type, ret_type_f, *args):
        args = (to_expr(arg)._ir for arg in args)
        new_id = Env.get_uid()
        lambda_result = to_expr(
            f(expressions.construct_variable(new_id, input_type, self._indices, self._aggregations)))

        indices, aggregations = unify_all(self, lambda_result)
        ir = irf(self._ir, new_id, lambda_result._ir, *args)
        return expressions.construct_expr(ir, ret_type_f(lambda_result._type), indices, aggregations)

    @property
    def dtype(self) -> HailType:
        """The data type of the expression.

        Returns
        -------
        :class:`.HailType`

        """
        return self._type

    def __len__(self):
        raise TypeError("'Expression' objects have no static length: use 'hl.len' for the length of collections")

    def __hash__(self):
        return super(Expression, self).__hash__()

    def __repr__(self):
        return f'<{self.__class__.__name__} of type {self.dtype}>'

    def __eq__(self, other):
        """Returns ``True`` if the two expressions are equal.

        Examples
        --------

        >>> x = hl.literal(5)
        >>> y = hl.literal(5)
        >>> z = hl.literal(1)

        >>> hl.eval(x == y)
        True

        >>> hl.eval(x == z)
        False

        Notes
        -----
        This method will fail with an error if the two expressions are not
        of comparable types.

        Parameters
        ----------
        other : :class:`.Expression`
            Expression for equality comparison.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the two expressions are equal.
        """
        other = to_expr(other)
        left, right, success = unify_exprs(self, other)
        if not success:
            raise TypeError(f"Invalid '==' comparison, cannot compare expressions "
                            f"of type '{self.dtype}' and '{other.dtype}'")
        return left._bin_op("==", right, tbool)

    def __ne__(self, other):
        """Returns ``True`` if the two expressions are not equal.

        Examples
        --------

        >>> x = hl.literal(5)
        >>> y = hl.literal(5)
        >>> z = hl.literal(1)

        >>> hl.eval(x != y)
        False

        >>> hl.eval(x != z)
        True

        Notes
        -----
        This method will fail with an error if the two expressions are not
        of comparable types.

        Parameters
        ----------
        other : :class:`.Expression`
            Expression for inequality comparison.

        Returns
        -------
        :class:`.BooleanExpression`
            ``True`` if the two expressions are not equal.
        """
        other = to_expr(other)
        left, right, success = unify_exprs(self, other)
        if not success:
            raise TypeError(f"Invalid '!=' comparison, cannot compare expressions "
                            f"of type '{self.dtype}' and '{other.dtype}'")
        return left._bin_op("!=", right, tbool)

    def _to_table(self, name):
        source = self._indices.source
        axes = self._indices.axes
        if not self._aggregations.empty():
            raise NotImplementedError('cannot convert aggregated expression to table')
        if source is None:
            # scalar expression
            df = Env.dummy_table()
            df = df.select(**{name: self})
            to_return = df
        elif len(axes) == 0:
            uid = Env.get_uid()
            source = source.select_globals(**{uid: self})
            df = Env.dummy_table()
            df = df.select(**{name: source.index_globals()[uid]})
            to_return = df
        elif len(axes) == 1:
            if isinstance(source, hail.Table):
                df = source
                field_name = source._fields_inverse.get(self)
                if field_name is not None:
                    if field_name in source.key:
                        df = df.select()
                    else:
                        df = df.select(field_name)
                    if field_name != name:
                        df = df.rename({field_name: name})
                else:
                    df = df.select(**{name: self})
                to_return = df.select_globals()
            else:
                assert isinstance(source, hail.MatrixTable)
                if self._indices == source._row_indices:
                    field_name = source._fields_inverse.get(self)
                    if field_name is not None:
                        if field_name in source.row_key:
                            m = source.select_rows()
                        else:
                            m = source.select_rows(field_name)
                        m = m.rename({field_name: name})
                    else:
                        m = source.select_rows(**{name: self})
                    to_return = m.rows().select_globals()
                else:
                    field_name = source._fields_inverse.get(self)
                    if field_name is not None:
                        if field_name in source.col_key:
                            m = source.select_cols()
                        else:
                            m = source.select_cols(field_name)
                        m = m.rename({field_name: name})
                    else:
                        m = source.select_cols(**{name: self})
                    to_return = m.key_cols_by().cols().select_globals()
        else:
            assert len(axes) == 2
            assert isinstance(source, hail.MatrixTable)
            source = source.select_entries(**{name: self}).select_rows().select_cols()
            to_return = source.key_cols_by().entries().select_globals()
        assert self.dtype == to_return[name].dtype, f'type mismatch:\n' \
                                                    f'  Actual:    {self.dtype}\n' \
                                                    f'  Should be: {to_return[name].dtype}'
        return to_return


    @typecheck_method(n=int, width=int, truncate=nullable(int), types=bool, handler=anyfunc)
    def show(self, n=10, width=90, truncate=None, types=True, handler=print):
        """Print the first few rows of the table to the console.

        Examples
        --------

        >>> table1.SEX.show()
        +-------+-----+
        |    ID | SEX |
        +-------+-----+
        | int32 | str |
        +-------+-----+
        |     1 | "M" |
        |     2 | "M" |
        |     3 | "F" |
        |     4 | "F" |
        +-------+-----+

        >>> hl.literal(123).show()
        +--------+
        | <expr> |
        +--------+
        |  int32 |
        +--------+
        |    123 |
        +--------+

        Warning
        -------
        Extremely experimental.

        Parameters
        ----------
        n : :obj:`int`
            Maximum number of rows to show.
        width : :obj:`int`
            Horizontal width at which to break columns.
        truncate : :obj:`int`, optional
            Truncate each field to the given number of characters. If
            ``None``, truncate fields to the given `width`.
        types : :obj:`bool`
            Print an extra header line with the type of each field.
        """
        handler(self._show(n, width, truncate, types))

    def _show(self, n=10, width=90, truncate=None, types=True):
        name = '<expr>'
        source = self._indices.source
        if isinstance(source, hl.Table):
            if self is source.row:
                return source._show(n, width, truncate, types)
            elif self is source.key:
                return source.select()._show(n, width, truncate, types)
        elif isinstance(source, hl.MatrixTable):
            if self is source.row:
                return source.rows()._show(n, width, truncate, types)
            elif self is source.row_key:
                return source.rows().select()._show(n, width, truncate, types)
            if self is source.col:
                return source.cols()._show(n, width, truncate, types)
            elif self is source.col_key:
                return source.cols().select()._show(n, width, truncate, types)
            if self is source.entry:
                return source.select_rows().select_cols().entries()._show(n, width, truncate, types)
        if source is not None:
            name = source._fields_inverse.get(self, name)
        t = self._to_table(name)
        if name in t.key:
            t = t.key_by(name).select()
        return t._show(n, width, truncate, types)


    @typecheck_method(n=int)
    def take(self, n):
        """Collect the first `n` records of an expression.

        Examples
        --------

        Take the first three rows:

        >>> table1.X.take(3)
        [5, 6, 7]

        Warning
        -------
        Extremely experimental.

        Parameters
        ----------
        n : int
            Number of records to take.

        Returns
        -------
        :obj:`list`
        """
        uid = Env.get_uid()
        return [r[uid] for r in self._to_table(uid).take(n)]

    def collect(self):
        """Collect all records of an expression into a local list.

        Examples
        --------

        Collect all the values from `C1`:

        >>> table1.C1.collect()
        [2, 2, 10, 11]

        Warning
        -------
        Extremely experimental.

        Warning
        -------
        The list of records may be very large.

        Returns
        -------
        :obj:`list`
        """
        uid = Env.get_uid()
        t = self._to_table(uid).key_by()
        return [r[uid] for r in t._select("collect", hl.struct(**{uid: t[uid]})).collect()]

    def _aggregation_method(self):
        src = self._indices.source
        assert src is not None
        assert len(self._indices.axes) > 0
        if isinstance(src, hl.MatrixTable):
            if self._indices == src._row_indices:
                return src.aggregate_rows
            elif self._indices == src._col_indices:
                return src.aggregate_cols
            else:
                return src.aggregate_entries
        else:
            return src.aggregate
