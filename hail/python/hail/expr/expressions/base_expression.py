from typing import Tuple, Mapping
import numpy as np
import pandas as pd

import hail
import hail as hl
from hail.expr import expressions
from hail.expr.types import HailType, is_numeric, is_compound, is_setlike, tint32, \
    tint64, tfloat32, tfloat64, tstr, tbool, tarray, \
    tndarray, tset, tdict, tstruct, ttuple, tinterval, \
    tlocus, tcall, from_numpy
from hail import ir
from hail.typecheck import typecheck_method, nullable, anyfunc, linked_list
from hail.utils.java import Env
from hail.utils.linkedlist import LinkedList
from .indices import Indices, Aggregation

from hail.expr.types import summary_type


class Summary(object):
    def __init__(self, type, count, summ_fields, nested, header=None):
        self.count = count
        self.summ_fields = summ_fields
        self.nested = nested
        self.type = type
        self.header = header

    @staticmethod
    def pct(x):
        return f'{x*100:.2f}%'

    @staticmethod
    def format(x):
        if isinstance(x, float):
            return f'{x:.2f}'
        else:
            return str(x)

    def __str__(self):
        return self._ascii_string(depth=0, prefix=None)

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        return self._html_string(prefix=None)

    def _ascii_string(self, depth, prefix):
        spacing = '  ' * depth

        summary = ''
        if self.header:
            summary += f'\n{spacing}{self.header}'

        if prefix is not None:
            summary += f'\n\n{spacing}- {prefix} ({summary_type(self.type)}):'

        if len(self.summ_fields) > 0:
            max_n_len = max(len(n) for n in self.summ_fields.keys())
            for name, v in self.summ_fields.items():
                summary += f'\n{spacing}  {name.rjust(max_n_len)}: {self.format(v)}'
        for name, field in self.nested.items():
            if prefix is not None:
                name = f'{prefix}{name}'
            summary += field._ascii_string(depth + 1, prefix=name)

        return summary

    def _html_string(self, prefix):
        import html
        summary = ''
        if self.header:
            summary += f'<p>{self.header}</p>'
        if prefix is not None:
            summary += f'<b>{html.escape(prefix)}</b> (<i>{html.escape(summary_type(self.type))}</i>):'

        summary += '<ul>'
        if len(self.summ_fields) > 0:
            summary += '<table><tbody>'
            for name, v in self.summ_fields.items():
                summary += f'<tr><td>{html.escape(name)}</td><td>{html.escape(self.format(v))}</td></tr>'
            summary += '</tbody></table>'
        for name, field in self.nested.items():
            if prefix is not None:
                name = f'{prefix}{name}'
            summary += '<li>' + field._html_string(prefix=name) + '</li>'
        summary += '</ul>'

        return summary


class NamedSummary(object):
    def __init__(self, summary, name, header):
        self.header = header
        self.name = name
        self.summary = summary

    def __str__(self):
        s = self.summary._ascii_string(depth=0, prefix=self.name)
        if self.header:
            s = f'{self.header}\n======' + s
        return s

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        import html
        s = self.summary._html_string(prefix=self.name)
        if self.header:
            s = f'<h3>{html.escape(self.header)}</h3>' + s
        return s


class ExpressionException(Exception):
    def __init__(self, msg=''):
        self.msg = msg
        super(ExpressionException, self).__init__(msg)


class ExpressionWarning(Warning):
    def __init__(self, msg=''):
        self.msg = msg
        super(ExpressionWarning, self).__init__(msg)


def impute_type(x, partial_type=None):
    t = _impute_type(x, partial_type=partial_type)
    raise_for_holes(t)
    return t


def _impute_type(x, partial_type):
    from hail.genetics import Locus, Call
    from hail.utils import Interval, Struct

    def refine(t, refined):
        if t is None:
            return refined
        if not isinstance(t, type(refined)):
            raise ExpressionException(
                "Incompatible partial_type, {}, for value {}".format(partial_type, x))
        return t

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
    elif isinstance(x, Struct) or isinstance(x, dict) and isinstance(partial_type, tstruct):
        partial_type = refine(partial_type, hl.tstruct())
        t = tstruct(**{k: _impute_type(x[k], partial_type.get(k)) for k in x})
        return t
    elif isinstance(x, tuple):
        partial_type = refine(partial_type, hl.ttuple())
        return ttuple(*[_impute_type(element, partial_type[index] if index < len(partial_type) else None)
                        for index, element in enumerate(x)])
    elif isinstance(x, list):
        partial_type = refine(partial_type, hl.tarray(None))
        if len(x) == 0:
            return partial_type
        ts = {_impute_type(element, partial_type.element_type) for element in x}
        unified_type = super_unify_types(*ts)
        if unified_type is None:
            raise ExpressionException("Hail does not support heterogeneous arrays: "
                                      "found list with elements of types {} ".format(list(ts)))
        return tarray(unified_type)

    elif is_setlike(x):
        partial_type = refine(partial_type, hl.tset(None))
        if len(x) == 0:
            return partial_type
        ts = {_impute_type(element, partial_type.element_type) for element in x}
        unified_type = super_unify_types(*ts)
        if not unified_type:
            raise ExpressionException("Hail does not support heterogeneous sets: "
                                      "found set with elements of types {} ".format(list(ts)))
        return tset(unified_type)

    elif isinstance(x, Mapping):
        user_partial_type = partial_type
        partial_type = refine(partial_type, hl.tdict(None, None))
        if len(x) == 0:
            return partial_type
        kts = {_impute_type(element, partial_type.key_type) for element in x.keys()}
        vts = {_impute_type(element, partial_type.value_type) for element in x.values()}
        unified_key_type = super_unify_types(*kts)
        unified_value_type = super_unify_types(*vts)
        if unified_key_type is None:
            raise ExpressionException("Hail does not support heterogeneous dicts: "
                                      "found dict with keys {} of types {} ".format(list(x.keys()), list(kts)))
        if not unified_value_type:
            if unified_key_type == hl.tstr and user_partial_type is None:
                return tstruct(**{k: _impute_type(x[k], None) for k in x})

            raise ExpressionException("Hail does not support heterogeneous dicts: "
                                      "found dict with values of types {} ".format(list(vts)))
        return tdict(unified_key_type, unified_value_type)
    elif isinstance(x, np.generic):
        return from_numpy(x.dtype)
    elif isinstance(x, np.ndarray):
        element_type = from_numpy(x.dtype)
        return tndarray(element_type, x.ndim)
    elif x is None or pd.isna(x):
        return partial_type
    elif isinstance(x, (hl.expr.builders.CaseBuilder, hl.expr.builders.SwitchBuilder)):
        raise ExpressionException("'switch' and 'case' expressions must end with a call to either"
                                  "'default' or 'or_missing'")
    else:
        raise ExpressionException("Hail cannot automatically impute type of {}: {}".format(type(x), x))


def raise_for_holes(t):
    if t is None:
        raise ExpressionException("Hail cannot impute type")
    if t in (tbool, tint32, tint64, tfloat32, tfloat64, tstr, tcall):
        return
    if isinstance(t, (tlocus, tinterval)):
        return
    if isinstance(t, tstruct):
        for k, vt in t.items():
            try:
                raise_for_holes(vt)
            except ExpressionException as exc:
                raise ExpressionException(f'cannot impute field {k}') from exc
        return
    if isinstance(t, ttuple):
        for k, vt in enumerate(t):
            try:
                raise_for_holes(vt)
            except ExpressionException as exc:
                raise ExpressionException(f'cannot impute {k}th element') from exc
        return
    if isinstance(t, (tarray, tset)):
        try:
            raise_for_holes(t.element_type)
        except ExpressionException as exc:
            raise ExpressionException('cannot impute array elements') from exc
        return
    if isinstance(t, tdict):
        try:
            raise_for_holes(t.key_type)
        except ExpressionException as exc:
            raise ExpressionException('cannot impute dict keys') from exc
        try:
            raise_for_holes(t.value_type)
        except ExpressionException as exc:
            raise ExpressionException('cannot impute dict values') from exc
        return


def to_expr(e, dtype=None, partial_type=None) -> 'Expression':
    assert dtype is None or partial_type is None
    if isinstance(e, Expression):
        if dtype and not dtype == e.dtype:
            raise TypeError("expected expression of type '{}', found expression of type '{}'".format(dtype, e.dtype))
        return e
    return cast_expr(e, dtype, partial_type)


def cast_expr(e, dtype=None, partial_type=None) -> 'Expression':
    assert dtype is None or partial_type is None
    if not dtype:
        dtype = impute_type(e, partial_type)
    x = _to_expr(e, dtype)
    if isinstance(x, Expression):
        return x
    else:
        return hl.literal(x, dtype)


def _to_expr(e, dtype):
    if e is None:
        return None
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
            assert len(elements) > 0
            exprs = [element if isinstance(element, Expression)
                     else hl.literal(element, dtype.element_type)
                     for element in elements]
            indices, aggregations = unify_all(*exprs)
        x = ir.MakeArray([e._ir for e in exprs], None)
        return expressions.construct_expr(x, dtype, indices, aggregations)
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
            assert len(elements) > 0
            exprs = [element if isinstance(element, Expression)
                     else hl.literal(element, dtype.element_type)
                     for element in elements]
            indices, aggregations = unify_all(*exprs)
            x = ir.ToSet(ir.toStream(ir.MakeArray([e._ir for e in exprs], None)))
            return expressions.construct_expr(x, dtype, indices, aggregations)
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
            x = ir.MakeTuple([expr._ir for expr in exprs])
            return expressions.construct_expr(x, dtype, indices, aggregations)
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
    elif isinstance(dtype, hl.tndarray):
        return hl.nd.array(e)
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
        raise ExpressionException(
            "Cannot combine expressions from different source objects."
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


def super_unify_types(*ts):
    ts = [t for t in ts if t is not None]
    if len(ts) == 0:
        return None
    t0 = ts[0]
    if all(is_numeric(t) for t in ts):
        return unify_types_limited(*ts)
    if any(not isinstance(t, type(t0)) for t in ts):
        return None
    if isinstance(t0, tarray):
        et = super_unify_types(*[t.element_type for t in ts])
        return tarray(et)
    if isinstance(t0, tset):
        et = super_unify_types(*[t.element_type for t in ts])
        return tset(et)
    if isinstance(t0, tdict):
        kt = super_unify_types(*[t.key_type for t in ts])
        vt = super_unify_types(*[t.value_type for t in ts])
        return tdict(kt, vt)
    if isinstance(t0, tstruct):
        keys = [k for t in ts for k in t.fields]
        kvs = {k: super_unify_types(*[t.get(k, None) for t in ts])
               for k in keys}
        return tstruct(**kvs)
    if all(t0 == t for t in ts):
        return t0

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

    __array_ufunc__ = None  # disable NumPy coercions, so Hail coercions take priority

    @typecheck_method(x=ir.IR, type=nullable(HailType), indices=Indices, aggregations=linked_list(Aggregation))
    def __init__(self,
                 x: ir.IR,
                 type: HailType,
                 indices: Indices = Indices(),
                 aggregations: LinkedList = LinkedList(Aggregation)):

        self._ir: ir.IR = x
        self._type = type
        self._indices = indices
        self._aggregations = aggregations
        self._summary = None

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
        return self._compare_op("<", other)

    def __le__(self, other):
        return self._compare_op("<=", other)

    def __gt__(self, other):
        return self._compare_op(">", other)

    def __ge__(self, other):
        return self._compare_op(">=", other)

    def __nonzero__(self):
        raise ExpressionException(
            "The truth value of an expression is undefined\n"
            "    Hint: instead of 'if x', use 'hl.if_else(x, ...)'\n"
            "    Hint: instead of 'x and y' or 'x or y', use 'x & y' or 'x | y'\n"
            "    Hint: instead of 'not x', use '~x'")

    def __iter__(self):
        raise ExpressionException(f"{repr(self)} object is not iterable")

    def _compare_op(self, op, other):
        other = to_expr(other)
        left, right, success = unify_exprs(self, other)
        if not success:
            raise TypeError(f"Invalid '{op}' comparison, cannot compare expressions "
                            f"of type '{self.dtype}' and '{other.dtype}'")
        res = left._bin_op(op, right, hl.tbool)
        return res

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
        elif isinstance(typ, tndarray) and not isinstance(self.dtype, tndarray):
            return coercer.ec.coerce(self)
        else:
            return coercer.coerce(self)

    @staticmethod
    def _div_ret_type_f(t):
        assert is_numeric(t)
        if t == tint32 or t == tint64:
            return tfloat64
        else:
            # Float64 or Float32
            return t

    def _bin_op_numeric_unify_types(self, name, other):
        def numeric_proxy(t):
            if t == tbool:
                return tint32
            else:
                return t

        def scalar_type(t):
            if isinstance(t, tarray):
                return numeric_proxy(t.element_type)
            elif isinstance(t, tndarray):
                return numeric_proxy(t.element_type)
            else:
                return numeric_proxy(t)

        t = unify_types(scalar_type(self.dtype), scalar_type(other.dtype))
        if t is None:
            raise NotImplementedError("'{}' {} '{}'".format(self.dtype, name, other.dtype))

        if isinstance(self.dtype, tarray) or isinstance(other.dtype, tarray):
            return tarray(t)
        elif isinstance(self.dtype, tndarray):
            return tndarray(t, self.ndim)
        elif isinstance(other.dtype, tndarray):
            return tndarray(t, other.ndim)

        return t

    def _bin_op_numeric(self, name, other, ret_type_f=None):
        other = to_expr(other)
        unified_type = self._bin_op_numeric_unify_types(name, other)
        me = self._promote_numeric(unified_type)
        other = other._promote_numeric(unified_type)
        if ret_type_f:
            if isinstance(unified_type, tarray):
                ret_type = tarray(ret_type_f(unified_type.element_type))
            elif isinstance(unified_type, tndarray):
                ret_type = tndarray(ret_type_f(unified_type.element_type), unified_type.ndim)
            else:
                ret_type = ret_type_f(unified_type)
        else:
            ret_type = unified_type
        return me._bin_op(name, other, ret_type)

    def _bin_op_numeric_reverse(self, name, other, ret_type_f=None):
        return to_expr(other)._bin_op_numeric(name, self, ret_type_f)

    def _unary_op(self, name):
        return expressions.construct_expr(ir.ApplyUnaryPrimOp(name, self._ir), self._type, self._indices, self._aggregations)

    def _bin_op(self, name, other, ret_type):
        other = to_expr(other)
        indices, aggregations = unify_all(self, other)
        if (name in {'+', '-', '*', '/', '//'}) and (ret_type in {tint32, tint64, tfloat32, tfloat64}):
            op = ir.ApplyBinaryPrimOp(name, self._ir, other._ir)
        elif name in {"==", "!=", "<", "<=", ">", ">="}:
            op = ir.ApplyComparisonOp(name, self._ir, other._ir)
        else:
            d = {
                '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '//': 'floordiv',
                '%': 'mod', '**': 'pow'
            }
            op = ir.Apply(d.get(name, name), ret_type, self._ir, other._ir)
        return expressions.construct_expr(op, ret_type, indices, aggregations)

    def _bin_op_reverse(self, name, other, ret_type):
        return to_expr(other)._bin_op(name, self, ret_type)

    def _method(self, name, ret_type, *args):
        args = tuple(to_expr(arg) for arg in args)
        indices, aggregations = unify_all(self, *args)
        x = ir.Apply(name, ret_type, self._ir, *(a._ir for a in args))
        return expressions.construct_expr(x, ret_type, indices, aggregations)

    def _index(self, ret_type, key):
        key = to_expr(key)
        return self._method("index", ret_type, key)

    def _ir_lambda_method(self, irf, f, input_type, ret_type_f, *args):
        args = (to_expr(arg)._ir for arg in args)
        new_id = Env.get_uid()
        lambda_result = to_expr(
            f(expressions.construct_variable(new_id, input_type, self._indices, self._aggregations)))

        indices, aggregations = unify_all(self, lambda_result)
        x = irf(self._ir, new_id, lambda_result._ir, *args)
        return expressions.construct_expr(x, ret_type_f(lambda_result._type), indices, aggregations)

    def _ir_lambda_method2(self, other, irf, f, input_type1, input_type2, ret_type_f, *args):
        args = (to_expr(arg)._ir for arg in args)
        new_id1 = Env.get_uid()
        new_id2 = Env.get_uid()
        lambda_result = to_expr(
            f(expressions.construct_variable(new_id1, input_type1, self._indices, self._aggregations),
              expressions.construct_variable(new_id2, input_type2, other._indices, other._aggregations)))
        indices, aggregations = unify_all(self, other, lambda_result)
        x = irf(self._ir, other._ir, new_id1, new_id2, lambda_result._ir, *args)
        return expressions.construct_expr(x, ret_type_f(lambda_result._type), indices, aggregations)

    @property
    def dtype(self) -> HailType:
        """The data type of the expression.

        Returns
        -------
        :class:`.HailType`

        """
        return self._type

    def __bool__(self):
        raise TypeError("'Expression' objects cannot be converted to a 'bool'. Use 'hl.if_else' instead of Python if statements.")

    def __len__(self):
        raise TypeError("'Expression' objects have no static length: use 'hl.len' for the length of collections")

    def __contains__(self, item):
        class_name = type(self).__name__
        raise TypeError(f"`{class_name}` objects don't support the `in` operator.")

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
        return self._compare_op("==", other)

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
        return self._compare_op("!=", other)

    def _to_table(self, name):
        name, ds = self._to_relational(name)
        if isinstance(ds, hail.MatrixTable):
            entries = ds.key_cols_by().entries()
            entries = entries.order_by(*ds.row_key)
            return name, entries.select(name)
        else:
            if len(ds.key) != 0:
                ds = ds.order_by(*ds.key)
            return name, ds.select(name)

    def _to_relational(self, fallback_name):
        source = self._indices.source
        axes = self._indices.axes
        if not self._aggregations.empty():
            raise NotImplementedError('cannot convert aggregated expression to table')

        if source is None:
            return fallback_name, hl.Table.parallelize([hl.struct(**{fallback_name: self})], n_partitions=1)

        name = source._fields_inverse.get(self)
        top_level = name is not None
        if not top_level:
            name = fallback_name
        named_self = {name: self}
        if len(axes) == 0:
            x = source.select_globals(**named_self)
            ds = hl.Table.parallelize([x.index_globals()], n_partitions=1)
        elif isinstance(source, hail.Table):
            if top_level and name in source.key:
                named_self = {}
            ds = source.select(**named_self).select_globals()
        elif isinstance(source, hail.MatrixTable):
            if self._indices == source._row_indices:
                if top_level and name in source.row_key:
                    named_self = {}
                ds = source.select_rows(**named_self).select_globals().rows()
            elif self._indices == source._col_indices:
                if top_level and name in source.col_key:
                    named_self = {}
                ds = source.select_cols(**named_self).select_globals().key_cols_by().cols()
            else:
                assert self._indices == source._entry_indices
                ds = source.select_entries(**named_self).select_globals().select_cols().select_rows()
        return name, ds

    @typecheck_method(n=nullable(int),
                      width=nullable(int),
                      truncate=nullable(int),
                      types=bool,
                      handler=nullable(anyfunc),
                      n_rows=nullable(int),
                      n_cols=nullable(int))
    def show(self,
             n=None,
             width=None,
             truncate=None,
             types=True,
             handler=None,
             n_rows=None,
             n_cols=None):
        """Print the first few records of the expression to the console.

        If the expression refers to a value on a keyed axis of a table or matrix
        table, then the accompanying keys will be shown along with the records.

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

        Notes
        -----
        The output can be passed piped to another output source using the `handler` argument:

        >>> ht.foo.show(handler=lambda x: logging.info(x))  # doctest: +SKIP


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
        kwargs = {
            'n': n, 'width': width, 'truncate': truncate, 'types': types,
            'handler': handler, 'n_rows': n_rows, 'n_cols': n_cols}
        if kwargs.get('n_rows') is None:
            kwargs['n_rows'] = kwargs['n']
        del kwargs['n']
        _, ds = self._to_relational_preserving_rows_and_cols('<expr>')
        return ds.show(**{k: v for k, v in kwargs.items() if v is not None})

    def _to_relational_preserving_rows_and_cols(self, fallback_name):
        source = self._indices.source
        if isinstance(source, hl.Table):
            if self is source.row:
                return None, source
            if self is source.key:
                return None, source.select()
        if isinstance(source, hl.MatrixTable):
            if self is source.row:
                return None, source.rows()
            if self is source.row_key:
                return None, source.rows().select()
            if self is source.col:
                return None, source.key_cols_by().cols()
            if self is source.col_key:
                return None, source.select_cols().key_cols_by().cols()
            if self is source.entry:
                return None, source.select_rows().select_cols()
        return self._to_relational(fallback_name)

    @typecheck_method(path=str, delimiter=str, missing=str, header=bool)
    def export(self, path, delimiter='\t', missing='NA', header=True):
        """Export a field to a text file.

        Examples
        --------

        >>> small_mt.GT.export('output/gt.tsv')
        >>> with open('output/gt.tsv', 'r') as f:
        ...     for line in f:
        ...         print(line, end='')
        locus      alleles 0       1       2       3
        1:1        ["A","C"]       0/1     0/1     1/1     0/0
        1:2        ["A","C"]       1/1     1/1     0/1     1/1
        1:3        ["A","C"]       0/0     0/0     1/1     0/1
        1:4        ["A","C"]       0/0     0/0     0/0     1/1

        >>> small_mt.GT.export('output/gt-no-header.tsv', header=False)
        >>> with open('output/gt-no-header.tsv', 'r') as f:
        ...     for line in f:
        ...         print(line, end='')
        1:1        ["A","C"]       0/1     0/1     1/1     0/0
        1:2        ["A","C"]       1/1     1/1     0/1     1/1
        1:3        ["A","C"]       0/0     0/0     1/1     0/1
        1:4        ["A","C"]       0/0     0/0     0/0     1/1

        >>> small_mt.pop.export('output/pops.tsv')
        >>> with open('output/pops.tsv', 'r') as f:
        ...     for line in f:
        ...         print(line, end='')
        sample_idx      pop
        0       0
        1       0
        2       2
        3       0
        <BLANKLINE>

        >>> small_mt.ancestral_af.export('output/ancestral_af.tsv')
        >>> with open('output/ancestral_af.tsv', 'r') as f:
        ...     for line in f:
        ...         print(line, end='')
        locus   alleles ancestral_af
        1:1     ["A","C"]       3.8152e-01
        1:2     ["A","C"]       7.0588e-01
        1:3     ["A","C"]       4.9991e-01
        1:4     ["A","C"]       3.9616e-01
        <BLANKLINE>

        >>> small_mt.bn.export('output/bn.tsv')
        >>> with open('output/bn.tsv', 'r') as f:
        ...     for line in f:
        ...         print(line, end='')
        bn
        {"n_populations":3,"n_samples":4,"n_variants":4,"n_partitions":4,"pop_dist":[1,1,1],"fst":[0.1,0.1,0.1],"mixture":false}
        <BLANKLINE>


        Notes
        -----

        For entry-indexed expressions, if there is one column key field, the
        result of calling :func:`~hail.expr.functions.str` on that field is used as
        the column header. Otherwise, each compound column key is converted to
        JSON and used as a column header. For example:

        >>> small_mt = small_mt.key_cols_by(s=small_mt.sample_idx, family='fam1')
        >>> small_mt.GT.export('output/gt-no-header.tsv')
        >>> with open('output/gt-no-header.tsv', 'r') as f:
        ...     for line in f:
        ...         print(line, end='')
        locus   alleles {"s":0,"family":"fam1"} {"s":1,"family":"fam1"} {"s":2,"family":"fam1"} {"s":3,"family":"fam1"}
        1:1     ["A","C"]       0/1     0/1     1/1     0/0
        1:2     ["A","C"]       1/1     1/1     0/1     1/1
        1:3     ["A","C"]       0/0     0/0     1/1     0/1
        1:4     ["A","C"]       0/0     0/0     0/0     1/1
        <BLANKLINE>


        Parameters
        ----------
        path : :class:`str`
            The path to which to export.
        delimiter : :class:`str`
            The string for delimiting columns.
        missing : :class:`str`
            The string to output for missing values.
        header : :obj:`bool`
            When ``True`` include a header line.
        """
        uid = Env.get_uid()
        self_name, ds = self._to_relational_preserving_rows_and_cols(uid)
        if isinstance(ds, hl.Table):
            ds.export(output=path, delimiter=delimiter, header=header)
        else:
            assert len(self._indices.axes) == 2
            entries, cols = Env.get_uid(), Env.get_uid()
            t = ds.select_cols().localize_entries(entries, cols)
            t = t.order_by(*t.key)
            output_col_name = Env.get_uid()
            entry_array = t[entries]
            if self_name:
                entry_array = hl.map(lambda x: x[self_name], entry_array)
            entry_array = hl.map(lambda x: hl.if_else(hl.is_missing(x), missing, hl.str(x)),
                                 entry_array)
            file_contents = t.select(
                **{k: hl.str(t[k]) for k in ds.row_key},
                **{output_col_name: hl.delimit(entry_array, delimiter)})
            if header:
                col_key = t[cols]
                if len(ds.col_key) == 1:
                    col_key = hl.map(lambda x: x[0], col_key)
                column_names = hl.map(hl.str, col_key).collect(_localize=False)[0]
                header_table = hl.utils.range_table(1).key_by().select(
                    **{k: k for k in ds.row_key},
                    **{output_col_name: hl.delimit(column_names, delimiter)})
                file_contents = header_table.union(file_contents)
            file_contents.export(path, delimiter=delimiter, header=False)

    @typecheck_method(n=int, _localize=bool)
    def take(self, n, _localize=True):
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
        name, t = self._to_table(uid)
        e = t.take(n, _localize=False).map(lambda r: r[name])
        if _localize:
            return hl.eval(e)
        return e

    @typecheck_method(_localize=bool)
    def collect(self, _localize=True):
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
        name, t = self._to_table(uid)
        e = t.collect(_localize=False).map(lambda r: r[name])
        if _localize:
            return hl.eval(e)
        return e

    def _extra_summary_fields(self, agg_result):
        return {}

    def _summary_fields(self, agg_result, top):
        if top:
            return {}, self._nested_summary(agg_result[2], top)
        n_missing = agg_result[0]
        n_defined = agg_result[1]
        tot = n_missing + n_defined
        missing_value_str = str(n_missing) if n_missing == 0 else f'{n_missing} ({(n_missing / tot) * 100:.2f}%)'
        defined_value_str = str(n_defined) if n_defined == 0 else f'{n_defined} ({(n_defined / tot) * 100:.2f}%)'
        if n_defined == 0:
            return {'Non-missing': defined_value_str, 'Missing': missing_value_str}, {}
        return {'Non-missing': defined_value_str, 'Missing': missing_value_str,
                **self._extra_summary_fields(agg_result[2])}, self._nested_summary(agg_result[2], top)

    def _nested_summary(self, agg_result, top):
        return {}

    def _summary_aggs(self):
        return hl.missing(hl.tint32)

    def _all_summary_aggs(self):
        return hl.tuple((
            hl.agg.filter(hl.is_missing(self), hl.agg.count()),
            hl.agg.filter(hl.is_defined(self), hl.agg.count()),
            self._summary_aggs()))

    def _summarize(self, agg_res=None, *, name=None, header=None, top=False):
        src = self._indices.source
        summary_header = None
        if src is None or len(self._indices.axes) == 0:
            raise ValueError("Cannot summarize a scalar expression")
        if agg_res is None:
            count, agg_res = self._aggregation_method()(hl.tuple((hl.agg.count(), self._all_summary_aggs())))
            summary_header = f'{count} records.'
        sum_fields, nested = self._summary_fields(agg_res, top)
        summary = Summary(self._type, agg_res[0], sum_fields, nested, header=summary_header)
        if name is None and header is None:
            return summary
        else:
            return NamedSummary(summary, name, header)

    def summarize(self, handler=None):
        """Compute and print summary information about the expression.

        .. include:: _templates/experimental.rst
        """

        src = self._indices.source
        if self in src._fields:
            field_name = src._fields_inverse[self]
            prefix = field_name
        else:
            if self._ir.is_nested_field:
                prefix = self._ir.name
            else:
                prefix = '<expr>'

        if handler is None:
            handler = hl.utils.default_handler()
        handler(self._summarize(name=prefix))

    def _selector_and_agg_method(self):
        src = self._indices.source
        assert src is not None
        assert len(self._indices.axes) > 0
        if isinstance(src, hl.MatrixTable):
            if self._indices == src._row_indices:
                return src.select_rows, lambda t: t.aggregate_rows
            elif self._indices == src._col_indices:
                return src.select_cols, lambda t: t.aggregate_cols
            else:
                return src.select_entries, lambda t: t.aggregate_entries
        else:
            return src.select, lambda t: t.aggregate

    def _aggregation_method(self):
        return self._selector_and_agg_method()[1](self._indices.source)

    def _persist(self):
        src = self._indices.source
        if src is not None:
            raise ValueError("Can only persist a scalar (no Table/MatrixTable source)")
        expr = Env.backend().persist_expression(self)
        assert expr.dtype == self.dtype
        return expr
