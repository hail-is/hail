import abc
from typing import Optional, Dict, Any, TypeVar, List

import hail as hl
from hail.expr.expressions import Expression, ExpressionException, to_expr
from hail.expr.types import HailType, tint32, tint64, tfloat32, tfloat64, \
    tstr, tbool, tarray, tndarray, tset, tdict, tstruct, tunion, \
    ttuple, tinterval, tlocus, tcall

from hail.typecheck import TypeChecker, TypecheckFailure
from hail.utils.java import escape_parsable

__all__ = [
    'expr_any',
    'expr_int32',
    'expr_int64',
    'expr_float32',
    'expr_float64',
    'expr_call',
    'expr_bool',
    'expr_str',
    'expr_locus',
    'expr_interval',
    'expr_array',
    'expr_ndarray',
    'expr_set',
    'expr_dict',
    'expr_tuple',
    'expr_struct',
    'expr_oneof',
    'expr_numeric',
    'coercer_from_dtype',
]

T = TypeVar('T')


class ExprCoercer(TypeChecker):
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def str_t(self) -> str:
        ...

    def requires_conversion(self, t: HailType) -> bool:
        assert self.can_coerce(t), t
        return self._requires_conversion(t)

    @abc.abstractmethod
    def _requires_conversion(self, t: HailType) -> bool:
        """Assumes that :meth:`can_coerce` is ``True``"""
        ...

    @abc.abstractmethod
    def can_coerce(self, t: HailType) -> bool:
        ...

    def coerce(self, x) -> Expression:
        x = to_expr(x)
        if not self.can_coerce(x.dtype):
            raise ExpressionException(f"cannot coerce type '{x.dtype}' to type '{self.str_t}'")
        if self._requires_conversion(x.dtype):
            return self._coerce(x)
        else:
            return x

    def _coerce(self, x: Expression) -> Expression:
        # many coercers don't write any coercion rules, so don't override
        raise AssertionError()

    def expects(self):
        return f"expression of type {self.str_t}"

    def check(self, x: Any, caller: str, param: str) -> Any:
        try:
            return self.coerce(to_expr(x))
        except ExpressionException as e:
            raise TypecheckFailure from e

    def format(self, arg):
        if isinstance(arg, Expression):
            return str(arg)
        else:
            return super(ExprCoercer, self).format(arg)


class AnyCoercer(ExprCoercer):
    @property
    def str_t(self):
        return 'any'

    def _requires_conversion(self, t: HailType) -> bool:
        return False

    def can_coerce(self, t: HailType) -> bool:
        return True


class BoolCoercer(ExprCoercer):
    @property
    def str_t(self):
        return 'bool'

    def _requires_conversion(self, t: HailType) -> bool:
        return False

    def can_coerce(self, t: HailType) -> bool:
        return t == tbool


class Int32Coercer(ExprCoercer):
    @property
    def str_t(self):
        return 'int32'

    def _requires_conversion(self, t: HailType) -> bool:
        return t != tint32

    def can_coerce(self, t: HailType) -> bool:
        return t in (tbool, tint32)

    def _coerce(self, x):
        return x._method("toInt32", tint32)


class Int64Coercer(ExprCoercer):
    @property
    def str_t(self):
        return 'int64'

    def _requires_conversion(self, t: HailType) -> bool:
        return t != tint64

    def can_coerce(self, t: HailType) -> bool:
        return t in (tbool, tint32, tint64)

    def _coerce(self, x):
        return x._method("toInt64", tint64)


class Float32Coercer(ExprCoercer):
    @property
    def str_t(self):
        return 'float32'

    def _requires_conversion(self, t: HailType) -> bool:
        return t != tfloat32

    def can_coerce(self, t: HailType) -> bool:
        return t in (tbool, tint32, tint64, tfloat32)

    def _coerce(self, x):
        return x._method("toFloat32", tfloat32)


class Float64Coercer(ExprCoercer):
    @property
    def str_t(self):
        return 'float64'

    def _requires_conversion(self, t: HailType) -> bool:
        return t != tfloat64

    def can_coerce(self, t: HailType) -> bool:
        return t in (tbool, tint32, tint64, tfloat32, tfloat64)

    def _coerce(self, x):
        return x._method("toFloat64", tfloat64)


class StringCoercer(ExprCoercer):
    @property
    def str_t(self):
        return 'str'

    def _requires_conversion(self, t: HailType) -> bool:
        return False

    def can_coerce(self, t: HailType) -> bool:
        return t == tstr


class CallCoercer(ExprCoercer):
    @property
    def str_t(self):
        return 'call'

    def _requires_conversion(self, t: HailType) -> bool:
        return False

    def can_coerce(self, t: HailType) -> bool:
        return t == tcall


class LocusCoercer(ExprCoercer):
    def __init__(self, rg: Optional['hl.ReferenceGenome'] = None):
        super(LocusCoercer, self).__init__()
        self.rg = rg

    @property
    def str_t(self):
        return f'locus<{"any" if not self.rg else self.rg}>'

    def _requires_conversion(self, t: HailType) -> bool:
        return False

    def can_coerce(self, t: HailType) -> bool:
        if self.rg:
            return t == tlocus(self.rg)
        else:
            return isinstance(t, tlocus)


class IntervalCoercer(ExprCoercer):
    def __init__(self, point_type: ExprCoercer = AnyCoercer()):
        super(IntervalCoercer, self).__init__()
        self.point_type = point_type

    @property
    def str_t(self):
        return f'interval<{self.point_type.str_t}>'

    def _requires_conversion(self, t: HailType) -> bool:
        assert isinstance(t, tinterval)
        return self.point_type._requires_conversion(t.point_type)

    def can_coerce(self, t: HailType) -> bool:
        return isinstance(t, tinterval) and self.point_type.can_coerce(t.point_type)

    def _coerce(self, x):
        assert isinstance(x, hl.expr.IntervalExpression)
        return hl.interval(self.point_type.coerce(x.start),
                           self.point_type.coerce(x.end),
                           includes_start=x.includes_start,
                           includes_end=x.includes_end)


class ArrayCoercer(ExprCoercer):
    def __init__(self, ec: ExprCoercer = AnyCoercer()):
        super(ArrayCoercer, self).__init__()
        self.ec = ec

    @property
    def str_t(self):
        return f'array<{self.ec.str_t}>'

    def _requires_conversion(self, t: HailType) -> bool:
        assert isinstance(t, tarray)
        return self.ec._requires_conversion(t.element_type)

    def can_coerce(self, t: HailType) -> bool:
        return isinstance(t, tarray) and self.ec.can_coerce(t.element_type)

    def _coerce(self, x: Expression):
        assert isinstance(x, hl.expr.ArrayExpression)
        return hl.map(lambda x_: self.ec.coerce(x_), x)


class NDArrayCoercer(ExprCoercer):
    def __init__(self, ec: ExprCoercer = AnyCoercer()):
        super(NDArrayCoercer, self).__init__()
        self.ec = ec

    @property
    def str_t(self):
        return f'ndarray<{self.ec.str_t}>'

    def _requires_conversion(self, t: HailType) -> bool:
        assert isinstance(t, tndarray)
        return self.ec._requires_conversion(t.element_type)

    def can_coerce(self, t: HailType) -> bool:
        return isinstance(t, tndarray) and self.ec.can_coerce(t.element_type)

    def _coerce(self, x: Expression):
        assert isinstance(x, hl.expr.expressions.NDArrayExpression)
        return hl.map(lambda x_: self.ec.coerce(x_), x)


class SetCoercer(ExprCoercer):

    def __init__(self, ec: ExprCoercer = AnyCoercer()):
        super(SetCoercer, self).__init__()
        self.ec = ec

    @property
    def str_t(self):
        return f'set<{self.ec.str_t}>'

    def _requires_conversion(self, t: HailType) -> bool:
        assert isinstance(t, tset)
        return self.ec._requires_conversion(t.element_type)

    def can_coerce(self, t: HailType) -> bool:
        return isinstance(t, tset) and self.ec.can_coerce(t.element_type)

    def _coerce(self, x: Expression):
        assert isinstance(x, hl.expr.SetExpression)
        return hl.map(lambda x_: self.ec.coerce(x_), x)


class DictCoercer(ExprCoercer):
    def __init__(self, kc: ExprCoercer = AnyCoercer(), vc: ExprCoercer = AnyCoercer()):
        super(DictCoercer, self).__init__()
        self.kc = kc
        self.vc = vc

    @property
    def str_t(self):
        return f'dict<{self.kc.str_t, self.vc.str_t}>'

    def _requires_conversion(self, t: HailType) -> bool:
        assert isinstance(t, tdict)
        return self.kc._requires_conversion(t.key_type) or self.vc._requires_conversion(t.value_type)

    def can_coerce(self, t: HailType) -> bool:
        return isinstance(t, tdict) and self.kc.can_coerce(t.key_type) and self.vc.can_coerce(t.value_type)

    def _coerce(self, x: Expression):
        assert isinstance(x, hl.expr.DictExpression)
        if not self.kc._requires_conversion(x.dtype.key_type):
            # fast path
            return x.map_values(self.vc.coerce)
        else:
            return hl.dict(hl.map(lambda e: (self.kc.coerce(e[0]), self.vc.coerce(e[1])),
                                  hl.array(x)))


class TupleCoercer(ExprCoercer):
    def __init__(self, elements: Optional[List[ExprCoercer]] = None):
        super(TupleCoercer, self).__init__()
        self.elements = elements

    @property
    def str_t(self):
        if self.elements is None:
            return 'tuple'
        else:
            return f'tuple({", ".join(c.str_t for c in self.elements)})'

    def _requires_conversion(self, t: HailType) -> bool:
        assert isinstance(t, ttuple)
        if self.elements is None:
            return False
        else:
            assert len(self.elements) == len(t.types)
            return any(c._requires_conversion(t_) for c, t_ in zip(self.elements, t.types))

    def can_coerce(self, t: HailType):
        if self.elements is None:
            return isinstance(t, ttuple)
        else:
            return (isinstance(t, ttuple)
                    and len(t.types) == len(self.elements)
                    and all(c.can_coerce(t_) for c, t_ in zip(self.elements, t.types)))

    def _coerce(self, x: Expression):
        assert isinstance(x, hl.expr.TupleExpression)
        return hl.tuple(c.coerce(e) for c, e in zip(self.elements, x))


class StructCoercer(ExprCoercer):
    def __init__(self, fields: Optional[Dict[str, ExprCoercer]] = None):
        super(StructCoercer, self).__init__()
        self.fields = fields

    @property
    def str_t(self) -> str:
        if self.fields is None:
            return 'struct'
        else:
            field_strs = ', '.join(f'{escape_parsable(name)}: {c.str_t}' for name, c in self.fields.items())
            return f'struct{{{field_strs}}})'

    def _requires_conversion(self, t: HailType) -> bool:
        assert isinstance(t, tstruct)
        if self.fields is None:
            return False
        else:
            return any(c._requires_conversion(t[name]) for name, c in self.fields.items())

    def can_coerce(self, t: HailType):
        if self.fields is None:
            return isinstance(t, tstruct)
        else:
            return (isinstance(t, tstruct)
                    and len(t) == len(self.fields)
                    and all(expected[0] == actual[0] and expected[1].can_coerce(actual[1])
                            for expected, actual in zip(self.fields.items(), t.items())))

    def _coerce(self, x: Expression):
        assert isinstance(x, hl.expr.StructExpression)
        assert list(x.keys()) == list(self.fields.keys())
        return hl.struct(**{name: c.coerce(x[name]) for name, c in self.fields.items()})


class UnionCoercer(ExprCoercer):
    def __init__(self, cases: Optional[Dict[str, ExprCoercer]] = None):
        super(UnionCoercer, self).__init__()
        self.cases = cases

    @property
    def str_t(self) -> str:
        if self.cases is None:
            return 'union'
        else:
            case_strs = ', '.join(f'{escape_parsable(name)}: {c.str_t}' for name, c in self.cases.items())
            return f'union{{{case_strs}}})'

    def _requires_conversion(self, t: HailType) -> bool:
        assert isinstance(t, tunion)
        if self.cases is None:
            return False
        else:
            return any(c._requires_conversion(t[name]) for name, c in self.cases.items())

    def can_coerce(self, t: HailType):
        if self.cases is None:
            return isinstance(t, tunion)
        else:
            return (isinstance(t, tunion)
                    and len(t) == len(self.cases)
                    and all(expected[0] == actual[0] and expected[1].can_coerce(actual[1])
                            for expected, actual in zip(self.cases.items(), t.items())))

    def _coerce(self, x: Expression):
        assert isinstance(x, hl.expr.StructExpression)
        assert list(x.keys()) == list(self.cases.keys())
        raise NotImplementedError()


class OneOfExprCoercer(ExprCoercer):
    def __init__(self, *options: ExprCoercer):
        super(OneOfExprCoercer, self).__init__()
        options_ = []
        for o in options:
            if isinstance(o, OneOfExprCoercer):
                options_.extend(o.options)
            else:
                options_.append(o)
        self.options = options_

    @property
    def str_t(self) -> str:
        return ' or '.join(o.str_t for o in self.options)

    def _requires_conversion(self, t: HailType) -> bool:
        return all(o._requires_conversion(t) for o in filter(lambda c: c.can_coerce(t), self.options))

    def can_coerce(self, t: HailType) -> bool:
        return any(o.can_coerce(t) for o in self.options)

    def _coerce(self, x: Expression) -> Expression:
        first_coercer = next(filter(lambda o: o.can_coerce(x.dtype), self.options))
        return first_coercer._coerce(x)


expr_any = AnyCoercer()
expr_oneof = OneOfExprCoercer
expr_int32 = Int32Coercer()
expr_int64 = Int64Coercer()
expr_float32 = Float32Coercer()
expr_float64 = Float64Coercer()
expr_call = CallCoercer()
expr_str = StringCoercer()
expr_bool = BoolCoercer()
expr_locus = LocusCoercer
expr_interval = IntervalCoercer
expr_array = ArrayCoercer
expr_ndarray = NDArrayCoercer
expr_set = SetCoercer
expr_dict = DictCoercer
expr_tuple = TupleCoercer
expr_struct = StructCoercer
expr_union = UnionCoercer
expr_numeric = expr_oneof(expr_int32, expr_int64, expr_float32, expr_float64)

primitives: Dict[HailType, ExprCoercer] = {
    tint32: expr_int32,
    tint64: expr_int64,
    tfloat32: expr_float32,
    tfloat64: expr_float64,
    tbool: expr_bool,
    tcall: expr_call,
    tstr: expr_str
}


def coercer_from_dtype(t: HailType) -> ExprCoercer:
    if t in primitives:
        return primitives[t]
    elif isinstance(t, tlocus):
        return expr_locus(t.reference_genome)
    elif isinstance(t, tinterval):
        return expr_interval(coercer_from_dtype(t.point_type))
    elif isinstance(t, tarray):
        return expr_array(coercer_from_dtype(t.element_type))
    elif isinstance(t, tndarray):
        return expr_ndarray(coercer_from_dtype(t.element_type))
    elif isinstance(t, tset):
        return expr_set(coercer_from_dtype(t.element_type))
    elif isinstance(t, tdict):
        return expr_dict(coercer_from_dtype(t.key_type),
                         coercer_from_dtype(t.value_type))
    elif isinstance(t, ttuple):
        return expr_tuple([coercer_from_dtype(t_) for t_ in t.types])
    elif isinstance(t, tstruct):
        return expr_struct({name: coercer_from_dtype(t_) for name, t_ in t.items()})
    else:
        assert isinstance(t, tunion)
        return expr_union({name: coercer_from_dtype(t_) for name, t_ in t.items()})
