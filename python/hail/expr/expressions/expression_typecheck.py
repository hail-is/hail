import hail as hl
from hail.expr.types import HailType as HailType
from hail.expr.expressions import Expression, ExpressionException, to_expr
from hail.typecheck import TypeChecker, TypecheckFailure, identity, oneof
from hail.expr.types import is_numeric, is_container
import abc
from typing import *


class ExpressionTypechecker(TypeChecker):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(ExpressionTypechecker, self).__init__()

    @abc.abstractmethod
    def expected_type_str(self) -> str:
        ...

    def check(self, x: Any, caller: str, param: str) -> Any:
        try:
            expr_x = to_expr(x)
            if self.exact_match(expr_x.dtype):
                return expr_x
            elif self.can_cast_from(expr_x.dtype):
                return self.cast_from(expr_x.dtype)(expr_x)
            else:
                self.cast_error(expr_x)
        except ExpressionException as e:
            raise TypecheckFailure from e

    def can_cast_from(self, dtype: HailType) -> bool:
        return False

    @abc.abstractmethod
    def exact_match(self, dtype: HailType) -> bool:
        ...

    def cast_from(self, dtype: HailType) -> Callable[[Expression], Expression]:
        raise AssertionError

    def expects(self):
        return "expression of type '{}'".format(self.expected_type_str())

    def format(self, arg: Any) -> str:
        # problem: won't be called if there's composition above, like 'oneof'
        try:
            # inefficient, but required with current typecheck infrastructure
            expr_arg = to_expr(arg)
            return f"expression of type '{expr_arg.dtype}'"
        except ExpressionException:
            return super(ExpressionTypechecker, self).format(arg)

    def cast_error(self, x: Expression):
        raise ExpressionException(f"cannot cast expression of type '{x.dtype}' to type '{self.expected_type_str()}'")


class AnyChecker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'any'

    def exact_match(self, dtype: HailType) -> bool:
        return True


def lift(tc: Any) -> ExpressionTypechecker:
    if tc is Ellipsis:
        return AnyChecker()
    else:
        return tc


class Int32Checker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'int32'

    def can_cast_from(self, dtype: HailType) -> bool:
        return dtype == hl.tbool

    def exact_match(self, dtype: HailType) -> bool:
        return dtype == hl.tint32

    def cast_from(self, dtype: HailType) -> Any:
        if dtype == hl.tbool:
            return hl.int32
        else:
            raise AssertionError


class Int64Checker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'int64'

    def can_cast_from(self, dtype: HailType) -> bool:
        return dtype == hl.tbool or dtype == hl.tint32

    def exact_match(self, dtype: HailType) -> bool:
        return dtype == hl.tint64

    def cast_from(self, dtype: HailType) -> Callable[[Expression], Expression]:
        if dtype == hl.tint32 or dtype == hl.tbool:
            return hl.int64
        else:
            raise AssertionError


class Float32Checker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'float32'

    def can_cast_from(self, dtype: HailType) -> bool:
        return dtype == hl.tbool or dtype == hl.tint32 or dtype == hl.tint64

    def exact_match(self, dtype: HailType) -> bool:
        return dtype == hl.tfloat32

    def cast_from(self, dtype: HailType) -> Callable[[Expression], Expression]:
        if dtype == hl.tint32 or dtype == hl.tint64 or dtype == hl.tbool:
            return hl.float32
        else:
            raise AssertionError


class Float64Checker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'float64'

    def can_cast_from(self, dtype: HailType) -> bool:
        return dtype == hl.tbool or dtype == hl.tint32 or dtype == hl.tint64 or dtype == hl.tfloat32

    def exact_match(self, dtype: HailType) -> bool:
        return dtype == hl.tfloat64

    def cast_from(self, dtype: HailType) -> Callable[[Expression], Expression]:
        if dtype == hl.tint32 or dtype == hl.tint64 or dtype == hl.tfloat64 or dtype == hl.tbool:
            return hl.float64
        else:
            raise AssertionError


class BoolChecker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'bool'

    def exact_match(self, dtype: HailType) -> bool:
        return dtype == hl.tbool


class StringChecker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'str'

    def exact_match(self, dtype: HailType) -> bool:
        return dtype == hl.tstr


class StrChecker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'str'

    def exact_match(self, dtype: HailType) -> bool:
        return dtype == hl.tstr


class CallChecker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'call'

    def exact_match(self, dtype: HailType) -> bool:
        return dtype == hl.tcall


class ArrayChecker(ExpressionTypechecker):
    def __init__(self, element_type: Union[ExpressionTypechecker]):
        self.element_type = lift(element_type)
        super(ArrayChecker, self).__init__()

    def can_cast_from(self, dtype: HailType) -> bool:
        # the is_container check is to prevent possibly expensive nested conversion
        # we can enable it in the future if we want
        return (isinstance(dtype, hl.tarray)
                and not is_container(dtype.element_type)
                and self.element_type.can_cast_from(dtype.element_type))

    def exact_match(self, dtype: HailType) -> bool:
        return isinstance(dtype, hl.tarray) and self.element_type.exact_match(dtype.element_type)

    def expected_type_str(self) -> str:
        return f'array<{self.element_type.expected_type_str()}>'

    def cast_from(self, dtype: HailType) -> Callable[[Expression], Expression]:
        assert isinstance(dtype, hl.tarray)
        coerce_element = self.element_type.cast_from(dtype.element_type)
        return lambda a: hl.map(coerce_element, a)


class SetChecker(ExpressionTypechecker):
    def __init__(self, element_type: Union[ExpressionTypechecker]):
        self.element_type = lift(element_type)
        super(SetChecker, self).__init__()

    def can_cast_from(self, dtype: HailType) -> bool:
        return (isinstance(dtype, hl.tset)
                and not is_container(dtype.element_type)
                and self.element_type.can_cast_from(dtype.element_type))

    def exact_match(self, dtype: HailType) -> bool:
        return isinstance(dtype, hl.tset) and self.element_type.exact_match(dtype.element_type)

    def expected_type_str(self) -> str:
        return f'set<{self.element_type.expected_type_str()}>'

    def cast_from(self, dtype: HailType) -> Callable[[Expression], Expression]:
        assert isinstance(dtype, hl.tset)
        coerce_element = self.element_type.cast_from(dtype.element_type)
        return lambda a: hl.map(coerce_element, a)


class DictChecker(ExpressionTypechecker):
    def __init__(self,
                 key_type: Union[ExpressionTypechecker],
                 value_type: Union[ExpressionTypechecker]):
        self.key_type = lift(key_type)
        self.value_type = lift(value_type)
        super(DictChecker, self).__init__()

    def exact_match(self, dtype: HailType) -> bool:
        return (isinstance(dtype, hl.tdict)
                and self.key_type.exact_match(dtype.key_type)
                and self.value_type.exact_match(dtype.value_type))

    def expected_type_str(self) -> str:
        return f'dict<{self.key_type.expected_type_str()}, {self.value_type.expected_type_str()}>'


class TupleChecker(ExpressionTypechecker):
    def __init__(self, elements: Optional[Tuple[Union[ExpressionTypechecker]]]):
        self.elements = elements
        super(TupleChecker, self).__init__()

    def expected_type_str(self) -> str:
        if self.elements is None:
            return 'tuple'
        else:
            return f"tuple({', '.join(t.expected_type_str() for t in self.elements)})"

    def exact_match(self, dtype: HailType) -> bool:
        return (isinstance(dtype, hl.ttuple)
                and (self.elements is None or
                     len(dtype.types) == len(self.elements) and
                     all(self.elements[i].exact_match(dtype.types[i]) for i in range(len(dtype.types)))))


class StructChecker(ExpressionTypechecker):
    def __init__(self):
        # don't currently support field types
        super(StructChecker, self).__init__()

    def expected_type_str(self) -> str:
        return 'struct'

    def exact_match(self, dtype: HailType) -> bool:
        return isinstance(dtype, hl.tstruct)


class LocusChecker(ExpressionTypechecker):
    def expected_type_str(self):
        return 'locus'

    def exact_match(self, dtype: HailType) -> bool:
        return isinstance(dtype, hl.tlocus)


class IntervalChecker(ExpressionTypechecker):
    def __init__(self, point_type: Union[ExpressionTypechecker]):
        self.point_type = lift(point_type)
        super(IntervalChecker, self).__init__()

    def expected_type_str(self) -> str:
        return f"interval<{self.point_type.expected_type_str()}>"

    def exact_match(self, dtype: HailType) -> bool:
        return isinstance(dtype, hl.tinterval) and self.point_type.exact_match(dtype.point_type)

class NumericChecker(ExpressionTypechecker):
    def expected_type_str(self) -> str:
        return 'numeric'
    def exact_match(self, dtype: HailType) -> bool:
        return is_numeric(dtype)


expr_any = AnyChecker()
expr_int32 = Int32Checker()
expr_int64 = Int64Checker()
expr_float32 = Float32Checker()
expr_float64 = Float64Checker()
expr_bool = BoolChecker()
expr_str = StringChecker()
expr_call = CallChecker()
expr_locus = LocusChecker()
expr_struct = StructChecker()
expr_numeric = NumericChecker()


def expr_interval(t):
    return IntervalChecker(lift(t))


def expr_array(t):
    return ArrayChecker(lift(t))


def expr_set(t):
    return SetChecker(lift(t))


def expr_dict(kt, vt):
    return DictChecker(lift(kt), lift(vt))


def expr_tuple(elements):
    if elements is Ellipsis:
        lifted = None
    else:
        lifted = tuple(map(lift, elements))
    return TupleChecker(lifted)
