from typing import Optional, Tuple, Sequence, Callable
from hail.expr.expressions import construct_expr, expr_any, unify_all, Expression
from hail.expr.types import hail_type, HailType
from hail.ir import Apply, Ref
from hail.typecheck import typecheck, nullable, tupleof, anytype
from hail.utils.java import Env


class Function():
    def __init__(self,
                 f: Callable[..., Expression],
                 param_types: Sequence[HailType],
                 ret_type: HailType,
                 name: str,
                 type_args: Tuple[HailType, ...] = ()
                 ):
        self._f = f
        self._name = name
        self._type_args = type_args
        self._param_types = param_types
        self._ret_type = ret_type

    def __call__(self, *args: Expression) -> Expression:
        return self._f(*args)


@typecheck(f=anytype, param_types=hail_type, _name=nullable(str), type_args=tupleof(hail_type))
def define_function(f: Callable[..., Expression],
                    *param_types: HailType,
                    _name: Optional[str] = None,
                    type_args: Tuple[HailType, ...] = ()
                    ) -> Function:
    mname = _name if _name is not None else Env.get_uid()
    param_names = [Env.get_uid(mname) for _ in param_types]
    body = f(*(construct_expr(Ref(pn), pt) for pn, pt in zip(param_names, param_types)))
    ret_type = body.dtype

    Env.backend().register_ir_function(mname, type_args, param_names, param_types, ret_type, body)

    @typecheck(args=expr_any)
    def fun(*args: Expression) -> Expression:
        indices, aggregations = unify_all(*args)
        return construct_expr(Apply(mname, ret_type, *(a._ir for a in args), type_args=type_args), ret_type, indices, aggregations)

    return Function(fun, param_types, ret_type, mname, type_args)
