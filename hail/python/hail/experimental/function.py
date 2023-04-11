from typing import Optional
from hail.expr.expressions import construct_expr, expr_any, unify_all
from hail.expr.types import hail_type
from hail.ir import Apply, Ref
from hail.typecheck import typecheck, nullable, tupleof, anytype
from hail.utils.java import Env


class Function(object):
    def __init__(self, f, param_types, ret_type, name, type_args=()):
        self._f = f
        self._name = name
        self._type_args = type_args
        self._param_types = param_types
        self._ret_type = ret_type

    def __call__(self, *args):
        return self._f(*args)


@typecheck(f=anytype, param_types=hail_type, _name=nullable(str), type_args=tupleof(hail_type))
def define_function(f, *param_types, _name: Optional[str] = None, type_args=()) -> Function:
    mname = _name if _name is not None else Env.get_uid()
    param_names = [Env.get_uid(mname) for _ in param_types]
    body = f(*(construct_expr(Ref(pn), pt) for pn, pt in zip(param_names, param_types)))
    ret_type = body.dtype

    Env.backend().register_ir_function(mname, type_args, param_names, param_types, ret_type, body)

    @typecheck(args=expr_any)
    def f(*args):
        indices, aggregations = unify_all(*args)
        return construct_expr(Apply(mname, ret_type, *(a._ir for a in args), type_args=type_args), ret_type, indices, aggregations)

    return Function(f, param_types, ret_type, mname, type_args)
