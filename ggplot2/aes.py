from typing import Any, Dict, Optional

from hail.expr import Expression, literal

from .typing import Mapping
from .typecheck.typecheck import typecheck

# TODO is it actually correct to ignore values that are None for aesthetics besides x and y? in other words, is None sometimes relevant for overriding default values for some aesthetics?
@typecheck
def aes(x: Any = None, y: Any = None, **rest: Any) -> Mapping:
    return Mapping(**to_expressions({"x": x, "y": y}), rest=to_expressions(rest))


def to_expressions(mapping: Dict[str, Any]) -> Dict[str, Expression]:
    return {
        k: v if (v is None and k in {"x", "y"}) or isinstance(v, Expression) else literal(v) for k, v in mapping.items()
    }
