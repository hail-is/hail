from dataclasses import fields, replace
from typing import Any, Optional

from .aes import aes
from .typecheck.dataclasses import pprint
from .typecheck.typecheck import typecheck
from .typing import Data, Geom, Mapping, Plot


@typecheck
def ggplot(data: Optional[Data] = None, mapping: Mapping = aes()) -> Plot:
    return Plot(data, mapping, [], None)


# TODO top-level config var for max undo stack depth before dropping old saves
def extend(plot: Plot, other: Any) -> Plot:
    kwargs: Optional[dict[str, Any]] = None
    if isinstance(other, Mapping):
        kwargs = {
            "mapping": replace(
                plot.mapping,
                **{k: v for k, v in {"x": other.x, "y": other.y}.items() if v is not None},
                rest={**plot.mapping.rest, **other.rest},
            )
        }
    elif isinstance(other, Geom):
        kwargs = {"geoms": [*plot.geoms, other]}

    if kwargs is not None:
        return replace(plot, **kwargs, _prev=plot)
    return plot


setattr(Plot, "__add__", extend)


@typecheck
def undo(plot: Plot, *, depth: int = 1) -> Plot:
    curr = plot
    index = depth
    while curr._prev is not None and index > 0:
        curr = curr._prev
        index -= 1
    return curr


@typecheck
def show(plot: Plot) -> None:
    pprint(plot)
