from typing import Optional, Union

from hail import MatrixTable, Table
from hail.expr import Expression

from .typecheck.dataclasses import typechecked_dataclass
from .typecheck.typecheck import typechecked_forward_ref


Plot = typechecked_forward_ref("Plot")
Data = Union[Table, MatrixTable]


# TODO should this actually have fields for all the supported aesthetics, instead of a dict with no validation?
#      is there a use case for aesthetics that we don't explicitly support?
@typechecked_dataclass
class Mapping:
    x: Optional[Expression]
    y: Optional[Expression]
    rest: dict[str, Expression]


@typechecked_dataclass
class Geom:
    mapping: Mapping
    data: Optional[Data]


@typechecked_dataclass
class GeomPoint(Geom):
    pass


@typechecked_dataclass
class GeomHistogram(Geom):
    bins: int


@typechecked_dataclass
class GeomLine(Geom):
    pass


@typechecked_dataclass
class Plot:
    data: Optional[Data]
    mapping: Mapping
    geoms: list[Geom]
    _prev: Optional[Plot]
