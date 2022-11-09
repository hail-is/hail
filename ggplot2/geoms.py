from typing import Optional

from .aes import aes
from .typecheck.typecheck import typecheck
from .typing import Mapping, Data, GeomHistogram, GeomLine, GeomPoint


@typecheck
def geom_point(mapping: Mapping = aes(), data: Optional[Data] = None) -> GeomPoint:
    return GeomPoint(mapping, data)


@typecheck
def geom_histogram(mapping: Mapping = aes(), data: Optional[Data] = None, bins: int = 30) -> GeomHistogram:
    return GeomHistogram(mapping, data, bins)


@typecheck
def geom_line(mapping: Mapping = aes(), data: Optional[Data] = None) -> GeomLine:
    return GeomLine(mapping, data)
