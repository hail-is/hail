import abc
import math

from typing import Dict, Tuple

from .geoms import FigureAttribute
from .utils import n_partitions

import hail as hl
from hail import Expression, StructExpression


def vars(*args: Expression) -> StructExpression:
    """

    Parameters
    ----------
    *args: :class:`hail.expr.Expression`
        Fields to facet by.

    Returns
    -------
    :class:`hail.expr.StructExpression`
        A struct to pass to a faceter.

    """
    return hl.struct(**{f"var_{i}": arg for i, arg in enumerate(args)})


def facet_wrap(facets: StructExpression, *, nrow: int = None, ncol: int = None, scales: str = "fixed") -> "FacetWrap":
    """Introduce a one dimensional faceting on specified fields.

    Parameters
    ----------
    facets: :class:`hail.expr.StructExpression` created by `hl.ggplot.vars` function.
        The fields to facet on.
    nrow: :class:`int`
        The number of rows into which the facets will be spread. Will be ignored if `ncol` is set.
    ncol: :class:`int`
        The number of columns into which the facets will be spread.
    scales: :class:`str`
        Whether the scales are the same across facets. For more information and a list of supported options, see `the ggplot documentation <https://ggplot2-book.org/facet.html#controlling-scales>`__.

    Returns
    -------
    :class:`FigureAttribute`
        The faceter.

    """
    return FacetWrap(facets, nrow, ncol, scales)


class Faceter(FigureAttribute):
    @abc.abstractmethod
    def get_expr_to_group_by(self) -> StructExpression:
        pass


class FacetWrap(Faceter):
    _base_scale_mappings = {
        "shared_xaxes": "all",
        "shared_yaxes": "all",
    }

    _scale_mappings = {
        "fixed": _base_scale_mappings,
        "free_x": {
            **_base_scale_mappings,
            "shared_xaxes": False,
        },
        "free_y": {
            **_base_scale_mappings,
            "shared_yaxes": False,
        },
        "free": {
            "shared_xaxes": False,
            "shared_yaxes": False,
        },
    }

    def __init__(self, facets: StructExpression, nrow: int = None, ncol: int = None, scales: str = "fixed"):
        if nrow is not None and ncol is not None:
            raise ValueError("Both `nrow` and `ncol` were specified. " "Please specify only one of these values.")
        if scales not in self._scale_mappings:
            raise ValueError(
                f"An unsupported value ({scales}) was provided for `scales`. "
                f"Supported values are: {[k for k in self._scale_mappings.keys()]}."
            )
        self.nrow = nrow
        self.ncol = ncol
        self.facets = facets
        self.scales = scales

    def get_expr_to_group_by(self) -> StructExpression:
        return self.facets

    def get_facet_nrows_and_ncols(self, num_facet_values: int) -> Tuple[int, int]:
        if self.ncol is not None:
            return (n_partitions(num_facet_values, self.ncol), self.ncol)
        elif self.nrow is not None:
            return (self.nrow, n_partitions(num_facet_values, self.nrow))
        else:
            ncol = int(math.ceil(math.sqrt(num_facet_values)))
            return (n_partitions(num_facet_values, ncol), ncol)

    def get_shared_axis_kwargs(self) -> Dict[str, str]:
        return self._scale_mappings[self.scales]
