import abc
import math

from .geoms import FigureAttribute

import hail as hl


def vars(*args):
    """

    Parameters
    ----------
    *args: class:`.Expression`
        Fields to facet by.

    Returns
    -------
    :class:`StructExpression`
        A struct to pass to a faceter.

    """
    return hl.struct(**{f"var_{i}": arg for i, arg in enumerate(args)})


def facet_wrap(facets):
    """Introduce a one dimensional faceting on specified fields.

    Parameters
    ----------
    facets: :class:`StructExpression` created by `hl.ggplot.vars` function.
        The fields to facet on.

    Returns
    -------
    :class:`FigureAttribute`
        The faceter.

    """
    return FacetWrap(facets)


class Faceter(FigureAttribute):

    @abc.abstractmethod
    def get_expr_to_group_by(self):
        pass


class FacetWrap(Faceter):

    def __init__(self, facets):
        self.facets = facets

    def get_expr_to_group_by(self):
        return self.facets

    def get_facet_nrows_and_ncols(self, num_facet_values):
        ncol = int(math.ceil(math.sqrt(num_facet_values)))
        nrow = int(math.ceil(num_facet_values / ncol))

        return (nrow, ncol)
