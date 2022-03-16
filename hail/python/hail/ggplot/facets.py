import abc

from .geoms import FigureAttribute

import hail as hl


def vars(*args):
    return hl.tuple(*args)


def facet_wrap(facets):
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
