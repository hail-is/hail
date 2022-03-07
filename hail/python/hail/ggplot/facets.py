import abc

from .geoms import FigureAttribute

import hail as hl


def vars(*args):
    return hl.tuple(*args)


def facet_grid(*, cols):
    return FacetGrid(cols=cols)


class Faceter(FigureAttribute):

    @abc.abstractmethod
    def get_expr_to_group_by(self):
        pass


class FacetGrid(Faceter):

    def __init__(self, cols):
        self.cols = cols

    def get_expr_to_group_by(self):
        return self.cols
