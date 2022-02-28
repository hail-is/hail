from geoms import FigureAttribute

import hail as hl


def vars(*args):
    return hl.tuple(*args)


def facet_grid(*, cols):
    return FacetGrid(cols=cols)


class FacetGrid(FigureAttribute):

    def __init__(self, cols):
        self.cols = cols

    def get_faceter(self):
        return lambda x: hl.agg.group_by(self.cols, x)
