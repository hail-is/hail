from geoms import FigureAttribute

import hail as hl


def vars(*args):
    return list(*args)


class FacetWrap(FigureAttribute):

    def __init__(self, facets):
        self.facets = facets

    def get_faceter(self):
        return hl.struct(**{f"facet_{i}": expr for i, expr in enumerate(self.facets)})
