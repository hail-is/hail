import abc

from .aes import aes

class FigureAttribute:
    pass


class Geom(FigureAttribute):

    def __init__(self, aes):
        self.aes = aes

    @abc.abstractmethod
    def apply_to_fig(self, parent, mapping_field_name, fig_so_far):
        return


def lookup(struct, mapping_field_name, target):
    maybe = lookup_opt(struct, mapping_field_name, target)
    if maybe is None:
        raise KeyError(f"aesthetic field {target} expected but not found")
    return maybe


def lookup_opt(struct, mapping_field_name, target):
    return struct["figure_mapping"].get(target, struct[mapping_field_name].get(target))


class GeomPoint(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, collected, mapping_field_name, fig_so_far):
        x = [lookup(element, mapping_field_name, "x") for element in collected]
        y = [lookup(element, mapping_field_name, "y") for element in collected]
        color = [lookup_opt(element, mapping_field_name, "color") for element in collected]
        fig_so_far.add_scatter(x=x, y=y, marker_color=color, mode="markers")


def geom_point(mapping=aes()):
    return GeomPoint(mapping)


class GeomLine(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, collected, mapping_field_name, fig_so_far):
        x = [lookup(element, mapping_field_name, "x") for element in collected]
        y = [lookup(element, mapping_field_name, "y") for element in collected]
        color = [lookup_opt(element, mapping_field_name, "color") for element in collected]
        #FIXME: How should this work? All the colors have to match, there can be only one.
        fig_so_far.add_scatter(x=x, y=y, line_color=color[0], mode="lines")


def geom_line(mapping=aes()):
    return GeomLine(mapping)
