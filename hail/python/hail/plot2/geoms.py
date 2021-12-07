import abc

from .aes import aes


class FigureAttribute:

    @abc.abstractmethod
    def apply_to_fig(self, parent, collected, mapping_field_name, fig_so_far):
        return


class Geom(FigureAttribute):

    def __init__(self, aes):
        self.aes = aes


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

    def apply_to_fig(self, parent, collected, mapping_field_name, fig_so_far):
        scatter_args = {}
        scatter_args["x"] = [lookup(element, mapping_field_name, "x") for element in collected]
        scatter_args["y"] = [lookup(element, mapping_field_name, "y") for element in collected]
        scatter_args["mode"] = "markers"
        if "color" in parent.aes or "color" in self.aes:
            scatter_args["marker_color"] = [lookup_opt(element, mapping_field_name, "color") for element in collected]
        fig_so_far.add_scatter(**scatter_args)


def geom_point(mapping=aes()):
    return GeomPoint(mapping)


class GeomLine(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, parent, collected, mapping_field_name, fig_so_far):
        scatter_args = {
            "x": [lookup(element, mapping_field_name, "x") for element in collected],
            "y": [lookup(element, mapping_field_name, "y") for element in collected],
            "mode": "lines"
        }
        if "color" in parent.aes or "color" in self.aes:
            #FIXME: How should this work? All the colors have to match, there can be only one.
            scatter_args["line_color"] = lookup_opt(collected[0], mapping_field_name, "color")
        fig_so_far.add_scatter(**scatter_args)


def geom_line(mapping=aes()):
    return GeomLine(mapping)

class GeomText(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, parent, collected, mapping_field_name, fig_so_far):
        scatter_args = {
            "x": [lookup(element, mapping_field_name, "x") for element in collected],
            "y": [lookup(element, mapping_field_name, "y") for element in collected],
            "text": [lookup(element, mapping_field_name, "label") for element in collected],
            "mode": "text"
        }
        if "color" in parent.aes or "color" in self.aes:
            scatter_args["textfont_color"] = [lookup_opt(element, mapping_field_name, "color") for element in collected]
        fig_so_far.add_scatter(**scatter_args)


def geom_text(mapping=aes()):
    return GeomText(mapping)