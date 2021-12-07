import abc

from .aes import aes
import hail as hl


class FigureAttribute:
    pass


class Geom(FigureAttribute):

    def __init__(self, aes):
        self.aes = aes

    @abc.abstractmethod
    def apply_to_fig(self, parent, agg_result, fig_so_far):
        return

    @abc.abstractmethod
    def get_stat(self):
        return ...


def lookup(struct, mapping_field_name, target):
    maybe = lookup_opt(struct, mapping_field_name, target)
    if maybe is None:
        raise KeyError(f"aesthetic field {target} expected but not found")
    return maybe


def lookup_opt(struct, mapping_field_name, target):
    return struct["figure_mapping"].get(target, struct[mapping_field_name].get(target))


class Stat:
    @abc.abstractmethod
    def make_agg(self, x_expr, parent_struct, geom_struct):
        return


class StatIdentity(Stat):
    def make_agg(self, x_expr, parent_struct, geom_struct):
        combined = parent_struct.annotate(**geom_struct)
        return hl.agg.collect(combined)


class StatCount(Stat):
    def make_agg(self, x_expr, parent_struct, geom_struct):
        return hl.agg.counter(x_expr)


class GeomPoint(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        scatter_args = {
            "x": [element["x"] for element in agg_result],
            "y": [element["y"] for element in agg_result],
            "mode": "markers"
        }
        if "color" in parent.aes or "color" in self.aes:
            scatter_args["marker_color"] = [element["color"] for element in agg_result]
        fig_so_far.add_scatter(**scatter_args)

    def get_stat(self):
        return StatIdentity()


def geom_point(mapping=aes()):
    return GeomPoint(mapping)


class GeomLine(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        scatter_args = {
            "x": [element["x"] for element in agg_result],
            "y": [element["y"] for element in agg_result],
            "mode": "lines"
        }
        if "color" in parent.aes or "color" in self.aes:
            #FIXME: How should this work? All the colors have to match, there can be only one.
            scatter_args["line_color"] = agg_result[0]["color"]
        fig_so_far.add_scatter(**scatter_args)

    def get_stat(self):
        return StatIdentity()


def geom_line(mapping=aes()):
    return GeomLine(mapping)


class GeomText(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        scatter_args = {
            "x": [element["x"] for element in agg_result],
            "y": [element["y"] for element in agg_result],
            "text": [element["label"] for element in agg_result],
            "mode": "text"
        }
        if "color" in parent.aes or "color" in self.aes:
            scatter_args["textfont_color"] = [element["color"] for element in agg_result]
        fig_so_far.add_scatter(**scatter_args)

    def get_stat(self):
        return StatIdentity()


def geom_text(mapping=aes()):
    return GeomText(mapping)