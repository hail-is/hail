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


class StatNone(Stat):
    def make_agg(self, x_expr, parent_struct, geom_struct):
        return hl.struct()


class StatCount(Stat):
    def make_agg(self, x_expr, parent_struct, geom_struct):
        return hl.agg.counter(x_expr)


class StatBin(Stat):

    def __init__(self, start, end, bins):
        self.start = start
        self.end = end
        self.bins = bins

    def make_agg(self, x_expr, parent_struct, geom_struct):
        return hl.agg.hist(x_expr, self.start, self.end, self.bins)


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

class GeomBar(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        item_list = list(agg_result.items())
        x_values = [item[0] for item in item_list]
        y_values = [item[1] for item in item_list]
        fig_so_far.add_bar(x=x_values, y=y_values)

    def get_stat(self):
        return StatCount()


def geom_bar(mapping=aes()):
    return GeomBar(mapping)


class GeomHistogram(Geom):

    def __init__(self, aes, min_bin=0, max_bin=100, bins=30):
        super().__init__(aes)
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.bins = bins

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        x_edges = agg_result.bin_edges
        y_values = agg_result.bin_freq
        num_edges = len(x_edges)
        x_values = []
        widths = []
        for i, x in enumerate(x_edges[:num_edges - 1]):
            x_values.append((x_edges[i + 1] - x) / 2 + x)
            widths.append(x_edges[i + 1] - x)
        print(x_values)
        print(widths)
        fig_so_far.add_bar(x=x_values, y=y_values, width=widths)

    def get_stat(self):
        return StatBin(self.min_bin, self.max_bin, self.bins)


def geom_histogram(mapping=aes(), min_bin=None, max_bin=None, bins=30):
    assert(min_bin is not None)
    assert(max_bin is not None)
    return GeomHistogram(mapping, min_bin, max_bin, bins)


linetype_dict = {
    "solid": "solid",
    "dashed": "dash",
    "dotted": "dot",
    "longdash": "longdash",
    "dotdash": "dashdot"
}


class GeomHLine(Geom):

    def __init__(self, yintercept, linetype="solid", color=None):
        self.yintercept = yintercept
        self.aes = aes()
        self.linetype = linetype
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        line_attributes = {
            "y": self.yintercept,
            "line_dash": linetype_dict[self.linetype]
        }
        if self.color is not None:
            line_attributes["line_color"] = self.color

        fig_so_far.add_hline(**line_attributes)

    def get_stat(self):
        return StatNone()


def geom_hline(yintercept, linetype="solid", color=None):
    return GeomHLine(yintercept, linetype=linetype, color=color)


