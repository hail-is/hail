import abc
import plotly

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

    def __init__(self, aes, color=None):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        def plot_one_color(one_color_data, color):
            scatter_args = {
                "x": [element["x"] for element in one_color_data],
                "y": [element["y"] for element in one_color_data],
                "mode": "markers",
                "marker_color": color
            }

            if "size" in parent.aes or "size" in self.aes:
                scatter_args["marker_size"] = [element["size"] for element in one_color_data]
            fig_so_far.add_scatter(**scatter_args)

        def plot_continuous_color(data, colors):
            scatter_args = {
                "x": [element["x"] for element in data],
                "y": [element["y"] for element in data],
                "mode": "markers",
                "marker_color": colors
            }

            if "size" in parent.aes or "size" in self.aes:
                scatter_args["marker_size"] = [element["size"] for element in data]
            fig_so_far.add_scatter(**scatter_args)

        if self.color is not None:
            plot_one_color(agg_result, self.color)
        elif "color" in parent.aes or "color" in self.aes:
            if isinstance(agg_result[0]["color"], int):
                # Should show colors in continuous scale.
                input_color_nums = [element["color"] for element in agg_result]
                min_color = min(input_color_nums)
                max_color = max(input_color_nums)
                adjust_color = lambda input_color: (input_color - min_color) / max_color - min_color
                color_mapping = plotly.colors.sample_colorscale(parent.continuous_color_scale, [adjust_color(input_color) for input_color in input_color_nums])
                plot_continuous_color(agg_result, color_mapping)

            else:
                unique_colors = set([element["color"] for element in agg_result])

                for color in unique_colors:
                    filtered_data = [element for element in agg_result if element["color"] == color]
                    plot_one_color(filtered_data, color)
        else:
            plot_one_color(agg_result, "black")

    def get_stat(self):
        return StatIdentity()


def geom_point(mapping=aes(), color=None):
    return GeomPoint(mapping, color=color)


class GeomLine(Geom):

    def __init__(self, aes, color=None):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):

        def plot_one_color(one_color_data, color):
            scatter_args = {
                "x": [element["x"] for element in one_color_data],
                "y": [element["y"] for element in one_color_data],
                "mode": "lines",
                "line_color": color
            }
            fig_so_far.add_scatter(**scatter_args)

        if self.color is not None:
            plot_one_color(agg_result, self.color)
        elif "color" in parent.aes or "color" in self.aes:
            unique_colors = set([element["color"] for element in agg_result])
            for color in unique_colors:
                filtered_data = [element for element in agg_result if element["color"] == color]
                plot_one_color(filtered_data, color)
        else:
            plot_one_color(agg_result, "black")

    def get_stat(self):
        return StatIdentity()


def geom_line(mapping=aes(), color=None):
    return GeomLine(mapping, color=color)


class GeomText(Geom):

    def __init__(self, aes, color=None):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        def plot_one_color(one_color_data, color):
            scatter_args = {
                "x": [element["x"] for element in one_color_data],
                "y": [element["y"] for element in one_color_data],
                "text": [element["label"] for element in one_color_data],
                "mode": "text",
                "textfont_color": color
            }

            if "size" in parent.aes or "size" in self.aes:
                scatter_args["textfont_size"] = [element["size"] for element in one_color_data]
            fig_so_far.add_scatter(**scatter_args)

        if self.color is not None:
            plot_one_color(agg_result, self.color)
        elif "color" in parent.aes or "color" in self.aes:
            unique_colors = set([element["color"] for element in agg_result])
            for color in unique_colors:
                filtered_data = [element for element in agg_result if element["color"] == color]
                plot_one_color(filtered_data, color)
        else:
            plot_one_color(agg_result, "black")

    def get_stat(self):
        return StatIdentity()


def geom_text(mapping=aes(), color=None):
    return GeomText(mapping, color=color)


class GeomBar(Geom):

    def __init__(self, aes, color=None):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        item_list = list(agg_result.items())
        x_values = [item[0] for item in item_list]
        y_values = [item[1] for item in item_list]
        color = self.color if self.color is not None else "black"
        fig_so_far.add_bar(x=x_values, y=y_values, marker_color=color)

    def get_stat(self):
        return StatCount()


def geom_bar(mapping=aes(), color=None):
    return GeomBar(mapping, color=color)


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


class GeomVLine(Geom):

    def __init__(self, xintercept, linetype="solid", color=None):
        self.xintercept = xintercept
        self.aes = aes()
        self.linetype = linetype
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        line_attributes = {
            "x": self.xintercept,
            "line_dash": linetype_dict[self.linetype]
        }
        if self.color is not None:
            line_attributes["line_color"] = self.color

        fig_so_far.add_vline(**line_attributes)

    def get_stat(self):
        return StatNone()


def geom_vline(xintercept, linetype="solid", color=None):
    return GeomVLine(xintercept, linetype=linetype, color=color)
