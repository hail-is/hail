import abc
import plotly

from .aes import aes
from .utils import categorical_strings_to_colors, continuous_nums_to_colors
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
        def plot_one_color(one_color_data, color, legend_name):
            scatter_args = {
                "x": [element["x"] for element in one_color_data],
                "y": [element["y"] for element in one_color_data],
                "mode": "markers",
                "marker_color": color,
                "name": legend_name
            }
            if "size" in parent.aes or "size" in self.aes:
                scatter_args["marker_size"] = [element["size"] for element in one_color_data]
            if "tooltip" in parent.aes or "tooltip" in self.aes:
                scatter_args["hovertext"] = [element["tooltip"] for element in one_color_data]
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
                color_mapping = continuous_nums_to_colors(input_color_nums, parent.continuous_color_scale)
                plot_continuous_color(agg_result, color_mapping)

            else:
                categorical_strings = set([element["color"] for element in agg_result])
                unique_color_mapping = categorical_strings_to_colors(categorical_strings, parent.discrete_color_scale)

                for category in categorical_strings:
                    filtered_data = [element for element in agg_result if element["color"] == category]
                    plot_one_color(filtered_data, unique_color_mapping[category], category)
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

        def plot_one_color(one_color_data, color, legend_name):
            scatter_args = {
                "x": [element["x"] for element in one_color_data],
                "y": [element["y"] for element in one_color_data],
                "mode": "lines",
                "name": legend_name,
                "line_color": color
            }
            if "tooltip" in parent.aes or "tooltip" in self.aes:
                scatter_args["hovertext"] = [element["tooltip"] for element in one_color_data]
            fig_so_far.add_scatter(**scatter_args)
        
        def plot_continuous_color(data, colors):
            scatter_args = {
                "x": [element["x"] for element in data],
                "y": [element["y"] for element in data],
                "mode": "lines",
                "line_color": colors
            }

            fig_so_far.add_scatter(**scatter_args)

        if self.color is not None:
            plot_one_color(agg_result, self.color)
        elif "color" in parent.aes or "color" in self.aes:
            if isinstance(agg_result[0]["color"], int):
                # Should show colors in continuous scale.
                raise ValueError("Do not currently support continuous color changing of lines")
            else:
                categorical_strings = set([element["color"] for element in agg_result])
                unique_color_mapping = categorical_strings_to_colors(categorical_strings, parent.discrete_color_scale)

                for category in categorical_strings:
                    filtered_data = [element for element in agg_result if element["color"] == category]
                    plot_one_color(filtered_data, unique_color_mapping[category], category)
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
        def plot_one_color(one_color_data, color, legend_name):
            scatter_args = {
                "x": [element["x"] for element in one_color_data],
                "y": [element["y"] for element in one_color_data],
                "text": [element["label"] for element in one_color_data],
                "mode": "text",
                "name": legend_name,
                "textfont_color": color
            }

            if "size" in parent.aes or "size" in self.aes:
                scatter_args["textfont_size"] = [element["size"] for element in one_color_data]
            if "tooltip" in parent.aes or "tooltip" in self.aes:
                scatter_args["hovertext"] = [element["tooltip"] for element in one_color_data]
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
                color_mapping = continuous_nums_to_colors(input_color_nums, parent.continuous_color_scale)
                plot_continuous_color(agg_result, color_mapping)

            else:
                categorical_strings = set([element["color"] for element in agg_result])
                unique_color_mapping = categorical_strings_to_colors(categorical_strings, parent.discrete_color_scale)

                for category in categorical_strings:
                    filtered_data = [element for element in agg_result if element["color"] == category]
                    plot_one_color(filtered_data, unique_color_mapping[category], category)

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
        bar_args = {
            "x": [item[0] for item in item_list],
            "y": [item[1] for item in item_list],
            "marker_color": self.color if self.color is not None else "black"
        }
        fig_so_far.add_bar(**bar_args)

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


class GeomTile(Geom):

    def __init__(self, aes):
        self.aes = aes

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        def plot_rects_multi_colors(agg_results, colors):
            for idx, row in enumerate(agg_results):
                x_center = row['x']
                y_center = row['y']
                width = row['width']
                height = row['height']
                x_left = x_center - width / 2
                x_right = x_center + width / 2
                y_up = y_center + height / 2
                y_down = y_center - height / 2
                fig_so_far.add_shape(type="rect", x0=x_left, y0=y_down, x1=x_right, y1=y_up, fillcolor=colors[idx])

        def plot_rects_one_color(agg_results, color):
            for idx, row in enumerate(agg_results):
                x_center = row['x']
                y_center = row['y']
                width = row['width']
                height = row['height']
                x_left = x_center - width / 2
                x_right = x_center + width / 2
                y_up = y_center + height / 2
                y_down = y_center - height / 2
                fig_so_far.add_shape(type="rect", x0=x_left, y0=y_down, x1=x_right, y1=y_up, fillcolor=color)

        if "fill" in parent.aes or "fill" in self.aes:
            if isinstance(agg_result[0]["fill"], int):
                input_color_nums = [element["fill"] for element in agg_result]
                color_mapping = continuous_nums_to_colors(input_color_nums, parent.continuous_color_scale)
                plot_rects_multi_colors(agg_result, color_mapping)
            else:
                categorical_strings = set([element["fill"] for element in agg_result])
                unique_color_mapping = categorical_strings_to_colors(categorical_strings, parent.discrete_color_scale)

                for category in categorical_strings:
                    filtered_data = [element for element in agg_result if element["fill"] == category]
                    plot_rects_one_color(filtered_data, unique_color_mapping[category])

        else:
            plot_rects_one_color(agg_result, "black")

    def get_stat(self):
        return StatIdentity()


def geom_tile(mapping=aes()):
    return GeomTile(mapping)
