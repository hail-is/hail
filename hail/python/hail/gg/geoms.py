import abc
import plotly

from .aes import aes
from .utils import categorical_strings_to_colors, continuous_nums_to_colors
import hail as hl

from ..ir.utils import is_continuous_type


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


class GeomLineBasic(Geom):
    def __init__(self, aes, color):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):

        def plot_group(data, color=None):
            scatter_args = {
                "x": [element["x"] for element in data],
                "y": [element["y"] for element in data],
                "mode": "lines",
                "line_color": color

            }
            if "color_legend" in data[0]:
                scatter_args["name"] = data[0]["color_legend"]

            if "tooltip" in parent.aes or "tooltip" in self.aes:
                scatter_args["hovertext"] = [element["tooltip"] for element in data]
            fig_so_far.add_scatter(**scatter_args)

        if self.color is not None:
            plot_group(agg_result, self.color)
        elif "color" in parent.aes or "color" in self.aes:
            groups = set([element["group"] for element in agg_result])
            for group in groups:
                just_one_group = [element for element in agg_result if element["group"] == group]
                plot_group(just_one_group, just_one_group[0]["color"])
        else:
            plot_group(agg_result, "black")

    @abc.abstractmethod
    def get_stat(self):
        return ...


class Stat:
    @abc.abstractmethod
    def make_agg(self, x_expr, parent_struct, geom_struct):
        return

    @abc.abstractmethod
    def listify(self, agg_result):
        #Turns the agg result into a data list to be plotted.
        return


class StatIdentity(Stat):
    def make_agg(self, mapping):
        return hl.agg.collect(mapping)

    def listify(self, agg_result):
        # Collect aggregator returns a list, nothing to do.
        return agg_result


class StatFunction(Stat):

    def __init__(self, fun):
        self.fun = fun

    def make_agg(self, combined):
        with_y_value = combined.annotate(y=self.fun(combined.x))
        return hl.agg.collect(with_y_value)

    def listify(self, agg_result):
        # Collect aggregator returns a list, nothing to do.
        return agg_result


class StatNone(Stat):
    def make_agg(self, mapping):
        return hl.struct()

    def listify(self, agg_result):
        return []


class StatCount(Stat):
    def make_agg(self, mapping):
        # Let's see. x_expr is the thing to group by. If any of the
        # aesthetics in geom_struct are just pointers to x_expr, that's fine.
        # Maybe I just do a `take(1) for every field of parent_struct and geom_struct?
        # Or better, a collect_as_set where I error if size is greater than 1?
        #group by all discrete variables and x
        discrete_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys()
                              if not is_continuous_type(mapping[aes_key].dtype)}
        discrete_variables["x"] = mapping["x"]
        return hl.agg.group_by(hl.struct(**discrete_variables), hl.agg.count())


    def listify(self, agg_result):
        unflattened_items = agg_result.items()
        res = []
        for discrete_variables, count in unflattened_items:
            arg_dict = {key: value for key, value in discrete_variables.items()}
            arg_dict["y"] = count
            new_struct = hl.Struct(**arg_dict)
            res.append(new_struct)
        return res


class StatBin(Stat):

    def __init__(self, start, end, bins):
        self.start = start
        self.end = end
        self.bins = bins

    def make_agg(self, mapping):
        discrete_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys()
                              if not is_continuous_type(mapping[aes_key].dtype)}
        return hl.agg.group_by(hl.struct(**discrete_variables), hl.agg.hist(mapping["x"], self.start, self.end, self.bins))

    def listify(self, agg_result):
        items = list(agg_result.items())
        x_edges = items[0][1].bin_edges
        num_edges = len(x_edges)
        data_rows = []
        for key, hist in items:
            y_values = hist.bin_freq
            for i, x in enumerate(x_edges[:num_edges - 1]):
                x_value = x
                data_rows.append(hl.Struct(x=x_value, y=y_values[i], **key))
        return data_rows


class GeomPoint(Geom):

    def __init__(self, aes, color=None):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        def plot_group(data, color=None):
            scatter_args = {
                "x": [element["x"] for element in data],
                "y": [element["y"] for element in data],
                "mode": "markers",
                "marker_color": color if color is not None else [element["color"] for element in data]
            }

            if "color_legend" in data[0]:
                scatter_args["name"] = data[0]["color_legend"]

            if "size" in parent.aes or "size" in self.aes:
                scatter_args["marker_size"] = [element["size"] for element in data]
            if "tooltip" in parent.aes or "tooltip" in self.aes:
                scatter_args["hovertext"] = [element["tooltip"] for element in data]
            fig_so_far.add_scatter(**scatter_args)

        if self.color is not None:
            plot_group(agg_result, self.color)
        elif "color" in parent.aes or "color" in self.aes:
            groups = set([element["group"] for element in agg_result])
            for group in groups:
                just_one_group = [element for element in agg_result if element["group"] == group]
                plot_group(just_one_group)
        else:
            plot_group(agg_result, "black")

    def get_stat(self):
        return StatIdentity()


def geom_point(mapping=aes(), *, color=None):
    return GeomPoint(mapping, color=color)


class GeomLine(GeomLineBasic):

    def __init__(self, aes, color=None):
        super().__init__(aes, color)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        super().apply_to_fig(parent, agg_result, fig_so_far)

    def get_stat(self):
        return StatIdentity()


def geom_line(mapping=aes(), *, color=None):
    return GeomLine(mapping, color=color)


class GeomText(Geom):

    def __init__(self, aes, color=None):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        def plot_group(data, color=None):
            scatter_args = {
                "x": [element["x"] for element in data],
                "y": [element["y"] for element in data],
                "text": [element["label"] for element in data],
                "mode": "text",
                "textfont_color": color
            }

            if "color_legend" in data[0]:
                scatter_args["name"] = data[0]["color_legend"]

            if "tooltip" in parent.aes or "tooltip" in self.aes:
                scatter_args["hovertext"] = [element["tooltip"] for element in data]

            if "size" in parent.aes or "size" in self.aes:
                scatter_args["marker_size"] = [element["size"] for element in data]
            fig_so_far.add_scatter(**scatter_args)

        if self.color is not None:
            plot_group(agg_result, self.color)
        elif "color" in parent.aes or "color" in self.aes:
            groups = set([element["group"] for element in agg_result])
            for group in groups:
                just_one_group = [element for element in agg_result if element["group"] == group]
                plot_group(just_one_group, just_one_group[0]["color"])
        else:
            plot_group(agg_result, "black")

    def get_stat(self):
        return StatIdentity()


def geom_text(mapping=aes(), *, color=None):
    return GeomText(mapping, color=color)


class GeomBar(Geom):

    def __init__(self, aes, fill=None, color=None, position="stack", size=None):
        super().__init__(aes)
        self.fill = fill
        self.color = color
        self.position = position
        self.size = size

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        def plot_group(data):
            if self.fill is None:
                if "fill" in data[0]:
                    fill = [element["fill"] for element in data]
                else:
                    fill = "black"
            else:
                fill = self.fill

            bar_args = {
                "x": [element["x"] for element in data],
                "y": [element["y"] for element in data],
                "marker_color": fill
            }
            if "color_legend" in data[0]:
                bar_args["name"] = data[0]["color_legend"]

            if self.color is None and "color" in data[0]:
                bar_args["marker_line_color"] = [element["color"] for element in data]
            elif self.color is not None:
                bar_args["marker_line_color"] = self.color


            if self.size is not None:
                bar_args["marker_line_width"] = self.size

            fig_so_far.add_bar(**bar_args)

        groups = set([element["group"] for element in agg_result])
        for group in groups:
            just_one_group = [element for element in agg_result if element["group"] == group]
            plot_group(just_one_group)

        ggplot_to_plotly = {'dodge': 'group', 'stack': 'stack'}
        fig_so_far.update_layout(barmode=ggplot_to_plotly[self.position])

    def get_stat(self):
        return StatCount()


def geom_bar(mapping=aes(), *, fill=None, color=None, position="stack", size=None):
    return GeomBar(mapping, fill=fill, color=color, position=position, size=size)


class GeomHistogram(Geom):

    def __init__(self, aes, min_bin=0, max_bin=100, bins=30, fill=None, color=None, position='stack', size=None):
        super().__init__(aes)
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.bins = bins
        self.fill = fill
        self.color = color
        self.position = position
        self.size = size

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        bin_width = (self.max_bin - self.min_bin) / self.bins

        def plot_group(data, num_groups):
            x = []

            for element in data:
                left_x = element.x
                if self.position == "dodge":
                    group = element.group
                    center_x = left_x + bin_width * (2 * group + 1) / (2 * num_groups)

                elif self.position == "stack":
                    center_x = left_x + bin_width / 2

                x.append(center_x)

            if self.fill is None:
                if "fill" in data[0]:
                    fill = [element["fill"] for element in data]
                else:
                    fill = "black"
            else:
                fill = self.fill

            hist_args = {
                "x": x,
                "y": [element["y"] for element in data],
                "marker_color": fill
            }

            if self.color is None and "color" in data[0]:
                hist_args["marker_line_color"] = [element["color"] for element in data]
            elif self.color is not None:
                hist_args["marker_line_color"] = self.color

            if self.size is not None:
                hist_args["marker_line_width"] = self.size

            width = bin_width if self.position == 'stack' else bin_width / num_groups
            hist_args["width"] = [width] * len(data)

            fig_so_far.add_bar(**hist_args)

        groups = set([element["group"] for element in agg_result])
        for group in groups:
            just_one_group = [element for element in agg_result if element["group"] == group]
            plot_group(just_one_group, len(groups))

        ggplot_to_plotly = {'dodge': 'group', 'stack': 'stack'}
        fig_so_far.update_layout(barmode=ggplot_to_plotly[self.position])

    def get_stat(self):
        return StatBin(self.min_bin, self.max_bin, self.bins)


def geom_histogram(mapping=aes(), min_bin=None, max_bin=None, bins=30, *, fill=None, color=None, position='stack',
                   size=None):
    assert(min_bin is not None)
    assert(max_bin is not None)
    return GeomHistogram(mapping, min_bin, max_bin, bins, fill, color, position, size)


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


def geom_hline(yintercept, *, linetype="solid", color=None):
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


def geom_vline(xintercept, *, linetype="solid", color=None):
    return GeomVLine(xintercept, linetype=linetype, color=color)


class GeomTile(Geom):

    def __init__(self, aes):
        self.aes = aes

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        def plot_group(data):
            for idx, row in enumerate(data):
                x_center = row['x']
                y_center = row['y']
                width = row['width']
                height = row['height']
                shape_args = {
                    "type": "rect",
                    "x0": x_center - width / 2,
                    "y0": y_center - height / 2,
                    "x1": x_center + width / 2,
                    "y1": y_center + height / 2,
                    "fillcolor": "black" if "fill" not in row else row["fill"],
                    "opacity": row.get('alpha', 1.0)
                }
                if "color" in row:
                    shape_args["line_color"] = row["color"]
                fig_so_far.add_shape(**shape_args)

        groups = set([element["group"] for element in agg_result])
        for group in groups:
            just_one_group = [element for element in agg_result if element["group"] == group]
            plot_group(just_one_group)

    def get_stat(self):
        return StatIdentity()


def geom_tile(mapping=aes()):
    return GeomTile(mapping)


class GeomFunction(GeomLineBasic):
    def __init__(self, aes, fun, color):
        super().__init__(aes, color)
        self.fun = fun

    def apply_to_fig(self, parent, agg_result, fig_so_far):
        super().apply_to_fig(parent, agg_result, fig_so_far)

    def get_stat(self):
        return StatFunction(self.fun)


def geom_func(mapping=aes(), fun=None, color=None):
    return GeomFunction(mapping, fun=fun, color=color)
