import abc

import numpy as np

from .aes import aes
from .stats import StatCount, StatIdentity, StatBin, StatNone, StatFunction, StatCDF
from .utils import bar_position_plotly_to_gg, linetype_plotly_to_gg


class FigureAttribute(abc.ABC):
    pass


class Geom(FigureAttribute):

    def __init__(self, aes):
        self.aes = aes

    @abc.abstractmethod
    def apply_to_fig(self, parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        pass

    @abc.abstractmethod
    def get_stat(self):
        pass

    def _add_aesthetics_to_trace_args(self, trace_args, df):
        for aes_name, (plotly_name, default) in self.aes_to_arg.items():
            if hasattr(self, aes_name) and getattr(self, aes_name) is not None:
                trace_args[plotly_name] = getattr(self, aes_name)
            elif aes_name in df.attrs:
                trace_args[plotly_name] = df.attrs[aes_name]
            elif aes_name in df.columns:
                trace_args[plotly_name] = df[aes_name]
            elif default is not None:
                trace_args[plotly_name] = default

    def _update_legend_trace_args(self, trace_args, legend_cache):
        if "name" in trace_args:
            trace_args["legendgroup"] = trace_args["name"]
            if trace_args["name"] in legend_cache:
                trace_args["showlegend"] = False
            else:
                trace_args["showlegend"] = True
                legend_cache.add(trace_args["name"])


class GeomLineBasic(Geom):
    aes_to_arg = {
        "color": ("line_color", "black"),
        "size": ("marker_size", None),
        "tooltip": ("hovertext", None),
        "color_legend": ("name", None)
    }

    def __init__(self, aes, color):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):

        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "mode": "lines",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

    @abc.abstractmethod
    def get_stat(self):
        return ...


class GeomPoint(Geom):

    aes_to_arg = {
        "color": ("marker_color", "black"),
        "size": ("marker_size", None),
        "tooltip": ("hovertext", None),
        "color_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, color=None, size=None, alpha=None):
        super().__init__(aes)
        self.color = color
        self.size = size
        self.alpha = alpha

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "mode": "markers",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


def geom_point(mapping=aes(), *, color=None, size=None, alpha=None):
    """Create a scatter plot.

    Supported aesthetics: ``x``, ``y``, ``color``, ``alpha``, ``tooltip``

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomPoint(mapping, color=color, size=size, alpha=alpha)


class GeomLine(GeomLineBasic):

    def __init__(self, aes, color=None):
        super().__init__(aes, color)
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        super().apply_to_fig(parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache)

    def get_stat(self):
        return StatIdentity()


def geom_line(mapping=aes(), *, color=None, size=None, alpha=None):
    """Create a line plot.

    Supported aesthetics: ``x``, ``y``, ``color``, ``tooltip``

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomLine(mapping, color=color)


class GeomText(Geom):
    aes_to_arg = {
        "color": ("textfont_color", "black"),
        "size": ("marker_size", None),
        "tooltip": ("hovertext", None),
        "color_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, color=None, size=None, alpha=None):
        super().__init__(aes)
        self.color = color
        self.size = size
        self.alpha = alpha

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "text": df.label,
                "mode": "text",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


def geom_text(mapping=aes(), *, color=None, size=None, alpha=None):
    """Create a scatter plot where each point is text from the ``text`` aesthetic.

    Supported aesthetics: ``x``, ``y``, ``label``, ``color``, ``tooltip``

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomText(mapping, color=color, size=size, alpha=alpha)


class GeomBar(Geom):

    aes_to_arg = {
        "fill": ("marker_color", "black"),
        "color": ("marker_line_color", None),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, fill=None, color=None, alpha=None, position="stack", size=None, stat=None):
        super().__init__(aes)
        self.fill = fill
        self.color = color
        self.position = position
        self.size = size
        self.alpha = alpha

        if stat is None:
            stat = StatCount()
        self.stat = stat

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_bar(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

        fig_so_far.update_layout(barmode=bar_position_plotly_to_gg(self.position))

    def get_stat(self):
        return self.stat


def geom_bar(mapping=aes(), *, fill=None, color=None, alpha=None, position="stack", size=None):
    """Create a bar chart that counts occurrences of the various values of the ``x`` aesthetic.

    Supported aesthetics: ``x``, ``color``, ``fill``, ``weight``

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """

    return GeomBar(mapping, fill=fill, color=color, alpha=alpha, position=position, size=size)


def geom_col(mapping=aes(), *, fill=None, color=None, alpha=None, position="stack", size=None):
    """Create a bar chart that uses bar heights specified in y aesthetic.

    Supported aesthetics: ``x``, ``y``, ``color``, ``fill``

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomBar(mapping, stat=StatIdentity(), fill=fill, color=color, alpha=alpha, position=position, size=size)


class GeomHistogram(Geom):
    aes_to_arg = {
        "fill": ("marker_color", "black"),
        "color": ("marker_line_color", None),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, min_val=None, max_val=None, bins=None, fill=None, color=None, alpha=None, position='stack', size=None):
        super().__init__(aes)
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins
        self.fill = fill
        self.color = color
        self.alpha = alpha
        self.position = position
        self.size = size

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        min_val = self.min_val if self.min_val is not None else precomputed.min_val
        max_val = self.max_val if self.max_val is not None else precomputed.max_val
        # This assumes it doesn't really make sense to use another stat for geom_histogram
        bins = self.bins if self.bins is not None else self.get_stat().DEFAULT_BINS
        bin_width = (max_val - min_val) / bins

        num_groups = len(grouped_data)

        def plot_group(df, idx):
            left_xs = df.x

            if self.position == "dodge":
                x = left_xs + bin_width * (2 * idx + 1) / (2 * num_groups)
                bar_width = bin_width / num_groups

            elif self.position in {"stack", "identity"}:
                x = left_xs + bin_width / 2
                bar_width = bin_width
            else:
                raise ValueError(f"Histogram does not support position = {self.position}")

            right_xs = left_xs + bin_width

            trace_args = {
                "x": x,
                "y": df.y,
                "row": facet_row,
                "col": facet_col,
                "customdata": list(zip(left_xs, right_xs)),
                "width": bar_width,
                "hovertemplate":
                    "Range: [%{customdata[0]:.3f}-%{customdata[1]:.3f})<br>"
                    "Count: %{y}<br>"
                    "<extra></extra>",
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_bar(**trace_args)

        for idx, group_df in enumerate(grouped_data):
            plot_group(group_df, idx)

        fig_so_far.update_layout(barmode=bar_position_plotly_to_gg(self.position))

    def get_stat(self):
        return StatBin(self.min_val, self.max_val, self.bins)


def geom_histogram(mapping=aes(), *, min_val=None, max_val=None, bins=None, fill=None, color=None, alpha=None, position='stack',
                   size=None):
    """Creates a histogram.

    Note: this function currently does not support same interface as R's ggplot.

    Supported aesthetics: ``x``, ``color``, ``fill``

    Parameters
    ----------
    mapping: :class:`Aesthetic`
        Any aesthetics specific to this geom.
    min_val: `int` or `float`
        Minimum value to include in histogram
    max_val: `int` or `float`
        Maximum value to include in histogram
    bins: `int`
        Number of bins to plot. 30 by default.
    fill:
        A single fill color for all bars of histogram, overrides ``fill`` aesthetic.
    color:
        A single outline color for all bars of histogram, overrides ``color`` aesthetic.
    alpha: `float`
        A measure of transparency between 0 and 1.
    position: :class:`str`
        Tells how to deal with different groups of data at same point. Options are "stack" and "dodge".

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomHistogram(mapping, min_val=min_val, max_val=max_val, bins=bins, fill=fill, color=color, alpha=alpha, position=position, size=size)


class GeomDensity(Geom):
    aes_to_arg = {
        "fill": ("marker_color", "black"),
        "color": ("marker_line_color", None),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, k=1000, smoothing=0.5, fill=None, color=None, alpha=None):
        super().__init__(aes)
        self.k = k
        self.smoothing = smoothing
        self.fill = fill
        self.color = color
        self.alpha = alpha

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df, idx):
            slope = 1.0 / (df.attrs['max'] - df.attrs['min'])
            n = df.attrs['n']
            min = df.attrs['min']
            max = df.attrs['max']
            values = df.value.to_numpy()
            weights = df.weight.to_numpy()

            def f(x, prev):
                inv_scale = (np.sqrt(n * slope) / self.smoothing) * np.sqrt(prev / weights)
                diff = x[:, np.newaxis] - values
                grid = (3 / (4 * n)) * weights * np.maximum(0, inv_scale - np.power(diff, 2) * np.power(inv_scale, 3))
                return np.sum(grid, axis=1)

            round1 = f(values, np.full(len(values), slope))
            x_d = np.linspace(min, max, 1000)
            final = f(x_d, round1)

            trace_args = {
                "x": x_d,
                "y": final,
                "mode": "lines",
                "fill": "tozeroy",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for idx, group_df in enumerate(grouped_data):
            plot_group(group_df, idx)

    def get_stat(self):
        return StatCDF(self.k)


def geom_density(mapping=aes(), *, k=1000, smoothing=0.5, fill=None, color=None, alpha=None):
    """Creates a smoothed density plot.

    This method uses the `hl.agg.approx_cdf` aggregator to compute a sketch
    of the distribution of the values of `x`. It then uses an ad hoc method to
    estimate a smoothed pdf consistent with that cdf.

    Note: this function currently does not support same interface as R's ggplot.

    Supported aesthetics: ``x``, ``color``, ``fill``

    Parameters
    ----------
    mapping: :class:`Aesthetic`
        Any aesthetics specific to this geom.
    k: `int`
        Passed to the `approx_cdf` aggregator. The size of the aggregator scales
        linearly with `k`. The default value of `1000` is likely sufficient for
        most uses.
    smoothing: `float`
        Controls the amount of smoothing applied.
    fill:
        A single fill color for all density plots, overrides ``fill`` aesthetic.
    color:
        A single line color for all density plots, overrides ``color`` aesthetic.
    alpha: `float`
        A measure of transparency between 0 and 1.

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomDensity(mapping, k, smoothing, fill, color, alpha)


class GeomHLine(Geom):

    def __init__(self, yintercept, linetype="solid", color=None):
        self.yintercept = yintercept
        self.aes = aes()
        self.linetype = linetype
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        line_attributes = {
            "y": self.yintercept,
            "line_dash": linetype_plotly_to_gg(self.linetype)
        }
        if self.color is not None:
            line_attributes["line_color"] = self.color

        fig_so_far.add_hline(**line_attributes)

    def get_stat(self):
        return StatNone()


def geom_hline(yintercept, *, linetype="solid", color=None):
    """Plots a horizontal line at ``yintercept``.


    Parameters
    ----------
    yintercept : :class:`float`
        Location to draw line.
    linetype : :class:`str`
        Type of line to draw. Choose from "solid", "dashed", "dotted", "longdash", "dotdash".
    color : :class:`str`
        Color of line to draw, black by default.

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomHLine(yintercept, linetype=linetype, color=color)


class GeomVLine(Geom):

    def __init__(self, xintercept, linetype="solid", color=None):
        self.xintercept = xintercept
        self.aes = aes()
        self.linetype = linetype
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        line_attributes = {
            "x": self.xintercept,
            "line_dash": linetype_plotly_to_gg(self.linetype)
        }
        if self.color is not None:
            line_attributes["line_color"] = self.color

        fig_so_far.add_vline(**line_attributes)

    def get_stat(self):
        return StatNone()


def geom_vline(xintercept, *, linetype="solid", color=None):
    """Plots a vertical line at ``xintercept``.


    Parameters
    ----------
    xintercept : :class:`float`
        Location to draw line.
    linetype : :class:`str`
        Type of line to draw. Choose from "solid", "dashed", "dotted", "longdash", "dotdash".
    color : :class:`str`
        Color of line to draw, black by default.

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomVLine(xintercept, linetype=linetype, color=color)


class GeomTile(Geom):

    def __init__(self, aes):
        self.aes = aes

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):

            for idx, row in df.iterrows():
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
                    "row": facet_row,
                    "col": facet_col,
                    "opacity": row.get('alpha', 1.0)
                }
                if "fill" in df.attrs:
                    shape_args["fillcolor"] = df.attrs["fill"]
                elif "fill" in row:
                    shape_args["fillcolor"] = row["fill"]
                else:
                    shape_args["fillcolor"] = "black"
                fig_so_far.add_shape(**shape_args)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


def geom_tile(mapping=aes()):
    return GeomTile(mapping)


class GeomFunction(GeomLineBasic):
    def __init__(self, aes, fun, color):
        super().__init__(aes, color)
        self.fun = fun

    def apply_to_fig(self, parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        super().apply_to_fig(parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache)

    def get_stat(self):
        return StatFunction(self.fun)


def geom_func(mapping=aes(), fun=None, color=None):
    return GeomFunction(mapping, fun=fun, color=color)


class GeomArea(Geom):
    aes_to_arg = {
        "fill": ("fillcolor", "black"),
        "color": ("line_color", "rgba(0, 0, 0, 0)"),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None)
    }

    def __init__(self, aes, fill, color):
        super().__init__(aes)
        self.fill = fill
        self.color = color

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "row": facet_row,
                "col": facet_col,
                "fill": 'tozeroy'
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


def geom_area(mapping=aes(), fill=None, color=None):
    """Creates a line plot with the area between the line and the x-axis filled in.

    Supported aesthetics: ``x``, ``y``, ``fill``, ``color``, ``tooltip``

    Parameters
    ----------
    mapping: :class:`Aesthetic`
        Any aesthetics specific to this geom.
    fill:
        Color of fill to draw, black by default. Overrides ``fill`` aesthetic.
    color:
        Color of line to draw outlining non x-axis facing side, none by default. Overrides ``color`` aesthetic.

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomArea(mapping, fill=fill, color=color)


class GeomRibbon(Geom):
    aes_to_arg = {
        "fill": ("fillcolor", "black"),
        "color": ("line_color", "rgba(0, 0, 0, 0)"),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None)
    }

    def __init__(self, aes, fill, color):
        super().__init__(aes)
        self.fill = fill
        self.color = color

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):

            trace_args_bottom = {
                "x": df.x,
                "y": df.ymin,
                "row": facet_row,
                "col": facet_col,
                "mode": "lines",
                "showlegend": False
            }
            self._add_aesthetics_to_trace_args(trace_args_bottom, df)
            self._update_legend_trace_args(trace_args_bottom, legend_cache)

            trace_args_top = {
                "x": df.x,
                "y": df.ymax,
                "row": facet_row,
                "col": facet_col,
                "mode": "lines",
                "fill": 'tonexty'
            }
            self._add_aesthetics_to_trace_args(trace_args_top, df)
            self._update_legend_trace_args(trace_args_top, legend_cache)

            fig_so_far.add_scatter(**trace_args_bottom)
            fig_so_far.add_scatter(**trace_args_top)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


def geom_ribbon(mapping=aes(), fill=None, color=None):
    """Creates filled in area between two lines specified by x, ymin, and ymax

    Supported aesthetics: ``x``, ``ymin``, ``ymax``, ``color``, ``fill``, ``tooltip``

    Parameters
    ----------
    mapping: :class:`Aesthetic`
        Any aesthetics specific to this geom.
    fill:
        Color of fill to draw, black by default. Overrides ``fill`` aesthetic.
    color:
        Color of line to draw outlining both side, none by default. Overrides ``color`` aesthetic.

    :return:
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomRibbon(mapping, fill=fill, color=color)
