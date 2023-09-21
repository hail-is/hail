from typing import Dict, Any, Optional
import abc
import numpy as np
import plotly.graph_objects as go

from .aes import aes
from .stats import StatCount, StatIdentity, StatBin, StatNone, StatFunction, StatCDF
from .utils import bar_position_plotly_to_gg, linetype_plotly_to_gg


class FigureAttribute(abc.ABC):
    pass


class Geom(FigureAttribute):

    def __init__(self, aes):
        self.aes = aes

    @abc.abstractmethod
    def apply_to_fig(self, agg_result, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
        """Add this geometry to the figure and indicate if this geometry demands a static figure."""
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
                legend_cache[trace_args["name"]] = {}


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

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):

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

    aes_to_plotly = {
        "color": "marker_color",
        "size": "marker_size",
        "tooltip": "hovertext",
        "alpha": "marker_opacity",
        "shape": "marker_symbol",
    }

    aes_defaults = {
        "color": "black",
        "shape": "circle",
    }

    aes_legend_groups = {
        "color",
        "shape",
    }

    def __init__(self, aes, color=None, size=None, alpha=None, shape=None):
        super().__init__(aes)
        self.color = color
        self.size = size
        self.alpha = alpha
        self.shape = shape

    def _map_to_plotly(self, mapping) -> Dict[str, Any]:
        plotly_kwargs = {self.aes_to_plotly[k]: v for k, v in mapping.items()}
        if 'tooltip' in mapping:
            plotly_kwargs['hoverinfo'] = 'text'
        return plotly_kwargs

    def _get_aes_value(self, df, aes_name):
        if getattr(self, aes_name, None) is not None:
            return getattr(self, aes_name)
        if df.attrs.get(aes_name) is not None:
            return df.attrs[aes_name]
        if df.get(aes_name) is not None:
            return df[aes_name]
        return self.aes_defaults.get(aes_name, None)

    def _get_aes_values(self, df):
        values = {}
        for aes_name in self.aes_to_plotly:
            value = self._get_aes_value(df, aes_name)
            if value is not None:
                values[aes_name] = value
        return values

    def _add_trace(self, fig_so_far: go.Figure, df, facet_row, facet_col, values, legend: Optional[str] = None):
        fig_so_far.add_scatter(
            **{
                **{
                    "x": df.x,
                    "y": df.y,
                    "mode": "markers",
                    "row": facet_row,
                    "col": facet_col,
                    **(
                        {"showlegend": False}
                        if legend is None else
                        {"name": legend, "showlegend": True}
                    )
                },
                **self._map_to_plotly(values)
            }
        )

    def _add_legend(self, fig_so_far: go.Figure, aes_name, category, value):
        fig_so_far.add_scatter(
            **{
                **{
                    "x": [None],
                    "y": [None],
                    "mode": "markers",
                    "name": category,
                    "showlegend": True,
                    "legendgroup": aes_name,
                    "legendgrouptitle_text": aes_name,
                },
                **self._map_to_plotly({**self.aes_defaults, aes_name: value})
            }
        )

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
        traces = []
        legends = {}
        for df in grouped_data:
            values = self._get_aes_values(df)
            trace_categories = []
            for aes_name in self.aes_legend_groups:
                category = self._get_aes_value(df, f"{aes_name}_legend")
                if category is not None:
                    trace_categories.append(category)
                legends[aes_name] = ({
                    **legends.get(aes_name, {}),
                    category: values[aes_name]
                })
            traces.append([fig_so_far, df, facet_row, facet_col, values, trace_categories])

        non_empty_legend_groups = [
            legend_group for legend_group in legends.values()
            if len(legend_group) > 1 or (len(legend_group) == 1 and list(legend_group.keys())[0] is not None)
        ]
        dummy_legend = is_faceted or len(non_empty_legend_groups) >= 2

        if dummy_legend:
            for trace in traces:
                self._add_trace(*trace[:-1])
            for aes_name, legend_group in legends.items():
                prev = legend_cache.get(aes_name, {})
                for category, value in legend_group.items():
                    if category is not None and prev.get(category, None) is None:
                        self._add_legend(fig_so_far, aes_name, category, value)
                legend_cache[aes_name] = {**prev, **legend_group}
        else:
            main_categories = non_empty_legend_groups[0].keys() if len(non_empty_legend_groups) == 1 else None
            for trace in traces:
                trace_categories = trace[-1]
                if main_categories is not None:
                    trace[-1] = [category for category in trace_categories if category in main_categories][0]
                elif len(trace_categories) == 1:
                    trace[-1] = [trace_categories][0]
                else:
                    trace[-1] = "trace1"
            for trace in traces:
                self._add_trace(*trace)

    def get_stat(self):
        return StatIdentity()


def geom_point(mapping=aes(), *, color=None, size=None, alpha=None, shape=None):
    """Create a scatter plot.

    Supported aesthetics: ``x``, ``y``, ``color``, ``alpha``, ``tooltip``, ``shape``

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomPoint(mapping, color=color, size=size, alpha=alpha, shape=shape)


class GeomLine(GeomLineBasic):

    def __init__(self, aes, color=None):
        super().__init__(aes, color)
        self.color = color

    def apply_to_fig(self, agg_result, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
        return super().apply_to_fig(agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache, is_faceted)

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

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
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

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
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

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
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

# Computes the maximum entropy distribution whose cdf is within +- e of the
# staircase-shaped cdf encoded by min_x, max_x, x, y.
#
# x is an array of n x-coordinates between min_x and max_x, and y is an array
# of (n+1) y-coordinates between 0 and 1, both sorted. Together they encode a
# staircase-shaped cdf.
# For example, if min_x = 1, max_x=4, x=[2], y=[.2, .6], then the cdf is the
# staircase tracing the points
# (1, 0) - (1, .2) - (2, .2) - (2, .6) - (4, .6) - (4, 1)
#
# Now consider the set of all possible cdfs within +-e of the one above. In
# other words, shift the staircase both up and down by e, capping above and
# below at 1 and 0, and consider all possible cdfs that lie in between. The
# distribution with maximum entropy whose cdf is between the two staircases
# is the one whose cdf is the graph constructed as follows: tie a rubber band
# to the points (min_x, 0) and (max_x, 1), place the middle between the two
# staircases, and let it contract. In other words, it will be the shortest
# path between the staircases.
#
# It's easy to see this path must be piecewise linear, and the points where the
# slopes change will be either
# * bending up at a point of the form (x[i], y[i]+e), or
# * bending down at a point of the form (x[i], y[i+1]-e)
#
# Returns (new_y, keep).
# keep is the array of indices i at which the piecewise linear max-ent cdf
# changes slope, as described in the previous paragraph.
# new_y is an array the same length as x. For each i in keep, new_y[i] is the
# y coordinate of the point on the max-ent cdf.
def _max_entropy_cdf(min_x, max_x, x, y, e):
    def point_on_bound(i, upper):
        if i == len(x):
            return max_x, 1
        else:
            yi = y[i] + e if upper else y[i+1] - e
            return x[i], yi

    # Result variables:
    new_y = np.full_like(x, 0.0, dtype=np.float64)
    keep = np.full_like(x, False, dtype=np.bool_)

    # State variables:
    # (fx, fy) is most recently fixed point on max-ent cdf
    fx, fy = min_x, 0
    li, ui = 0, 0
    j = 1

    def slope_from_fixed(i, upper):
        xi, yi = point_on_bound(i, upper)
        return (yi - fy) / (xi - fx)

    def fix_point_on_result(i, upper):
        nonlocal fx, fy, new_y, keep
        xi, yi = point_on_bound(i, upper)
        fx, fy = xi, yi
        new_y[i] = fy
        keep[i] = True

    min_slope = slope_from_fixed(li, upper=False)
    max_slope = slope_from_fixed(ui, upper=True)

    # Consider a line l from (fx, fy) to (x[j], y?). As we increase y?, l first
    # bumps into the upper staircase at (x[ui], y[ui] + e), and as we decrease
    # y?, l first bumps into the lower staircase at (x[li], y[li+1] - e).
    # We track the min and max slopes l can have while staying between the
    # staircases, as well as the points li and ui where the line must bend if
    # forced too high or too low.

    while True:
        lower_slope = slope_from_fixed(j, upper=False)
        upper_slope = slope_from_fixed(j, upper=True)
        if upper_slope < min_slope:
            # Line must bend down at x[li]. We know the max-entropy cdf passes
            # through this point, so record it in new_y, keep.
            # This becomes the new fixed point, and we must restart the scan
            # from there.
            fix_point_on_result(li, upper=False)
            j = li + 1
            if j >= len(x):
                break
            li, ui = j, j
            min_slope = slope_from_fixed(li, upper=False)
            max_slope = slope_from_fixed(ui, upper=True)
            j += 1
            continue
        elif lower_slope > max_slope:
            # Line must bend up at x[ui]. We know the max-entropy cdf passes
            # through this point, so record it in new_y, keep.
            # This becomes the new fixed point, and we must restart the scan
            # from there.
            fix_point_on_result(ui, upper=True)
            j = ui + 1
            if j >= len(x):
                break
            li, ui = j, j
            min_slope = slope_from_fixed(li, upper=False)
            max_slope = slope_from_fixed(ui, upper=True)
            j += 1
            continue
        if j >= len(x):
            break
        if upper_slope < max_slope:
            ui = j
            max_slope = upper_slope
        if lower_slope > min_slope:
            li = j
            min_slope = lower_slope
        j += 1
    return new_y, keep


class GeomDensity(Geom):
    aes_to_arg = {
        "fill": ("marker_color", "black"),
        "color": ("marker_line_color", None),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, k=1000, smoothing=0.5, fill=None, color=None, alpha=None, smoothed=False):
        super().__init__(aes)
        self.k = k
        self.smoothing = smoothing
        self.fill = fill
        self.color = color
        self.alpha = alpha
        self.smoothed = smoothed

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
        from hail.expr.functions import _error_from_cdf_python
        def plot_group(df, idx):
            data = df.attrs['data']

            if self.smoothed:
                n = data['ranks'][-1]
                weights = np.diff(data['ranks'][1:-1])
                min = data['values'][0]
                max = data['values'][-1]
                values = np.array(data['values'][1:-1])
                slope = 1 / (max - min)

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
            else:
                confidence = 5

                y = np.array(data['ranks'][1:-1]) / data['ranks'][-1]
                x = np.array(data['values'][1:-1])
                min_x = data['values'][0]
                max_x = data['values'][-1]
                err = _error_from_cdf_python(data, 10 ** (-confidence), all_quantiles=True)

                new_y, keep = _max_entropy_cdf(min_x, max_x, x, y, err)
                slopes = np.diff([0, *new_y[keep], 1]) / np.diff([min_x, *x[keep], max_x])

                left = np.concatenate([[min_x], x[keep]])
                right = np.concatenate([x[keep], [max_x]])
                widths = right - left

                trace_args = {
                    "x": [min_x, *x[keep]],
                    "y": slopes,
                    "row": facet_row,
                    "col": facet_col,
                    "width": widths,
                    "offset": 0
                }

                self._add_aesthetics_to_trace_args(trace_args, df)
                self._update_legend_trace_args(trace_args, legend_cache)

                fig_so_far.add_bar(**trace_args)

        for idx, group_df in enumerate(grouped_data):
            plot_group(group_df, idx)

    def get_stat(self):
        return StatCDF(self.k)


def geom_density(mapping=aes(), *, k=1000, smoothing=0.5, fill=None, color=None, alpha=None, smoothed=False):
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
    smoothed: `boolean`
        If true, attempts to fit a smooth kernel density estimator.
        If false, uses a custom method do generate a variable width histogram
        directly from the approx_cdf results.

    Returns
    -------
    :class:`FigureAttribute`
        The geom to be applied.
    """
    return GeomDensity(mapping, k, smoothing, fill, color, alpha, smoothed)


class GeomHLine(Geom):

    def __init__(self, yintercept, linetype="solid", color=None):
        self.yintercept = yintercept
        self.aes = aes()
        self.linetype = linetype
        self.color = color

    def apply_to_fig(self, agg_result, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
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

    def apply_to_fig(self, agg_result, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
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

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
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

    def apply_to_fig(self, agg_result, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
        return super().apply_to_fig(agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache, is_faceted)

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

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
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

    def apply_to_fig(self, grouped_data, fig_so_far: go.Figure, precomputed, facet_row, facet_col, legend_cache, is_faceted: bool):
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
