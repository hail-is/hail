import math

import collections
import numpy as np
import pandas as pd
import bokeh
import bokeh.io
import bokeh.models
import warnings
from bokeh.models import (
    HoverTool,
    ColorBar,
    LogTicker,
    LogColorMapper,
    LinearColorMapper,
    CategoricalColorMapper,
    ColumnDataSource,
    BasicTicker,
    Plot,
    CDSView,
    GroupFilter,
    IntersectionFilter,
    Legend,
    LegendItem,
    Renderer,
    CustomJS,
    Select,
    Column,
    Span,
    DataRange1d,
    Slope,
    Label,
    ColorMapper,
    GridPlot,
)
import bokeh.plotting
import bokeh.palettes
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.layouts import gridplot

from hail.expr import aggregators
from hail.expr.expressions import (
    Expression,
    NumericExpression,
    StringExpression,
    LocusExpression,
    Int32Expression,
    Int64Expression,
    Float32Expression,
    Float64Expression,
    expr_numeric,
    expr_float64,
    expr_any,
    expr_locus,
    expr_str,
    raise_unless_row_indexed,
)
from hail.expr.functions import _error_from_cdf_python
from hail.typecheck import typecheck, oneof, nullable, sized_tupleof, numeric, sequenceof, dictof
from hail import Table, MatrixTable
from hail.utils.struct import Struct
from hail.utils.java import warning
from typing import List, Tuple, Dict, Union, Callable, Optional, Sequence, Any, Set
import hail

palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def output_notebook():
    """Configure the Bokeh output state to generate output in notebook
    cells when :func:`bokeh.io.show` is called.  Calls
    :func:`bokeh.io.output_notebook`.

    """
    bokeh.io.output_notebook()


def show(obj, interact=None):
    """Immediately display a Bokeh object or application.  Calls
    :func:`bokeh.io.show`.

    Parameters
    ----------
    obj
        A Bokeh object to display.
    interact
        A handle returned by a plotting method with `interactive=True`.
    """
    if interact is None:
        bokeh.io.show(obj)
    else:
        handle = bokeh.io.show(obj, notebook_handle=True)
        interact(handle)


def cdf(data, k=350, legend=None, title=None, normalize=True, log=False) -> figure:
    """Create a cumulative density plot.

    Parameters
    ----------
    data : :class:`.Struct` or :class:`.Float64Expression`
        Sequence of data to plot.
    k : int
        Accuracy parameter (passed to :func:`~.approx_cdf`).
    legend : str
        Label of data on the x-axis.
    title : str
        Title of the histogram.
    normalize: bool
        Whether or not the cumulative data should be normalized.
    log: bool
        Whether or not the y-axis should be of type log.

    Returns
    -------
    :class:`bokeh.plotting.figure`
    """
    if isinstance(data, Expression):
        if data._indices is None:
            raise ValueError('Invalid input')
        agg_f = data._aggregation_method()
        data = agg_f(aggregators.approx_cdf(data, k))

    if legend is None:
        legend = ""

    if normalize:
        y_axis_label = 'Quantile'
    else:
        y_axis_label = 'Rank'
    if log:
        y_axis_type = 'log'
    else:
        y_axis_type = 'linear'
    p = figure(
        title=title,
        x_axis_label=legend,
        y_axis_label=y_axis_label,
        y_axis_type=y_axis_type,
        width=600,
        height=400,
        background_fill_color='#EEEEEE',
        tools='xpan,xwheel_zoom,reset,save',
        active_scroll='xwheel_zoom',
    )
    p.add_tools(HoverTool(tooltips=[("value", "$x"), ("rank", "@top")], mode='vline'))

    ranks = np.array(data.ranks)
    values = np.array(data['values'])
    if normalize:
        ranks = ranks / ranks[-1]

    # invisible, there to support tooltips
    p.quad(top=ranks[1:-1], bottom=ranks[1:-1], left=values[:-1], right=values[1:], fill_alpha=0, line_alpha=0)
    p.step(x=[*values, values[-1]], y=ranks, line_width=2, line_color='black', legend_label=legend)
    return p


def pdf(
    data, k=1000, confidence=5, legend=None, title=None, log=False, interactive=False
) -> Union[figure, Tuple[figure, Callable]]:
    if isinstance(data, Expression):
        if data._indices is None:
            raise ValueError('Invalid input')
        agg_f = data._aggregation_method()
        data = agg_f(aggregators.approx_cdf(data, k))

    if legend is None:
        legend = ""

    y_axis_label = 'Frequency'
    if log:
        y_axis_type = 'log'
    else:
        y_axis_type = 'linear'
    fig = figure(
        title=title,
        x_axis_label=legend,
        y_axis_label=y_axis_label,
        y_axis_type=y_axis_type,
        width=600,
        height=400,
        tools='xpan,xwheel_zoom,reset,save',
        active_scroll='xwheel_zoom',
        background_fill_color='#EEEEEE',
    )

    y = np.array(data['ranks'][1:-1]) / data['ranks'][-1]
    x = np.array(data['values'][1:-1])
    min_x = data['values'][0]
    max_x = data['values'][-1]
    err = _error_from_cdf_python(data, 10 ** (-confidence), all_quantiles=True)

    new_y, keep = _max_entropy_cdf(min_x, max_x, x, y, err)
    slopes = np.diff([0, *new_y[keep], 1]) / np.diff([min_x, *x[keep], max_x])
    if log:
        plot = fig.step(x=[min_x, *x[keep], max_x], y=[*slopes, slopes[-1]], mode='after')
    else:
        plot = fig.quad(left=[min_x, *x[keep]], right=[*x[keep], max_x], bottom=0, top=slopes, legend_label=legend)

    if interactive:

        def mk_interact(handle):
            def update(confidence=confidence):
                err = _error_from_cdf_python(data, 10 ** (-confidence), all_quantiles=True) / 1.8
                new_y, keep = _max_entropy_cdf(min_x, max_x, x, y, err)
                slopes = np.diff([0, *new_y[keep], 1]) / np.diff([min_x, *x[keep], max_x])
                if log:
                    new_data = {'x': [min_x, *x[keep], max_x], 'y': [*slopes, slopes[-1]]}
                else:
                    new_data = {
                        'left': [min_x, *x[keep]],
                        'right': [*x[keep], max_x],
                        'bottom': np.full(len(slopes), 0),
                        'top': slopes,
                    }
                plot.data_source.data = new_data
                bokeh.io.push_notebook(handle=handle)

            from ipywidgets import interact

            interact(update, confidence=(1, 10, 0.01))

        return fig, mk_interact
    else:
        return fig


def _max_entropy_cdf(min_x, max_x, x, y, e):
    def compare(x1, y1, x2, y2):
        return x1 * y2 - x2 * y1

    new_y = np.full_like(x, 0.0, dtype=np.float64)
    keep = np.full_like(x, False, dtype=np.bool_)

    fx = min_x  # fixed x
    fy = 0  # fixed y
    li = 0  # index of lower slope
    ui = 0  # index of upper slope
    ldx = x[li] - fx
    udx = x[ui] - fx
    ldy = y[li + 1] - e - fy
    udy = y[ui] + e - fy
    j = 1
    while ui < len(x) and li < len(x):
        if j == len(x):
            ub = 1
            lb = 1
            xj = max_x
        else:
            ub = y[j] + e
            lb = y[j + 1] - e
            xj = x[j]
        dx = xj - fx
        judy = ub - fy
        jldy = lb - fy
        if compare(ldx, ldy, dx, judy) < 0:
            # line must bend down at j
            fx = x[li]
            fy = y[li + 1] - e
            new_y[li] = fy
            keep[li] = True
            j = li + 1
            if j >= len(x):
                break
            li = j
            ldx = x[li] - fx
            ldy = y[li + 1] - e - fy
            ui = j
            udx = x[ui] - fx
            udy = y[ui] + e - fy
            j += 1
            continue
        elif compare(udx, udy, dx, jldy) > 0:
            # line must bend up at j
            fx = x[ui]
            fy = y[ui] + e
            new_y[ui] = fy
            keep[ui] = True
            j = ui + 1
            if j >= len(x):
                break
            li = j
            ldx = x[li] - fx
            ldy = y[li + 1] - e - fy
            ui = j
            udx = x[ui] - fx
            udy = y[ui] + e - fy
            j += 1
            continue
        if j >= len(x):
            break
        if compare(udx, udy, dx, judy) < 0:
            ui = j
            udx = x[ui] - fx
            udy = y[ui] + e - fy
        if compare(ldx, ldy, dx, jldy) > 0:
            li = j
            ldx = x[li] - fx
            ldy = y[li + 1] - e - fy
        j += 1
    return new_y, keep


def smoothed_pdf(
    data, k=350, smoothing=0.5, legend=None, title=None, log=False, interactive=False, figure=None
) -> Union[figure, Tuple[figure, Callable]]:
    """Create a density plot.

    Parameters
    ----------
    data : :class:`.Struct` or :class:`.Float64Expression`
        Sequence of data to plot.
    k : int
        Accuracy parameter.
    smoothing : float
        Degree of smoothing.
    legend : str
        Label of data on the x-axis.
    title : str
        Title of the histogram.
    log : bool
        Plot the log10 of the bin counts.
    interactive : bool
        If `True`, return a handle to pass to :func:`bokeh.io.show`.
    figure : :class:`bokeh.plotting.figure`
        If not None, add density plot to figure. Otherwise, create a new figure.

    Returns
    -------
    :class:`bokeh.plotting.figure`
    """
    if isinstance(data, Expression):
        if data._indices is None:
            raise ValueError('Invalid input')
        agg_f = data._aggregation_method()
        data = agg_f(aggregators.approx_cdf(data, k))

    if legend is None:
        legend = ""

    y_axis_label = 'Frequency'
    if log:
        y_axis_type = 'log'
    else:
        y_axis_type = 'linear'

    if figure is None:
        p = bokeh.plotting.figure(
            title=title,
            x_axis_label=legend,
            y_axis_label=y_axis_label,
            y_axis_type=y_axis_type,
            width=600,
            height=400,
            tools='xpan,xwheel_zoom,reset,save',
            active_scroll='xwheel_zoom',
            background_fill_color='#EEEEEE',
        )
    else:
        p = figure

    n = data['ranks'][-1]
    weights = np.diff(data['ranks'][1:-1])
    min = data['values'][0]
    max = data['values'][-1]
    values = np.array(data['values'][1:-1])
    slope = 1 / (max - min)

    def f(x, prev, smoothing=smoothing):
        inv_scale = (np.sqrt(n * slope) / smoothing) * np.sqrt(prev / weights)
        diff = x[:, np.newaxis] - values
        grid = (3 / (4 * n)) * weights * np.maximum(0, inv_scale - np.power(diff, 2) * np.power(inv_scale, 3))
        return np.sum(grid, axis=1)

    round1 = f(values, np.full(len(values), slope))
    x_d = np.linspace(min, max, 1000)
    final = f(x_d, round1)

    line = p.line(x_d, final, line_width=2, line_color='black', legend_label=legend)

    if interactive:

        def mk_interact(handle):
            def update(smoothing=smoothing):
                final = f(x_d, round1, smoothing)
                line.data_source.data = {'x': x_d, 'y': final}
                bokeh.io.push_notebook(handle=handle)

            from ipywidgets import interact

            interact(update, smoothing=(0.02, 0.8, 0.005))

        return p, mk_interact
    else:
        return p


@typecheck(
    data=oneof(Struct, expr_float64),
    range=nullable(sized_tupleof(numeric, numeric)),
    bins=int,
    legend=nullable(str),
    title=nullable(str),
    log=bool,
    interactive=bool,
)
def histogram(
    data, range=None, bins=50, legend=None, title=None, log=False, interactive=False
) -> Union[figure, Tuple[figure, Callable]]:
    """Create a histogram.

    Notes
    -----
    `data` can be a :class:`.Float64Expression`, or the result of the :func:`~.aggregators.hist`
    or :func:`~.aggregators.approx_cdf` aggregators.

    Parameters
    ----------
    data : :class:`.Struct` or :class:`.Float64Expression`
        Sequence of data to plot.
    range : Tuple[float]
        Range of x values in the histogram.
    bins : int
        Number of bins in the histogram.
    legend : str
        Label of data on the x-axis.
    title : str
        Title of the histogram.
    log : bool
        Plot the log10 of the bin counts.

    Returns
    -------
    :class:`bokeh.plotting.figure`
    """
    if isinstance(data, Expression):
        if data._indices.source is not None:
            if interactive:
                raise ValueError("'interactive' flag can only be used on data from 'approx_cdf'.")
            agg_f = data._aggregation_method()
            if range is not None:
                start = range[0]
                end = range[1]
            else:
                finite_data = hail.bind(lambda x: hail.case().when(hail.is_finite(x), x).or_missing(), data)
                start, end = agg_f((aggregators.min(finite_data), aggregators.max(finite_data)))
                if start is None and end is None:
                    raise ValueError("'data' contains no values that are defined and finite")
            data = agg_f(aggregators.hist(data, start, end, bins))
        else:
            raise ValueError('Invalid input')
    elif 'values' in data:
        cdf = data
        hist, edges = np.histogram(cdf['values'], bins=bins, weights=np.diff(cdf.ranks), density=True)
        data = Struct(bin_freq=hist, bin_edges=edges, n_larger=0, n_smaller=0)

    if legend is None:
        legend = ""

    if log:
        bin_freq = []
        count_problems = 0
        for x in data.bin_freq:
            if x == 0.0:
                count_problems += 1
                bin_freq.append(x)
            else:
                bin_freq.append(math.log10(x))

        if count_problems > 0:
            warning(
                f"There were {count_problems} bins with height 0, those cannot be log transformed and were left as 0s."
            )

        changes = {
            "bin_freq": bin_freq,
            "n_larger": math.log10(data.n_larger) if data.n_larger > 0.0 else data.n_larger,
            "n_smaller": math.log10(data.n_smaller) if data.n_smaller > 0.0 else data.n_smaller,
        }
        data = data.annotate(**changes)
        y_axis_label = 'log10 Frequency'
    else:
        y_axis_label = 'Frequency'

    x_span = data.bin_edges[-1] - data.bin_edges[0]
    x_start = data.bin_edges[0] - 0.05 * x_span
    x_end = data.bin_edges[-1] + 0.05 * x_span
    p = figure(
        title=title,
        x_axis_label=legend,
        y_axis_label=y_axis_label,
        background_fill_color='#EEEEEE',
        x_range=(x_start, x_end),
    )
    q = p.quad(
        bottom=0,
        top=data.bin_freq,
        left=data.bin_edges[:-1],
        right=data.bin_edges[1:],
        legend_label=legend,
        line_color='black',
    )
    if data.n_larger > 0:
        p.quad(
            bottom=0,
            top=data.n_larger,
            left=data.bin_edges[-1],
            right=(data.bin_edges[-1] + (data.bin_edges[1] - data.bin_edges[0])),
            line_color='black',
            fill_color='green',
            legend_label='Outliers Above',
        )
    if data.n_smaller > 0:
        p.quad(
            bottom=0,
            top=data.n_smaller,
            left=data.bin_edges[0] - (data.bin_edges[1] - data.bin_edges[0]),
            right=data.bin_edges[0],
            line_color='black',
            fill_color='red',
            legend_label='Outliers Below',
        )
    if interactive:

        def mk_interact(handle):
            def update(bins=bins, phase=0):
                if phase > 0 and phase < 1:
                    bins = bins + 1
                    delta = (cdf['values'][-1] - cdf['values'][0]) / bins
                    edges = np.linspace(cdf['values'][0] - (1 - phase) * delta, cdf['values'][-1] + phase * delta, bins)
                else:
                    edges = np.linspace(cdf['values'][0], cdf['values'][-1], bins)
                hist, edges = np.histogram(cdf['values'], bins=edges, weights=np.diff(cdf.ranks), density=True)
                new_data = {'top': hist, 'left': edges[:-1], 'right': edges[1:], 'bottom': np.full(len(hist), 0)}
                q.data_source.data = new_data
                bokeh.io.push_notebook(handle=handle)

            from ipywidgets import interact

            interact(update, bins=(0, 5 * bins), phase=(0, 1, 0.01))

        return p, mk_interact
    else:
        return p


@typecheck(
    data=oneof(Struct, expr_float64),
    range=nullable(sized_tupleof(numeric, numeric)),
    bins=int,
    legend=nullable(str),
    title=nullable(str),
    normalize=bool,
    log=bool,
)
def cumulative_histogram(data, range=None, bins=50, legend=None, title=None, normalize=True, log=False) -> figure:
    """Create a cumulative histogram.

    Parameters
    ----------
    data : :class:`.Struct` or :class:`.Float64Expression`
        Sequence of data to plot.
    range : Tuple[float]
        Range of x values in the histogram.
    bins : int
        Number of bins in the histogram.
    legend : str
        Label of data on the x-axis.
    title : str
        Title of the histogram.
    normalize: bool
        Whether or not the cumulative data should be normalized.
    log: bool
        Whether or not the y-axis should be of type log.

    Returns
    -------
    :class:`bokeh.plotting.figure`
    """
    if isinstance(data, Expression):
        if data._indices.source is not None:
            agg_f = data._aggregation_method()
            if range is not None:
                start = range[0]
                end = range[1]
            else:
                start, end = agg_f((aggregators.min(data), aggregators.max(data)))
            data = agg_f(aggregators.hist(data, start, end, bins))
        else:
            raise ValueError('Invalid input')

    if legend is None:
        legend = ""

    cumulative_data = np.cumsum(data.bin_freq) + data.n_smaller
    np.append(cumulative_data, [cumulative_data[-1] + data.n_larger])
    num_data_points = max(cumulative_data)

    if normalize:
        cumulative_data = cumulative_data / num_data_points
    if title is not None:
        title = f'{title} ({num_data_points:,} data points)'
    if log:
        p = figure(
            title=title,
            x_axis_label=legend,
            y_axis_label='Frequency',
            background_fill_color='#EEEEEE',
            y_axis_type='log',
        )
    else:
        p = figure(title=title, x_axis_label=legend, y_axis_label='Frequency', background_fill_color='#EEEEEE')
    p.line(data.bin_edges[:-1], cumulative_data, line_color='#036564', line_width=3)
    return p


@typecheck(p=figure, font_size=str)
def set_font_size(p, font_size: str = '12pt'):
    """Set most of the font sizes in a bokeh figure

    Parameters
    ----------
    p : :class:`bokeh.plotting.figure`
        Input figure.
    font_size : str
        String of font size in points (e.g. '12pt').

    Returns
    -------
    :class:`bokeh.plotting.figure`
    """
    p.legend.label_text_font_size = font_size
    p.xaxis.axis_label_text_font_size = font_size
    p.yaxis.axis_label_text_font_size = font_size
    p.xaxis.major_label_text_font_size = font_size
    p.yaxis.major_label_text_font_size = font_size
    if hasattr(p.title, 'text_font_size'):
        p.title.text_font_size = font_size
    if hasattr(p.xaxis, 'group_text_font_size'):
        p.xaxis.group_text_font_size = font_size
    return p


@typecheck(
    x=expr_numeric,
    y=expr_numeric,
    bins=oneof(int, sequenceof(int)),
    range=nullable(sized_tupleof(nullable(sized_tupleof(numeric, numeric)), nullable(sized_tupleof(numeric, numeric)))),
    title=nullable(str),
    width=int,
    height=int,
    colors=sequenceof(str),
    log=bool,
)
def histogram2d(
    x: NumericExpression,
    y: NumericExpression,
    bins: int = 40,
    range: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    width: int = 600,
    height: int = 600,
    colors: Sequence[str] = bokeh.palettes.all_palettes['Blues'][7][::-1],
    log: bool = False,
) -> figure:
    """Plot a two-dimensional histogram.

    ``x`` and ``y`` must both be a :class:`.NumericExpression` from the same :class:`.Table`.

    If ``x_range`` or ``y_range`` are not provided, the function will do a pass through the data to determine
    min and max of each variable.

    Examples
    --------

    >>> ht = hail.utils.range_table(1000).annotate(x=hail.rand_norm(), y=hail.rand_norm())
    >>> p_hist = hail.plot.histogram2d(ht.x, ht.y)

    >>> ht = hail.utils.range_table(1000).annotate(x=hail.rand_norm(), y=hail.rand_norm())
    >>> p_hist = hail.plot.histogram2d(ht.x, ht.y, bins=10, range=((0, 1), None))

    Parameters
    ----------
    x : :class:`.NumericExpression`
        Expression for x-axis (from a Hail table).
    y : :class:`.NumericExpression`
        Expression for y-axis (from the same Hail table as ``x``).
    bins : int or [int, int]
        The bin specification:
        -   If int, the number of bins for the two dimensions (nx = ny = bins).
        -   If [int, int], the number of bins in each dimension (nx, ny = bins).
        The default value is 40.
    range : None or ((float, float), (float, float))
        The leftmost and rightmost edges of the bins along each dimension:
        ((xmin, xmax), (ymin, ymax)). All values outside of this range will be considered outliers
        and not tallied in the histogram. If this value is None, or either of the inner lists is None,
        the range will be computed from the data.
    width : int
        Plot width (default 600px).
    height : int
        Plot height (default 600px).
    title : str
        Title of the plot.
    colors : Sequence[str]
        List of colors (hex codes, or strings as described
        `here <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__). Compatible with one of the many
        built-in palettes available `here <https://bokeh.pydata.org/en/latest/docs/reference/palettes.html>`__.
    log : bool
        Plot the log10 of the bin counts.

    Returns
    -------
    :class:`bokeh.plotting.figure`
    """
    data = _generate_hist2d_data(x, y, bins, range).to_pandas()

    # Use python prettier float -> str function
    data['x'] = data['x'].apply(lambda e: str(float(e)))
    data['y'] = data['y'].apply(lambda e: str(float(e)))

    mapper: ColorMapper
    if log:
        mapper = LogColorMapper(palette=colors, low=data.c.min(), high=data.c.max())
    else:
        mapper = LinearColorMapper(palette=colors, low=data.c.min(), high=data.c.max())

    x_axis = sorted(set(data.x), key=lambda z: float(z))
    y_axis = sorted(set(data.y), key=lambda z: float(z))
    p = figure(
        title=title,
        x_range=x_axis,
        y_range=y_axis,
        x_axis_location="above",
        width=width,
        height=height,
        tools="hover,save,pan,box_zoom,reset,wheel_zoom",
        toolbar_location='below',
    )

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    import math

    p.xaxis.major_label_orientation = math.pi / 3

    p.rect(
        x='x', y='y', width=1, height=1, source=data, fill_color={'field': 'c', 'transform': mapper}, line_color=None
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=LogTicker(desired_num_ticks=len(colors)) if log else BasicTicker(desired_num_ticks=len(colors)),
        label_standoff=12 if log else 6,
        border_line_color=None,
        location=(0, 0),
    )
    p.add_layout(color_bar, 'right')

    hovertool = p.select_one(HoverTool)
    assert hovertool is not None
    hovertool.tooltips = [
        ('x', '@x'),
        (
            'y',
            '@y',
        ),
        ('count', '@c'),
    ]

    return p


@typecheck(
    x=expr_numeric,
    y=expr_numeric,
    bins=oneof(int, sequenceof(int)),
    range=nullable(sized_tupleof(nullable(sized_tupleof(numeric, numeric)), nullable(sized_tupleof(numeric, numeric)))),
)
def _generate_hist2d_data(x, y, bins, range):
    source = x._indices.source
    y_source = y._indices.source
    if source is None or y_source is None:
        raise ValueError("histogram_2d expects two expressions of 'Table', found scalar expression")
    if isinstance(source, hail.MatrixTable):
        raise ValueError("histogram_2d requires source to be Table, not MatrixTable")
    if source != y_source:
        raise ValueError(f"histogram_2d expects two expressions from the same 'Table', found {source} and {y_source}")
    raise_unless_row_indexed('histogram_2d', x)
    raise_unless_row_indexed('histogram_2d', y)
    if isinstance(bins, int):
        x_bins = y_bins = bins
    else:
        x_bins, y_bins = bins
    if range is None:
        x_range = y_range = None
    else:
        x_range, y_range = range
    if x_range is None or y_range is None:
        warning('At least one range was not defined in histogram_2d. Doing two passes...')
        ranges = source.aggregate(hail.struct(x_stats=hail.agg.stats(x), y_stats=hail.agg.stats(y)))
        if x_range is None:
            x_range = (ranges.x_stats.min, ranges.x_stats.max)
        if y_range is None:
            y_range = (ranges.y_stats.min, ranges.y_stats.max)
    else:
        warning(
            'If x_range or y_range are specified in histogram_2d, and there are points '
            'outside of these ranges, they will not be plotted'
        )
    x_range = list(map(float, x_range))
    y_range = list(map(float, y_range))
    x_spacing = (x_range[1] - x_range[0]) / x_bins
    y_spacing = (y_range[1] - y_range[0]) / y_bins

    def frange(start, stop, step):
        from itertools import count, takewhile

        return takewhile(lambda x: x <= stop, count(start, step))

    x_levels = hail.literal(list(frange(x_range[0], x_range[1], x_spacing))[::-1])
    y_levels = hail.literal(list(frange(y_range[0], y_range[1], y_spacing))[::-1])
    grouped_ht = source.group_by(
        x=hail.str(x_levels.find(lambda w: x >= w)), y=hail.str(y_levels.find(lambda w: y >= w))
    ).aggregate(c=hail.agg.count())
    data = grouped_ht.filter(
        hail.is_defined(grouped_ht.x)
        & (grouped_ht.x != str(x_range[1]))
        & hail.is_defined(grouped_ht.y)
        & (grouped_ht.y != str(y_range[1]))
    )
    return data


def _collect_scatter_plot_data(
    x: Tuple[str, NumericExpression],
    y: Tuple[str, NumericExpression],
    fields: Optional[Dict[str, Expression]] = None,
    n_divisions: Optional[int] = None,
    missing_label: str = 'NA',
) -> pd.DataFrame:
    expressions = dict()
    if fields is not None:
        expressions.update({
            k: hail.or_else(v, missing_label) if isinstance(v, StringExpression) else v for k, v in fields.items()
        })

    if n_divisions is None:
        collect_expr = hail.struct(**dict((k, v) for k, v in (x, y)), **expressions)
        plot_data = [point for point in collect_expr.collect() if point[x[0]] is not None and point[y[0]] is not None]
        source_pd = pd.DataFrame(plot_data)
    else:
        # FIXME: remove the type conversion logic if/when downsample supports continuous values for labels
        # Save all numeric types to cast in DataFrame
        numeric_expr = {k: 'int32' for k, v in expressions.items() if isinstance(v, Int32Expression)}
        numeric_expr.update({k: 'int64' for k, v in expressions.items() if isinstance(v, Int64Expression)})
        numeric_expr.update({k: 'float32' for k, v in expressions.items() if isinstance(v, Float32Expression)})
        numeric_expr.update({k: 'float64' for k, v in expressions.items() if isinstance(v, Float64Expression)})

        # Cast non-string types to string
        expressions = {k: hail.str(v) if not isinstance(v, StringExpression) else v for k, v in expressions.items()}

        agg_f = x[1]._aggregation_method()
        res = agg_f(
            hail.agg.downsample(
                x[1], y[1], label=list(expressions.values()) if expressions else None, n_divisions=n_divisions
            )
        )
        source_pd = pd.DataFrame([
            dict(
                **{x[0]: point[0], y[0]: point[1]},
                **(dict(zip(expressions, point[2])) if point[2] is not None else {}),
            )
            for point in res
        ])
        source_pd = source_pd.astype(numeric_expr, copy=False)

    return source_pd


def _get_categorical_palette(factors: List[str]) -> ColorMapper:
    n = max(3, len(factors))
    _palette: Sequence[str]
    if n < len(palette):
        _palette = palette
    elif n < 21:
        from bokeh.palettes import Category20

        _palette = Category20[n]
    else:
        from bokeh.palettes import viridis

        _palette = viridis(n)

    return CategoricalColorMapper(factors=factors, palette=_palette)


def _get_scatter_plot_elements(
    sp: Plot,
    source_pd: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_cols: List[str],
    colors: Optional[Dict[str, ColorMapper]] = None,
    size: int = 4,
    hover_cols: Optional[Set[str]] = None,
) -> Union[
    Tuple[Plot, Dict[str, List[LegendItem]], Legend, ColorBar, Dict[str, ColorMapper], List[Renderer]],
    Tuple[Plot, None, None, None, None, None],
]:
    if not source_pd.shape[0]:
        print("WARN: No data to plot.")
        return sp, None, None, None, None, None

    possible_tooltips = [(x_col, f'@{x_col}'), (y_col, f'@{y_col}')] + [
        (c, f'@{c}') for c in source_pd.columns if c not in [x_col, y_col]
    ]

    if hover_cols is not None:
        possible_tooltips = [x for x in possible_tooltips if x[0] in hover_cols]
    sp.tools.append(HoverTool(tooltips=possible_tooltips))

    cds = ColumnDataSource(source_pd)

    if not label_cols:
        sp.circle(x_col, y_col, source=cds, size=size)
        return sp, None, None, None, None, None
    continuous_cols = [
        col
        for col in label_cols
        if (str(source_pd.dtypes[col]).startswith('float') or str(source_pd.dtypes[col]).startswith('int'))
    ]
    factor_cols = [col for col in label_cols if col not in continuous_cols]

    #  Assign color mappers to columns
    if colors is None:
        colors = {}
    color_mappers: Dict[str, ColorMapper] = {}

    for col in continuous_cols:
        low = np.nanmin(source_pd[col])
        if np.isnan(low):
            low = 0
            high = 0
        else:
            high = np.nanmax(source_pd[col])
        color_mappers[col] = colors[col] if col in colors else LinearColorMapper(palette='Magma256', low=low, high=high)

    for col in factor_cols:
        if col in colors:
            color_mappers[col] = colors[col]
        else:
            factors = list(set(source_pd[col]))
            color_mappers[col] = _get_categorical_palette(factors)

    # Create initial glyphs
    initial_col = label_cols[0]
    initial_mapper = color_mappers[initial_col]
    legend_items: Dict[str, List[LegendItem]] = {}

    if not factor_cols:
        all_renderers = [sp.circle(x_col, y_col, color=transform(initial_col, initial_mapper), source=cds, size=size)]

    else:
        all_renderers = []
        legend_items_by_key_by_factor = {col: collections.defaultdict(list) for col in factor_cols}
        for key in source_pd.groupby(factor_cols).groups.keys():
            key = key if len(factor_cols) > 1 else [key]
            cds_view = CDSView(
                filter=IntersectionFilter(
                    operands=[GroupFilter(column_name=factor_cols[i], group=key[i]) for i in range(0, len(factor_cols))]
                )
            )
            renderer = sp.circle(
                x_col, y_col, color=transform(initial_col, initial_mapper), source=cds, view=cds_view, size=size
            )
            all_renderers.append(renderer)
            for i in range(0, len(factor_cols)):
                legend_items_by_key_by_factor[factor_cols[i]][key[i]].append(renderer)

        legend_items = {
            factor: [LegendItem(label=key, renderers=renderers) for key, renderers in key_renderers.items()]
            for factor, key_renderers in legend_items_by_key_by_factor.items()
        }

    # Add legend / color bar
    legend = (
        Legend(visible=False, click_policy='hide', orientation='vertical')
        if initial_col not in factor_cols
        else Legend(items=legend_items[initial_col], click_policy='hide', orientation='vertical')
    )
    color_bar = ColorBar(color_mapper=color_mappers[initial_col])
    if initial_col not in continuous_cols:
        color_bar.visible = False
    sp.add_layout(legend, 'left')
    sp.add_layout(color_bar, 'left')

    return sp, legend_items, legend, color_bar, color_mappers, all_renderers


def _downsampling_factor(fname: str, n_divisions: Optional[int], collect_all: Optional[bool]) -> Optional[int]:
    if collect_all is not None:
        warnings.warn(f'{fname}: `collect_all` has been deprecated. Use `n_divisions` instead.')
        if n_divisions is not None and collect_all is not None:
            raise ValueError('At most one of `collect_all` or `n_divisions` must be specified.')

    n_divisions = None if collect_all else n_divisions

    if n_divisions is not None and n_divisions < 1:
        raise ValueError('`n_divisions` must be a positive whole number or `None`')

    return n_divisions


@typecheck(
    x=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
    y=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
    label=nullable(oneof(dictof(str, expr_any), expr_any)),
    title=nullable(str),
    xlabel=nullable(str),
    ylabel=nullable(str),
    size=int,
    legend=bool,
    hover_fields=nullable(dictof(str, expr_any)),
    colors=nullable(oneof(bokeh.models.mappers.ColorMapper, dictof(str, bokeh.models.mappers.ColorMapper))),
    width=int,
    height=int,
    collect_all=nullable(bool),
    n_divisions=nullable(int),
    missing_label=str,
)
def scatter(
    x: Union[NumericExpression, Tuple[str, NumericExpression]],
    y: Union[NumericExpression, Tuple[str, NumericExpression]],
    label: Optional[Union[Expression, Dict[str, Expression]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    size: int = 4,
    legend: bool = True,
    hover_fields: Optional[Dict[str, Expression]] = None,
    colors: Optional[Union[ColorMapper, Dict[str, ColorMapper]]] = None,
    width: int = 800,
    height: int = 800,
    collect_all: Optional[bool] = None,
    n_divisions: Optional[int] = 500,
    missing_label: str = 'NA',
) -> Union[Plot, Column]:
    """Create an interactive scatter plot.

    ``x`` and ``y`` must both be either:
    - a :class:`.NumericExpression` from the same :class:`.Table`.
    - a tuple (str, :class:`.NumericExpression`) from the same :class:`.Table`. If passed as a tuple the first element is used as the hover label.

    If no label or a single label is provided, then returns :class:`bokeh.plotting.figure`
    Otherwise returns a :class:`bokeh.models.layouts.Column` containing:
    - a :class:`bokeh.models.widgets.inputs.Select` dropdown selection widget for labels
    - a :class:`bokeh.plotting.figure` containing the interactive scatter plot

    Points will be colored by one of the labels defined in the ``label`` using the color scheme defined in
    the corresponding entry of ``colors`` if provided (otherwise a default scheme is used). To specify your color
    mapper, check `the bokeh documentation <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__
    for CategoricalMapper for categorical labels, and for LinearColorMapper and LogColorMapper
    for continuous labels.
    For categorical labels, clicking on one of the items in the legend will hide/show all points with the corresponding label.
    Note that using many different labelling schemes in the same plots, particularly if those labels contain many
    different classes could slow down the plot interactions.

    Hovering on points will display their coordinates, labels and any additional fields specified in ``hover_fields``.

    Parameters
    ----------
    x : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
        List of x-values to be plotted.
    y : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
        List of y-values to be plotted.
    label : :class:`.Expression` or Dict[str, :class:`.Expression`]], optional
        Either a single expression (if a single label is desired), or a
        dictionary of label name -> label value for x and y values.
        Used to color each point w.r.t its label.
        When multiple labels are given, a dropdown will be displayed with the different options.
        Can be used with categorical or continuous expressions.
    title : str, optional
        Title of the scatterplot.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    size : int
        Size of markers in screen space units.
    legend: bool
        Whether or not to show the legend in the resulting figure.
    hover_fields : Dict[str, :class:`.Expression`], optional
        Extra fields to be displayed when hovering over a point on the plot.
    colors : :class:`bokeh.models.mappers.ColorMapper` or Dict[str, :class:`bokeh.models.mappers.ColorMapper`], optional
        If a single label is used, then this can be a color mapper, if multiple labels are used, then this should
        be a Dict of label name -> color mapper.
        Used to set colors for the labels defined using ``label``.
        If not used at all, or label names not appearing in this dict will be colored using a default color scheme.
    width: int
        Plot width
    height: int
        Plot height
    collect_all : bool, optional
        Deprecated. Use `n_divisions` instead.
    n_divisions : int, optional
        Factor by which to downsample (default value = 500).
        A lower input results in fewer output datapoints.
        Use `None` to collect all points.
    missing_label: str
        Label to use when a point is missing data for a categorical label

    Returns
    -------
    :class:`bokeh.models.Plot` if no label or a single label was given, otherwise :class:`bokeh.models.layouts.Column`
    """
    hover_fields = {} if hover_fields is None else hover_fields

    label_by_col: Dict[str, Expression]
    if label is None:
        label_by_col = {}
    elif isinstance(label, Expression):
        label_by_col = {'label': label}
    else:
        assert isinstance(label, dict)
        label_by_col = label

    if isinstance(colors, ColorMapper):
        colors_by_col = {'label': colors}
    else:
        colors_by_col = colors

    label_cols = list(label_by_col.keys())
    if isinstance(x, NumericExpression):
        _x = ('x', x)
    else:
        _x = x

    if isinstance(y, NumericExpression):
        _y = ('y', y)
    else:
        _y = y

    source_pd = _collect_scatter_plot_data(
        _x,
        _y,
        fields={**hover_fields, **label_by_col},
        n_divisions=_downsampling_factor('scatter', n_divisions, collect_all),
        missing_label=missing_label,
    )
    sp = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, height=height, width=width)
    sp, sp_legend_items, sp_legend, sp_color_bar, sp_color_mappers, sp_scatter_renderers = _get_scatter_plot_elements(
        sp, source_pd, _x[0], _y[0], label_cols, colors_by_col, size, hover_cols={'x', 'y'} | set(hover_fields)
    )

    if not legend:
        assert sp_legend is not None
        assert sp_color_bar is not None
        sp_legend.visible = False
        sp_color_bar.visible = False

    # If multiple labels, create JS call back selector
    if len(label_cols) > 1:
        callback_args: Dict[str, Any]
        callback_args = dict(color_mappers=sp_color_mappers, scatter_renderers=sp_scatter_renderers)
        callback_code = """
        for (var i = 0; i < scatter_renderers.length; i++){
            scatter_renderers[i].glyph.fill_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
            scatter_renderers[i].glyph.line_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
            scatter_renderers[i].visible = true
        }

        """

        if legend:
            callback_args.update(dict(legend_items=sp_legend_items, legend=sp_legend, color_bar=sp_color_bar))
            callback_code += """
        if (cb_obj.value in legend_items){
            legend.items=legend_items[cb_obj.value]
            legend.visible=true
            color_bar.visible=false
        }else{
            legend.visible=false
            color_bar.visible=true
        }

        """

        callback = CustomJS(args=callback_args, code=callback_code)
        select = Select(title="Color by", value=label_cols[0], options=label_cols)
        select.js_on_change('value', callback)
        return Column(children=[select, sp])

    return sp


@typecheck(
    x=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
    y=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
    label=nullable(oneof(dictof(str, expr_any), expr_any)),
    title=nullable(str),
    xlabel=nullable(str),
    ylabel=nullable(str),
    size=int,
    legend=bool,
    hover_fields=nullable(dictof(str, expr_any)),
    colors=nullable(oneof(bokeh.models.mappers.ColorMapper, dictof(str, bokeh.models.mappers.ColorMapper))),
    width=int,
    height=int,
    collect_all=nullable(bool),
    n_divisions=nullable(int),
    missing_label=str,
)
def joint_plot(
    x: Union[NumericExpression, Tuple[str, NumericExpression]],
    y: Union[NumericExpression, Tuple[str, NumericExpression]],
    label: Optional[Union[Expression, Dict[str, Expression]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    size: int = 4,
    legend: bool = True,
    hover_fields: Optional[Dict[str, StringExpression]] = None,
    colors: Optional[Union[ColorMapper, Dict[str, ColorMapper]]] = None,
    width: int = 800,
    height: int = 800,
    collect_all: Optional[bool] = None,
    n_divisions: Optional[int] = 500,
    missing_label: str = 'NA',
) -> GridPlot:
    """Create an interactive scatter plot with marginal densities on the side.

    ``x`` and ``y`` must both be either:
    - a :class:`.NumericExpression` from the same :class:`.Table`.
    - a tuple (str, :class:`.NumericExpression`) from the same :class:`.Table`. If passed as a tuple the first element is used as the hover label.

    This function returns a :class:`bokeh.models.layouts.Column` containing two :class:`figure.Row`:
    - The first row contains the X-axis marginal density and a selection widget if multiple entries are specified in the ``label``
    - The second row contains the scatter plot and the y-axis marginal density

    Points will be colored by one of the labels defined in the ``label`` using the color scheme defined in
    the corresponding entry of ``colors`` if provided (otherwise a default scheme is used). To specify your color
    mapper, check `the bokeh documentation <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__
    for CategoricalMapper for categorical labels, and for LinearColorMapper and LogColorMapper
    for continuous labels.
    For categorical labels, clicking on one of the items in the legend will hide/show all points with the corresponding label in the scatter plot.
    Note that using many different labelling schemes in the same plots, particularly if those labels contain many
    different classes could slow down the plot interactions.

    Hovering on points in the scatter plot displays their coordinates, labels and any additional fields specified in ``hover_fields``.

     Parameters
     ----------
     ----------
     x : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
         List of x-values to be plotted.
     y : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
         List of y-values to be plotted.
     label : :class:`.Expression` or Dict[str, :class:`.Expression`]], optional
         Either a single expression (if a single label is desired), or a
         dictionary of label name -> label value for x and y values.
         Used to color each point w.r.t its label.
         When multiple labels are given, a dropdown will be displayed with the different options.
         Can be used with categorical or continuous expressions.
     title : str, optional
         Title of the scatterplot.
     xlabel : str, optional
         X-axis label.
     ylabel : str, optional
         Y-axis label.
     size : int
         Size of markers in screen space units.
     legend: bool
         Whether or not to show the legend in the resulting figure.
     hover_fields : Dict[str, :class:`.Expression`], optional
         Extra fields to be displayed when hovering over a point on the plot.
     colors : :class:`bokeh.models.mappers.ColorMapper` or Dict[str, :class:`bokeh.models.mappers.ColorMapper`], optional
         If a single label is used, then this can be a color mapper, if multiple labels are used, then this should
         be a Dict of label name -> color mapper.
         Used to set colors for the labels defined using ``label``.
         If not used at all, or label names not appearing in this dict will be colored using a default color scheme.
     width: int
         Plot width
     height: int
         Plot height
     collect_all : bool, optional
         Deprecated. Use `n_divisions` instead.
     n_divisions : int, optional
         Factor by which to downsample (default value = 500).
         A lower input results in fewer output datapoints.
         Use `None` to collect all points.
     missing_label: str
         Label to use when a point is missing data for a categorical label


     Returns
     -------
     :class:`.GridPlot`
    """
    # Collect data
    hover_fields = {} if hover_fields is None else hover_fields

    label_by_col: Dict[str, Expression]
    if label is None:
        label_by_col = {}
    elif isinstance(label, Expression):
        label_by_col = {'label': label}
    else:
        assert isinstance(label, dict)
        label_by_col = label

    if isinstance(colors, ColorMapper):
        colors_by_col = {'label': colors}
    else:
        colors_by_col = colors
    if isinstance(x, NumericExpression):
        _x = ('x', x)
    else:
        _x = x

    if isinstance(y, NumericExpression):
        _y = ('y', y)
    else:
        _y = y

    label_cols = list(label_by_col.keys())
    source_pd = _collect_scatter_plot_data(
        _x,
        _y,
        fields={**hover_fields, **label_by_col},
        n_divisions=_downsampling_factor('join_plot', n_divisions, collect_all),
        missing_label=missing_label,
    )
    sp = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, height=height, width=width)
    sp, sp_legend_items, sp_legend, sp_color_bar, sp_color_mappers, sp_scatter_renderers = _get_scatter_plot_elements(
        sp, source_pd, _x[0], _y[0], label_cols, colors_by_col, size, hover_cols={'x', 'y'} | set(hover_fields)
    )

    continuous_cols = [
        col
        for col in label_cols
        if (str(source_pd.dtypes[col]).startswith('float') or str(source_pd.dtypes[col]).startswith('int'))
    ]
    factor_cols = [col for col in label_cols if col not in continuous_cols]

    # Density plots
    def get_density_plot_items(
        source_pd,
        data_col,
        p,
        x_axis,
        colors: Optional[Dict[str, ColorMapper]],
        continuous_cols: List[str],
        factor_cols: List[str],
    ):
        density_renderers = []
        max_densities = {}
        if not factor_cols or continuous_cols:
            dens, edges = np.histogram(source_pd[data_col], density=True)
            edges = edges[:-1]
            xy = (edges, dens) if x_axis else (dens, edges)
            cds = ColumnDataSource({'x': xy[0], 'y': xy[1]})
            line = p.line('x', 'y', source=cds)
            density_renderers.extend([(col, "", line) for col in continuous_cols])
            max_densities = {col: np.max(dens) for col in continuous_cols}

        for factor_col in factor_cols:
            assert colors is not None, (colors, factor_cols)
            factor_colors = colors.get(factor_col, _get_categorical_palette(list(set(source_pd[factor_col]))))
            factor_colors = dict(zip(factor_colors.factors, factor_colors.palette))
            density_data = (
                source_pd[[factor_col, data_col]]
                .groupby(factor_col)
                .apply(lambda df: np.histogram(df['x' if x_axis else 'y'], density=True))
            )
            for factor, (dens, edges) in density_data.iteritems():
                edges = edges[:-1]
                xy = (edges, dens) if x_axis else (dens, edges)
                cds = ColumnDataSource({'x': xy[0], 'y': xy[1]})
                density_renderers.append((
                    factor_col,
                    factor,
                    p.line('x', 'y', color=factor_colors.get(factor, 'gray'), source=cds),
                ))
                max_densities[factor_col] = np.max(list(dens) + [max_densities.get(factor_col, 0)])

        p.grid.visible = False
        p.outline_line_color = None
        return p, density_renderers, max_densities

    xp = figure(title=title, height=int(height / 3), width=width, x_range=sp.x_range)
    xp, x_renderers, x_max_densities = get_density_plot_items(
        source_pd,
        _x[0],
        xp,
        x_axis=True,
        colors=sp_color_mappers,
        continuous_cols=continuous_cols,
        factor_cols=factor_cols,
    )
    xp.xaxis.visible = False
    yp = figure(height=height, width=int(width / 3), y_range=sp.y_range)
    yp, y_renderers, y_max_densities = get_density_plot_items(
        source_pd,
        _y[0],
        yp,
        x_axis=False,
        colors=sp_color_mappers,
        continuous_cols=continuous_cols,
        factor_cols=factor_cols,
    )
    yp.yaxis.visible = False
    density_renderers = x_renderers + y_renderers
    first_row = [xp]

    if not legend:
        assert sp_legend is not None
        assert sp_color_bar is not None
        sp_legend.visible = False
        sp_color_bar.visible = False

    # If multiple labels, create JS call back selector
    if len(label_cols) > 1:
        for factor_col, _, renderer in density_renderers:
            renderer.visible = factor_col == label_cols[0]

        if label_cols[0] in factor_cols:
            xp.y_range.start = 0
            xp.y_range.end = x_max_densities[label_cols[0]]
            yp.x_range.start = 0
            yp.x_range.end = y_max_densities[label_cols[0]]

        callback_args: Dict[str, Any]
        callback_args = dict(
            scatter_renderers=sp_scatter_renderers,
            color_mappers=sp_color_mappers,
            density_renderers=x_renderers + y_renderers,
            x_range=xp.y_range,
            x_max_densities=x_max_densities,
            y_range=yp.x_range,
            y_max_densities=y_max_densities,
        )

        callback_code = """
                for (var i = 0; i < scatter_renderers.length; i++){
                    scatter_renderers[i].glyph.fill_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
                    scatter_renderers[i].glyph.line_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
                    scatter_renderers[i].visible = true
                }

                for (var i = 0; i < density_renderers.length; i++){
                    density_renderers[i][2].visible = density_renderers[i][0] == cb_obj.value
                }

                x_range.start = 0
                y_range.start = 0
                x_range.end = x_max_densities[cb_obj.value]
                y_range.end = y_max_densities[cb_obj.value]

                """

        if legend:
            callback_args.update(dict(legend_items=sp_legend_items, legend=sp_legend, color_bar=sp_color_bar))
            callback_code += """
                if (cb_obj.value in legend_items){
                    legend.items=legend_items[cb_obj.value]
                    legend.visible=true
                    color_bar.visible=false
                }else{
                    legend.visible=false
                    color_bar.visible=true
                }

                """

        callback = CustomJS(args=callback_args, code=callback_code)
        select = Select(title="Color by", value=label_cols[0], options=label_cols)
        select.js_on_change('value', callback)
        first_row.append(select)

    return gridplot([first_row, [sp, yp]])


@typecheck(
    pvals=expr_numeric,
    label=nullable(oneof(dictof(str, expr_any), expr_any)),
    title=nullable(str),
    xlabel=nullable(str),
    ylabel=nullable(str),
    size=int,
    legend=bool,
    hover_fields=nullable(dictof(str, expr_any)),
    colors=nullable(oneof(bokeh.models.mappers.ColorMapper, dictof(str, bokeh.models.mappers.ColorMapper))),
    width=int,
    height=int,
    collect_all=nullable(bool),
    n_divisions=nullable(int),
    missing_label=str,
)
def qq(
    pvals: NumericExpression,
    label: Optional[Union[Expression, Dict[str, Expression]]] = None,
    title: Optional[str] = 'Q-Q plot',
    xlabel: Optional[str] = 'Expected -log10(p)',
    ylabel: Optional[str] = 'Observed -log10(p)',
    size: int = 6,
    legend: bool = True,
    hover_fields: Optional[Dict[str, Expression]] = None,
    colors: Optional[Union[ColorMapper, Dict[str, ColorMapper]]] = None,
    width: int = 800,
    height: int = 800,
    collect_all: Optional[bool] = None,
    n_divisions: Optional[int] = 500,
    missing_label: str = 'NA',
) -> Union[figure, Column]:
    """Create a Quantile-Quantile plot. (https://en.wikipedia.org/wiki/Q-Q_plot)

    If no label or a single label is provided, then returns :class:`bokeh.plotting.figure`
    Otherwise returns a :class:`bokeh.models.layouts.Column` containing:
    - a :class:`bokeh.models.widgets.inputs.Select` dropdown selection widget for labels
    - a :class:`bokeh.plotting.figure` containing the interactive qq plot

    Points will be colored by one of the labels defined in the ``label`` using the color scheme defined in
    the corresponding entry of ``colors`` if provided (otherwise a default scheme is used). To specify your color
    mapper, check `the bokeh documentation <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__
    for CategoricalMapper for categorical labels, and for LinearColorMapper and LogColorMapper
    for continuous labels.
    For categorical labels, clicking on one of the items in the legend will hide/show all points with the corresponding label.
    Note that using many different labelling schemes in the same plots, particularly if those labels contain many
    different classes could slow down the plot interactions.

    Hovering on points will display their coordinates, labels and any additional fields specified in ``hover_fields``.

    Parameters
    ----------
    pvals : :class:`.NumericExpression`
        List of x-values to be plotted.
    label : :class:`.Expression` or Dict[str, :class:`.Expression`]]
        Either a single expression (if a single label is desired), or a
        dictionary of label name -> label value for x and y values.
        Used to color each point w.r.t its label.
        When multiple labels are given, a dropdown will be displayed with the different options.
        Can be used with categorical or continuous expressions.
    title : str, optional
        Title of the scatterplot.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    size : int
        Size of markers in screen space units.
    legend: bool
        Whether or not to show the legend in the resulting figure.
    hover_fields : Dict[str, :class:`.Expression`], optional
        Extra fields to be displayed when hovering over a point on the plot.
    colors : :class:`bokeh.models.mappers.ColorMapper` or Dict[str, :class:`bokeh.models.mappers.ColorMapper`], optional
        If a single label is used, then this can be a color mapper, if multiple labels are used, then this should
        be a Dict of label name -> color mapper.
        Used to set colors for the labels defined using ``label``.
        If not used at all, or label names not appearing in this dict will be colored using a default color scheme.
    width: int
        Plot width
    height: int
        Plot height
    collect_all : bool
        Deprecated. Use `n_divisions` instead.
    n_divisions : int, optional
        Factor by which to downsample (default value = 500).
        A lower input results in fewer output datapoints.
        Use `None` to collect all points.
    missing_label: str
        Label to use when a point is missing data for a categorical label

    Returns
    -------
    :class:`bokeh.plotting.figure` if no label or a single label was given, otherwise :class:`bokeh.models.layouts.Column`
    """
    hover_fields = {} if hover_fields is None else hover_fields
    label_by_col: Dict[str, Expression]
    if label is None:
        label_by_col = {}
    elif isinstance(label, Expression):
        label_by_col = {'label': label}
    else:
        assert isinstance(label, dict)
        label_by_col = label

    source = pvals._indices.source
    if isinstance(source, Table):
        ht = source.select(p_value=pvals, **hover_fields, **label_by_col)
    else:
        assert isinstance(source, MatrixTable)
        ht = source.select_rows(p_value=pvals, **hover_fields, **label_by_col).rows()
    ht = ht.key_by().select('p_value', *hover_fields, *label_by_col).key_by('p_value')
    n = ht.aggregate(aggregators.count(), _localize=False)
    ht = ht.annotate(observed_p=-hail.log10(ht['p_value']), expected_p=-hail.log10((hail.scan.count() + 1) / n))
    if 'p' not in hover_fields:
        hover_fields['p_value'] = ht['p_value']
    p = scatter(
        ht.expected_p,
        ht.observed_p,
        label={x: ht[x] for x in label_by_col},
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        size=size,
        legend=legend,
        hover_fields={x: ht[x] for x in hover_fields},
        colors=colors,
        width=width,
        height=height,
        n_divisions=_downsampling_factor('qq', n_divisions, collect_all),
        missing_label=missing_label,
    )
    from hail.methods.statgen import _lambda_gc_agg

    lambda_gc, max_p = ht.aggregate((
        _lambda_gc_agg(ht['p_value']),
        hail.agg.max(hail.max(ht.observed_p, ht.expected_p)),
    ))
    if isinstance(p, Column):
        qq = p.children[1]
    else:
        qq = p
    qq.x_range = DataRange1d(start=0, end=max_p + 1)
    qq.y_range = DataRange1d(start=0, end=max_p + 1)
    qq.add_layout(Slope(gradient=1, y_intercept=0, line_color='red'))

    label_color = 'red' if lambda_gc > 1.25 else 'orange' if lambda_gc > 1.1 else 'black'
    lgc_label = Label(
        x=max_p * 0.85,
        y=1,
        text=f'λ GC: {lambda_gc:.2f}',
        text_font_style='bold',
        text_color=label_color,
        text_font_size='14pt',
    )
    p.add_layout(lgc_label)

    return p


@typecheck(
    pvals=expr_float64,
    locus=nullable(expr_locus()),
    title=nullable(str),
    size=int,
    hover_fields=nullable(dictof(str, expr_any)),
    collect_all=nullable(bool),
    n_divisions=nullable(int),
    significance_line=nullable(numeric),
)
def manhattan(
    pvals: 'Float64Expression',
    locus: 'Optional[LocusExpression]' = None,
    title: 'Optional[str]' = None,
    size: int = 4,
    hover_fields: 'Optional[Dict[str, Expression]]' = None,
    collect_all: 'Optional[bool]' = None,
    n_divisions: 'Optional[int]' = 500,
    significance_line: 'Optional[Union[int, float]]' = 5e-8,
) -> Plot:
    """Create a Manhattan plot. (https://en.wikipedia.org/wiki/Manhattan_plot)

    Parameters
    ----------
    pvals : :class:`.Float64Expression`
        P-values to be plotted.
    locus : :class:`.LocusExpression`, optional
        Locus values to be plotted.
    title : str, optional
        Title of the plot.
    size : int
        Size of markers in screen space units.
    hover_fields : Dict[str, :class:`.Expression`], optional
        Dictionary of field names and values to be shown in the HoverTool of the plot.
    collect_all : bool, optional
        Deprecated - use `n_divisions` instead.
    n_divisions : int, optional.
        Factor by which to downsample (default value = 500).
        A lower input results in fewer output datapoints.
        Use `None` to collect all points.
    significance_line : float, optional
        p-value at which to add a horizontal, dotted red line indicating
        genome-wide significance.  If ``None``, no line is added.

    Returns
    -------
    :class:`bokeh.models.Plot`
    """
    if locus is None:
        locus = pvals._indices.source.locus

    ref = locus.dtype.reference_genome

    if hover_fields is None:
        hover_fields = {}

    hover_fields['locus'] = hail.str(locus)

    pvals = -hail.log10(pvals)

    source_pd = _collect_scatter_plot_data(
        ('_global_locus', locus.global_position()),
        ('_pval', pvals),
        fields=hover_fields,
        n_divisions=_downsampling_factor('manhattan', n_divisions, collect_all),
    )
    source_pd['p_value'] = [10 ** (-p) for p in source_pd['_pval']]
    source_pd['_contig'] = [locus.split(":")[0] for locus in source_pd['locus']]

    observed_contigs = [contig for contig in ref.contigs.copy() if contig in set(source_pd['_contig'])]

    contig_ticks = [ref._contig_global_position(contig) + ref.contig_length(contig) // 2 for contig in observed_contigs]
    color_mapper = CategoricalColorMapper(factors=ref.contigs, palette=palette[:2] * int((len(ref.contigs) + 1) / 2))

    p = figure(title=title, x_axis_label='Chromosome', y_axis_label='P-value (-log10 scale)', width=1000)
    p, _, legend, _, _, _ = _get_scatter_plot_elements(
        p,
        source_pd,
        x_col='_global_locus',
        y_col='_pval',
        label_cols=['_contig'],
        colors={'_contig': color_mapper},
        size=size,
        hover_cols={'locus', 'p_value'} | set(hover_fields),
    )
    assert legend is not None
    legend.visible = False
    p.xaxis.ticker = contig_ticks
    p.xaxis.major_label_overrides = dict(zip(contig_ticks, [contig.replace("chr", "") for contig in observed_contigs]))

    if significance_line is not None:
        p.renderers.append(
            Span(
                location=-math.log10(significance_line),
                dimension='width',
                line_color='red',
                line_dash='dashed',
                line_width=1.5,
            )
        )

    return p


@typecheck(
    entry_field=expr_any,
    row_field=nullable(oneof(expr_numeric, expr_locus())),
    column_field=nullable(expr_str),
    window=nullable(int),
    plot_width=int,
    plot_height=int,
)
def visualize_missingness(
    entry_field, row_field=None, column_field=None, window=6000000, plot_width=1800, plot_height=900
) -> figure:
    """Visualize missingness in a MatrixTable.

    Inspired by `naniar <https://cran.r-project.org/web/packages/naniar/index.html>`__.

    Row field is windowed by default, and missingness is aggregated over this window to generate a proportion defined.
    This windowing is set to 6,000,000 by default, so that the human genome is divided into ~500 rows.
    With ~2,000 columns, this function returns a sensibly-sized plot with this windowing.

    Warning
    -------
    Generating a plot with more than ~1M points takes a long time for Bokeh to render. Consider windowing carefully.

    Parameters
    ----------
    entry_field : :class:`.Expression`
        Field for which to check missingness.
    row_field : :class:`.NumericExpression` or :class:`.LocusExpression`
        Row field to use for y-axis (can be windowed). If not provided, the row key will be used.
    column_field : :class:`.StringExpression`
        Column field to use for x-axis. If not provided, the column key will be used.
    window : int, optional
        Size of window to summarize by ``row_field``. If set to None, each field will be used individually.
    plot_width : int
        Plot width in px.
    plot_height : int
        Plot height in px.

    Returns
    -------
    :class:`bokeh.plotting.figure`
    """
    mt = entry_field._indices.source
    if row_field is None:
        if isinstance(mt.row_key.dtype, hail.tstruct) and len(mt.row_key) == 1:
            row_field = mt.row_key[0]
        else:
            row_field = mt.row_key
    if column_field is None:
        column_field = hail.str(mt.col_key)
    row_source = row_field._indices.source
    column_source = column_field._indices.source
    if mt is None or row_source is None or column_source is None:
        raise ValueError("visualize_missingness expects expressions of 'MatrixTable', found scalar expression")
    if isinstance(mt, hail.Table):
        raise ValueError("visualize_missingness requires source to be MatrixTable, not Table")
    columns = column_field.collect()
    if not (mt == row_source == column_source):
        raise ValueError(
            f"visualize_missingness expects expressions from the same 'MatrixTable', "
            f"found {mt} and {row_source} and {column_source}"
        )
    # raise_unless_row_indexed('visualize_missingness', row_source)
    if window:
        row_field_is_locus = isinstance(row_field.dtype, hail.tlocus)
        row_field_is_numeric = row_field.dtype in (hail.tint32, hail.tint64, hail.tfloat32, hail.tfloat64)
        if row_field_is_locus:
            grouping = hail.locus_from_global_position(
                hail.int64(window) * hail.int64(row_field.global_position() / window)
            )
        elif row_field_is_numeric:
            grouping = hail.int64(window) * hail.int64(row_field / window)
        else:
            raise ValueError(
                f'When window is not None and row key must be numeric, but row key type was {mt.row_key.dtype}.'
            )
        mt = (
            mt.group_rows_by(_new_row_key=grouping)
            .partition_hint(100)
            .aggregate(is_defined=hail.agg.fraction(hail.is_defined(entry_field)))
        )
    else:
        mt = mt._select_all(
            row_exprs={'_new_row_key': row_field}, entry_exprs={'is_defined': hail.is_defined(entry_field)}
        )
    ht = mt.localize_entries('entry_fields', 'phenos')
    ht = ht.select(entry_fields=ht.entry_fields.map(lambda entry: entry.is_defined))
    data = ht.entry_fields.collect()
    if len(data) > 200:
        warning(
            f'Missingness dataset has {len(data)} rows. '
            f'This may take {"a very long time" if len(data) > 1000 else "a few minutes"} to plot.'
        )
    rows = hail.str(ht._new_row_key).collect()

    df = pd.DataFrame(data)
    df = df.rename(columns=dict(enumerate(columns))).rename(index=dict(enumerate(rows)))
    df.index.name = 'row'
    df.columns.name = 'column'

    df = pd.DataFrame(df.stack(), columns=['defined']).reset_index()

    p = figure(
        x_range=columns,
        y_range=list(reversed(rows)),
        x_axis_location="above",
        width=plot_width,
        height=plot_height,
        toolbar_location='below',
        tooltips=[('defined', '@defined'), ('row', '@row'), ('column', '@column')],
    )

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter

    mapper = LinearColorMapper(palette=colors, low=df.defined.min(), high=df.defined.max())

    p.rect(
        x='column',
        y='row',
        width=1,
        height=1,
        source=df,
        fill_color={'field': 'defined', 'transform': mapper},
        line_color=None,
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        major_label_text_font_size="5pt",
        ticker=BasicTicker(desired_num_ticks=len(colors)),
        formatter=PrintfTickFormatter(format="%d"),
        label_standoff=6,
        border_line_color=None,
        location=(0, 0),
    )
    p.add_layout(color_bar, 'right')
    return p
