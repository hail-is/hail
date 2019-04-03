import warnings
from math import log, isnan, log10

import numpy as np
import pandas as pd
import bokeh
import bokeh.io
from bokeh.models import *
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.layouts import row, column, gridplot
from itertools import cycle

from hail.expr import aggregators
from hail.expr.expressions import *
from hail.expr.expressions import Expression
from hail.typecheck import *
from hail import Table
from hail.utils.struct import Struct
from typing import *
import hail

palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def output_notebook():
    """Configure the Bokeh output state to generate output in notebook
    cells when :func:`show` is called.  Calls
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

def cdf(data, k=350, legend=None, title=None, normalize=True, log=False):
    """Create a cumulative density plot.

    Parameters
    ----------
    data : :class:`.Struct` or :class:`.Float64Expression`
        Sequence of data to plot.
    k : int
        Accuracy parameter.
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
    :class:`bokeh.plotting.figure.Figure`
    """
    if isinstance(data, Expression):
        if data._indices is None:
            return ValueError('Invalid input')
        agg_f = data._aggregation_method()
        data = agg_f(aggregators.approx_cdf(data, k))

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
        plot_width=600,
        plot_height=400,
        background_fill_color='#EEEEEE',
        tools='xpan,xwheel_zoom,reset,save',
        active_scroll='xwheel_zoom')
    p.add_tools(HoverTool(tooltips=[("value", "$x"), ("rank", "@top")], mode='vline'))

    ranks = np.array(data.ranks)
    values = np.array(data.values)
    if normalize:
        ranks = ranks / ranks[-1]

    # invisible, there to support tooltips
    p.quad(top=ranks[1:-1],
           bottom=ranks[1:-1],
           left=values[:-1],
           right=values[1:],
           fill_alpha=0,
           line_alpha=0)
    p.step(x=[*values, values[-1]],
           y=ranks,
           line_width=2,
           line_color='black',
           legend=legend)
    return p


def pdf(data, k=350, smoothing=.5, legend=None, title=None, log=False, interactive=False):
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
        If `True`, return a handle to pass to :func:`.show`.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
    """
    if isinstance(data, Expression):
        if data._indices is None:
            return ValueError('Invalid input')
        agg_f = data._aggregation_method()
        data = agg_f(aggregators.approx_cdf(data, k))

    y_axis_label = 'Frequency'
    if log:
        y_axis_type = 'log'
    else:
        y_axis_type = 'linear'
    p = figure(
        title=title,
        x_axis_label=legend,
        y_axis_label=y_axis_label,
        y_axis_type=y_axis_type,
        plot_width=600,
        plot_height=400,
        tools='xpan,xwheel_zoom,reset,save',
        active_scroll='xwheel_zoom',
        background_fill_color='#EEEEEE')

    n = data.ranks[-1]
    weights = np.diff(data.ranks[1:-1])
    min = data.values[0]
    max = data.values[-1]
    values = np.array(data.values[1:-1])
    slope = 1 / (max - min)

    def f(x, prev, smoothing=smoothing):
        inv_scale = (np.sqrt(n * slope) / smoothing) * np.sqrt(prev / weights)
        diff = x[:, np.newaxis] - values
        grid = (3 / (4 * n)) * weights * np.maximum(0, inv_scale - np.power(diff, 2) * np.power(inv_scale, 3))
        return np.sum(grid, axis=1)

    round1 = f(values, np.full(len(values), slope))
    x_d = np.linspace(min, max, 1000)
    final = f(x_d, round1)

    l = p.line(x_d, final, line_width=2, line_color='black', legend=legend)

    if interactive:
        def mk_interact(handle):
            def update(smoothing=smoothing):
                final = f(x_d, round1, smoothing)
                l.data_source.data = {'x': x_d, 'y': final}
                bokeh.io.push_notebook(handle)

            from ipywidgets import interact
            interact(update, smoothing=(.02, .8, .005))

        return p, mk_interact
    else:
        return p


@typecheck(data=oneof(Struct, expr_float64), range=nullable(sized_tupleof(numeric, numeric)),
           bins=int, legend=nullable(str), title=nullable(str), log=bool, interactive=bool)
def histogram(data, range=None, bins=50, legend=None, title=None, log=False, interactive=False):
    """Create a histogram.

    Notes
    -----
    `data` can be a :class:`.Float64Expression`, or the result of the :func:`.agg.hist`
    or :func:`.agg.approx_cdf` aggregators.

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
    :class:`bokeh.plotting.figure.Figure`
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
                start, end = agg_f((aggregators.min(finite_data),
                                    aggregators.max(finite_data)))
                if start is None and end is None:
                    raise ValueError(f"'data' contains no values that are defined and finite")
            data = agg_f(aggregators.hist(data, start, end, bins))
        else:
            return ValueError('Invalid input')
    elif 'values' in data:
        cdf = data
        hist, edges = np.histogram(cdf.values, bins=bins, weights=np.diff(cdf.ranks), density=True)
        data = Struct(bin_freq=hist, bin_edges=edges, n_larger=0, n_smaller=0)


    if log:
        data.bin_freq = [log10(x) for x in data.bin_freq]
        data.n_larger = log10(data.n_larger)
        data.n_smaller = log10(data.n_smaller)
        y_axis_label = 'log10 Frequency'
    else:
        y_axis_label = 'Frequency'

    x_span = data.bin_edges[-1] - data.bin_edges[0]
    x_start = data.bin_edges[0] - .05 * x_span
    x_end = data.bin_edges[-1] + .05 * x_span
    p = figure(
        title=title,
        x_axis_label=legend,
        y_axis_label=y_axis_label,
        background_fill_color='#EEEEEE',
        x_range=(x_start, x_end))
    q = p.quad(
        bottom=0, top=data.bin_freq,
        left=data.bin_edges[:-1], right=data.bin_edges[1:],
        legend=legend, line_color='black')
    if data.n_larger > 0:
        p.quad(
            bottom=0, top=data.n_larger,
            left=data.bin_edges[-1], right=(data.bin_edges[-1] + (data.bin_edges[1] - data.bin_edges[0])),
            line_color='black', fill_color='green', legend='Outliers Above')
    if data.n_smaller > 0:
        p.quad(
            bottom=0, top=data.n_smaller,
            left=data.bin_edges[0] - (data.bin_edges[1] - data.bin_edges[0]), right=data.bin_edges[0],
            line_color='black', fill_color='red', legend='Outliers Below')
    if interactive:
        def mk_interact(handle):
            def update(bins=bins, phase=0):
                if phase > 0 and phase < 1:
                    bins = bins + 1
                    delta = (cdf.values[-1] - cdf.values[0]) / bins
                    edges = np.linspace(cdf.values[0] - (1 - phase) * delta, cdf.values[-1] + phase * delta, bins)
                else:
                    edges = np.linspace(cdf.values[0], cdf.values[-1], bins)
                hist, edges = np.histogram(cdf.values, bins=edges, weights=np.diff(cdf.ranks), density=True)
                new_data = {'top': hist, 'left': edges[:-1], 'right': edges[1:], 'bottom': np.full(len(hist), 0)}
                q.data_source.data = new_data
                bokeh.io.push_notebook(handle)

            from ipywidgets import interact
            interact(update, bins=(0, 5*bins), phase=(0, 1, .01))

        return p, mk_interact
    else:
        return p


@typecheck(data=oneof(Struct, expr_float64), range=nullable(sized_tupleof(numeric, numeric)),
           bins=int, legend=nullable(str), title=nullable(str), normalize=bool, log=bool)
def cumulative_histogram(data, range=None, bins=50, legend=None, title=None, normalize=True, log=False):
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
    :class:`bokeh.plotting.figure.Figure`
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
            return ValueError('Invalid input')

    cumulative_data = np.cumsum(data.bin_freq) + data.n_smaller
    np.append(cumulative_data, [cumulative_data[-1] + data.n_larger])
    num_data_points = max(cumulative_data)

    if normalize:
        cumulative_data = cumulative_data / num_data_points
    if title is not None:
        title = f'{title} ({num_data_points:,} data points)'
    if log:
        p = figure(title=title, x_axis_label=legend, y_axis_label='Frequency',
                   background_fill_color='#EEEEEE', y_axis_type='log')
    else:
        p = figure(title=title, x_axis_label=legend, y_axis_label='Frequency', background_fill_color='#EEEEEE')
    p.line(data.bin_edges[:-1], cumulative_data, line_color='#036564', line_width=3)
    return p


@typecheck(x=expr_numeric, y=expr_numeric, bins=oneof(int, sequenceof(int)),
           range=nullable(sized_tupleof(nullable(sized_tupleof(numeric, numeric)),
                                        nullable(sized_tupleof(numeric, numeric)))),
           title=nullable(str), width=int, height=int,
           font_size=str, colors=sequenceof(str))
def histogram2d(x, y, bins=40, range=None,
                 title=None, width=600, height=600, font_size='7pt',
                 colors=bokeh.palettes.all_palettes['Blues'][7][::-1]):
    """Plot a two-dimensional histogram.

    ``x`` and ``y`` must both be a :class:`NumericExpression` from the same :class:`Table`.

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
    font_size : str
        String of font size in points (default '7pt').
    colors : List[str]
        List of colors (hex codes, or strings as described
        `here <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__). Compatible with one of the many
        built-in palettes available `here <https://bokeh.pydata.org/en/latest/docs/reference/palettes.html>`__.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
    """
    source = x._indices.source
    y_source = y._indices.source

    if source is None or y_source is None:
        raise ValueError("histogram_2d expects two expressions of 'Table', found scalar expression")
    if isinstance(source, hail.MatrixTable):
        raise ValueError("histogram_2d requires source to be Table, not MatrixTable")
    if source != y_source:
        raise ValueError(f"histogram_2d expects two expressions from the same 'Table', found {source} and {y_source}")
    check_row_indexed('histogram_2d', x)
    check_row_indexed('histogram_2d', y)
    if isinstance(bins, int):
        x_bins = y_bins = bins
    else:
        x_bins, y_bins = bins
    if range is None:
        x_range = y_range = None
    else:
        x_range, y_range = range
    if x_range is None or y_range is None:
        warnings.warn('At least one range was not defined in histogram_2d. Doing two passes...')
        ranges = source.aggregate(hail.struct(x_stats=hail.agg.stats(x),
                                              y_stats=hail.agg.stats(y)))
        if x_range is None:
            x_range = (ranges.x_stats.min, ranges.x_stats.max)
        if y_range is None:
            y_range = (ranges.y_stats.min, ranges.y_stats.max)
    else:
        warnings.warn('If x_range or y_range are specified in histogram_2d, and there are points '
                      'outside of these ranges, they will not be plotted')
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
        x=hail.str(x_levels.find(lambda w: x >= w)),
        y=hail.str(y_levels.find(lambda w: y >= w))
    ).aggregate(c=hail.agg.count())
    data = grouped_ht.filter(hail.is_defined(grouped_ht.x) & (grouped_ht.x != str(x_range[1])) &
                             hail.is_defined(grouped_ht.y) & (grouped_ht.y != str(y_range[1]))).to_pandas()

    mapper = LinearColorMapper(palette=colors, low=data.c.min(), high=data.c.max())

    x_axis = sorted(set(data.x), key=lambda z: float(z))
    y_axis = sorted(set(data.y), key=lambda z: float(z))
    p = figure(title=title,
               x_range=x_axis, y_range=y_axis,
               x_axis_location="above", plot_width=width, plot_height=height,
               tools="hover,save,pan,box_zoom,reset,wheel_zoom", toolbar_location='below')

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.axis.major_label_text_font_size = font_size
    import math
    p.xaxis.major_label_orientation = math.pi / 3

    p.rect(x='x', y='y', width=1, height=1,
           source=data,
           fill_color={'field': 'c', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size=font_size,
                         ticker=BasicTicker(desired_num_ticks=6),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    def set_font_size(p, font_size: str = '12pt'):
        """Set most of the font sizes in a bokeh figure

        Parameters
        ----------
        p : :class:`bokeh.plotting.figure.Figure`
            Input figure.
        font_size : str
            String of font size in points (e.g. '12pt').

        Returns
        -------
        :class:`bokeh.plotting.figure.Figure`
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

    p.select_one(HoverTool).tooltips = [('x', '@x'), ('y', '@y',), ('count', '@c')]
    p = set_font_size(p, font_size)
    return p


def _collect_scatter_plot_data(
        x: Tuple[str, NumericExpression],
        y: Tuple[str, NumericExpression],
        fields: Dict[str, Expression] = None,
        n_divisions: int = None,
        missing_label: str =  'NA'
) -> pd.DataFrame:

    expressions = dict()
    if fields is not None:
        expressions.update({k: hail.or_else(v, missing_label) if isinstance(v, StringExpression) else v for k, v in fields.items()})

    if n_divisions is None:
        collect_expr = hail.struct(**dict((k,v) for k,v in (x,y)), **expressions)
        plot_data = [point for point in collect_expr.collect() if point[x[0]] is not None and point[y[0]] is not None]
        source_pd = pd.DataFrame(plot_data)
    else:
        # FIXME: remove the type conversion logic if/when downsample supports continuous values for labels
        # Save all numeric types to cast in DataFrame
        numeric_expr = {k: 'int32' for k,v in expressions.items() if isinstance(v, Int32Expression)}
        numeric_expr.update({k: 'int64' for k,v in expressions.items() if isinstance(v, Int64Expression)})
        numeric_expr.update({k: 'float32' for k, v in expressions.items() if isinstance(v, Float32Expression)})
        numeric_expr.update({k: 'float64' for k, v in expressions.items() if isinstance(v, Float64Expression)})

        # Cast non-string types to string
        expressions = {k: hail.str(v) if not isinstance(v, StringExpression) else v for k,v in expressions.items()}

        agg_f = x[1]._aggregation_method()
        res = agg_f(hail.agg.downsample(x[1], y[1], label=list(expressions.values()) if expressions else None, n_divisions=n_divisions))
        source_pd = pd.DataFrame([
            dict(
                **{x[0]: point[0], y[0]: point[1]},
                **(dict(zip(expressions, point[2])) if point[2] is not None else {})
            ) for point in res
        ])
        source_pd = source_pd.astype(numeric_expr, copy=False)

    return source_pd


def _get_categorical_palette(factors: List[str]) -> Dict[str, str]:
    n = max(3, len(factors))
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
        sp: Plot, source_pd: pd.DataFrame, x_col: str, y_col: str, label_cols: List[str],
        colors: Dict[str, ColorMapper] = None, size: int = 4,
) -> Tuple[bokeh.plotting.Figure, Dict[str, List[LegendItem]], Legend, ColorBar, Dict[str, ColorMapper], List[Renderer]] :

    if not source_pd.shape[0]:
        print("WARN: No data to plot.")
        return sp, None, None, None, None, None

    sp.tools.append(HoverTool(tooltips=[(x_col, f'@{x_col}'), (y_col, f'@{y_col}')] +
                                       [(c, f'@{c}') for c in source_pd.columns if c not in [x_col, y_col]]
                              ))

    cds = ColumnDataSource(source_pd)

    if not label_cols:
        sp.circle(x_col, y_col, source=cds, size=size)
        return sp, None, None, None, None, None

    continuous_cols = [col for col in label_cols if
                       (str(source_pd.dtypes[col]).startswith('float') or
                        str(source_pd.dtypes[col]).startswith('int'))]
    factor_cols = [col for col in label_cols if col not in continuous_cols]

    #  Assign color mappers to columns
    if colors is None:
        colors = {}
    color_mappers = {}

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
    legend_items = {}

    if not factor_cols:
        all_renderers = [
            sp.circle(x_col, y_col, color=transform(initial_col, initial_mapper), source=cds, size=size)
        ]

    else:
        all_renderers = []
        legend_items = {col: DefaultDict(list) for col in factor_cols}
        for key in source_pd.groupby(factor_cols).groups.keys():
            key = key if len(factor_cols) > 1 else [key]
            cds_view = CDSView(source=cds, filters=[GroupFilter(column_name=factor_cols[i], group=key[i]) for i in range(0, len(factor_cols))])
            renderer = sp.circle(x_col, y_col, color=transform(initial_col, initial_mapper), source=cds, view=cds_view, size=size)
            all_renderers.append(renderer)
            for i in range(0, len(factor_cols)):
                legend_items[factor_cols[i]][key[i]].append(renderer)

        legend_items = {factor: [LegendItem(label=key, renderers=renderers) for key, renderers in key_renderers.items()] for factor, key_renderers in legend_items.items()}

    # Add legend / color bar
    legend = Legend(visible=False, click_policy='hide', orientation='vertical') if initial_col not in factor_cols else Legend(items=legend_items[initial_col], click_policy='hide', orientation='vertical')
    color_bar = ColorBar(visible=False) if initial_col not in continuous_cols else ColorBar(color_mapper=color_mappers[initial_col])
    sp.add_layout(legend, 'left')
    sp.add_layout(color_bar, 'left')

    return sp, legend_items, legend, color_bar, color_mappers, all_renderers


@typecheck(x=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
           y=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
           label=nullable(oneof(dictof(str, expr_any), expr_any)), title=nullable(str),
           xlabel=nullable(str), ylabel=nullable(str), size=int, legend=bool,
           hover_fields=nullable(dictof(str, expr_any)),
           colors=nullable(oneof(bokeh.models.mappers.ColorMapper, dictof(str, bokeh.models.mappers.ColorMapper))),
           width=int, height=int, collect_all=bool, n_divisions=nullable(int), missing_label=str)
def scatter(
        x: Union[NumericExpression, Tuple[str, NumericExpression]],
        y: Union[NumericExpression, Tuple[str, NumericExpression]],
        label: Union[Expression, Dict[str, Expression]] = None,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        size: int =4,
        legend: bool = True,
        hover_fields: Dict[str, Expression] = None,
        colors: Union[ColorMapper, Dict[str, ColorMapper]] = None,
        width: int = 800,
        height: int = 800,
        collect_all: bool = False,
        n_divisions: int = 500,
        missing_label: str = 'NA'
) -> Union[bokeh.plotting.Figure, Column]:
    """Create an interactive scatter plot.

    ``x`` and ``y`` must both be either:
    - a :class:`NumericExpression` from the same :class:`Table`.
    - a tuple (str, :class:`NumericExpression`) from the same :class:`Table`. If passed as a tuple the first element is used as the hover label.

    If no label or a single label is provided, then returns :class:`bokeh.plotting.figure.Figure`
    Otherwise returns a :class:`bokeh.plotting.figure.Column` containing:
    - a :class:`bokeh.models.widgets.Select` dropdown selection widget for labels
    - a :class:`bokeh.plotting.figure.Figure` containing the interactive scatter plot

    Points will be colored by one of the labels defined in the ``label`` using the color scheme defined in
    the corresponding entry of ``colors`` if provided (otherwise a default scheme is used). To specify your color
    mapper, check `the bokeh documentation <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__
    for CategoricalMapper for categorical labels, and for LinearColorMapper and LogColorMapper
    for continuous labels.
    For categorical labels, clicking on one of the items in the legend will hide/show all points with the corresponding label.
    Note that using many different labelling schemes in the same plots, particularly if those labels contain many
    different classes could slow down the plot interactions.

    Hovering on points will display their coordinates, labels and any additional fields specified in ``source_fields``.

    Parameters
    ----------
    x : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
        List of x-values to be plotted.
    y : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
        List of y-values to be plotted.
    label : :class:`.Expression` or Dict[str, :class:`.Expression`]]
        Either a single expression (if a single label is desired), or a
        dictionary of label name -> label value for x and y values.
        Used to color each point w.r.t its label.
        When multiple labels are given, a dropdown will be displayed with the different options.
        Can be used with categorical or continuous expressions.
    title : str
        Title of the scatterplot.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    size : int
        Size of markers in screen space units.
    legend: bool
        Whether or not to show the legend in the resulting figure.
    hover_fields : Dict[str, :class:`.Expression`]
        Extra fields to be displayed when hovering over a point on the plot.
    colors : :class:`bokeh.models.mappers.ColorMapper` or Dict[str, :class:`bokeh.models.mappers.ColorMapper`]
        If a single label is used, then this can be a color mapper, if multiple labels are used, then this should
        be a Dict of label name -> color mapper.
        Used to set colors for the labels defined using ``label``.
        If not used at all, or label names not appearing in this dict will be colored using a default color scheme.
    width: int
        Plot width
    height: int
        Plot height
    collect_all : bool
        Whether to collect all values or downsample before plotting.
    n_divisions : int
        Factor by which to downsample (default value = 500). A lower input results in fewer output datapoints.
    missing_label: str
        Label to use when a point is missing data for a categorical label

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure` if no label or a single label was given, otherwise :class:`bokeh.plotting.figure.Column`
    """
    hover_fields = {} if hover_fields is None else hover_fields
    label = {} if label is None else {'label': label} if isinstance(label, Expression) else label
    colors = {'label': colors} if isinstance(colors, ColorMapper) else colors
    label_cols = list(label.keys())
    if isinstance(x, NumericExpression):
        x = ('x', x)

    if isinstance(y, NumericExpression):
        y = ('y', y)

    source_pd = _collect_scatter_plot_data(x, y, fields={**hover_fields, **label}, n_divisions=None if collect_all else n_divisions, missing_label=missing_label)
    sp = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, height=height, width=width)
    sp, sp_legend_items, sp_legend, sp_color_bar, sp_color_mappers, sp_scatter_renderers = _get_scatter_plot_elements(sp, source_pd, x[0], y[0], label_cols, colors, size)

    if not legend:
        sp_legend.visible = False
        sp_color_bar.visible = False

    # If multiple labels, create JS call back selector
    if len(label_cols) > 1:
        callback_args=dict(
            color_mappers=sp_color_mappers,
            scatter_renderers=sp_scatter_renderers
        )
        callback_code = """
        for (var i = 0; i < scatter_renderers.length; i++){
            scatter_renderers[i].glyph.fill_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
            scatter_renderers[i].glyph.line_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
            scatter_renderers[i].visible = true
        }
        
        """

        if legend:
            callback_args.update(dict(
                legend_items=sp_legend_items,
                legend=sp_legend,
                color_bar=sp_color_bar
            ))
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


@typecheck(x=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
           y=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
           label=nullable(oneof(dictof(str, expr_any), expr_any)), title=nullable(str),
           xlabel=nullable(str), ylabel=nullable(str), size=int, legend=bool,
           hover_fields=nullable(dictof(str, expr_any)),
           colors=nullable(oneof(bokeh.models.mappers.ColorMapper, dictof(str, bokeh.models.mappers.ColorMapper))),
           width=int, height=int, collect_all=bool, n_divisions=nullable(int), missing_label=str)
def joint_plot(
        x: Union[NumericExpression, Tuple[str, NumericExpression]],
        y: Union[NumericExpression, Tuple[str, NumericExpression]],
        label: Union[Expression, Dict[str, Expression]] = None,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        size: int =4,
        legend: bool = True,
        hover_fields: Dict[str, StringExpression] = None,
        colors: Union[ColorMapper, Dict[str, ColorMapper]] = None,
        width: int = 800,
        height: int = 800,
        collect_all: bool = False,
        n_divisions: int = 500,
        missing_label: str = 'NA'
) -> Column:
    """Create an interactive scatter plot with marginal densities on the side.

       ``x`` and ``y`` must both be either:
       - a :class:`NumericExpression` from the same :class:`Table`.
       - a tuple (str, :class:`NumericExpression`) from the same :class:`Table`. If passed as a tuple the first element is used as the hover label.

       This function returns a :class:`bokeh.plotting.figure.Column` containing two :class:`bokeh.plotting.figure.Row`:
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

       Hovering on points in the scatter plot displays their coordinates, labels and any additional fields specified in ``source_fields``.

        Parameters
        ----------
        ----------
        x : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
            List of x-values to be plotted.
        y : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
            List of y-values to be plotted.
        label : :class:`.Expression` or Dict[str, :class:`.Expression`]]
            Either a single expression (if a single label is desired), or a
            dictionary of label name -> label value for x and y values.
            Used to color each point w.r.t its label.
            When multiple labels are given, a dropdown will be displayed with the different options.
            Can be used with categorical or continuous expressions.
        title : str
            Title of the scatterplot.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        size : int
            Size of markers in screen space units.
        legend: bool
            Whether or not to show the legend in the resulting figure.
        hover_fields : Dict[str, :class:`.Expression`]
            Extra fields to be displayed when hovering over a point on the plot.
        colors : :class:`bokeh.models.mappers.ColorMapper` or Dict[str, :class:`bokeh.models.mappers.ColorMapper`]
            If a single label is used, then this can be a color mapper, if multiple labels are used, then this should
            be a Dict of label name -> color mapper.
            Used to set colors for the labels defined using ``label``.
            If not used at all, or label names not appearing in this dict will be colored using a default color scheme.
        width: int
            Plot width
        height: int
            Plot height
        collect_all : bool
            Whether to collect all values or downsample before plotting.
        n_divisions : int
            Factor by which to downsample (default value = 500). A lower input results in fewer output datapoints.
        missing_label: str
            Label to use when a point is missing data for a categorical label


        Returns
        -------
        :class:`bokeh.plotting.figure.Column`
        """
    # Collect data
    hover_fields = {} if hover_fields is None else hover_fields
    label = {} if label is None else {'label': label} if isinstance(label, Expression) else label
    colors = {'label': colors} if isinstance(colors, ColorMapper) else colors
    if isinstance(x, NumericExpression):
        x = ('x', x)

    if isinstance(y, NumericExpression):
        y = ('y', y)

    label_cols = list(label.keys())
    source_pd = _collect_scatter_plot_data(x, y, fields={**hover_fields, **label}, n_divisions=None if collect_all else None, missing_label=missing_label)
    sp = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, height=height, width=width)
    sp, sp_legend_items, sp_legend, sp_color_bar, sp_color_mappers, sp_scatter_renderers = _get_scatter_plot_elements(sp, source_pd, x[0], y[0], label_cols, colors, size)

    continuous_cols = [col for col in label_cols if
                       (str(source_pd.dtypes[col]).startswith('float') or
                        str(source_pd.dtypes[col]).startswith('int'))]
    factor_cols = [col for col in label_cols if col not in continuous_cols]

    # Density plots
    def get_density_plot_items(
            source_pd,
            data_col,
            p,
            x_axis,
            colors: Dict[str, ColorMapper],
            continuous_cols: List[str],
            factor_cols: List[str]
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
            factor_colors = colors.get(factor_col, _get_categorical_palette(list(set(source_pd[factor_col]))))
            factor_colors = dict(zip(factor_colors.factors, factor_colors.palette))
            density_data = source_pd[[factor_col, data_col]].groupby(factor_col).apply(lambda df: np.histogram(df['x' if x_axis else 'y'], density=True))
            for factor, (dens, edges) in density_data.iteritems():
                edges = edges[:-1]
                xy = (edges, dens) if x_axis else (dens, edges)
                cds = ColumnDataSource({'x': xy[0], 'y': xy[1]})
                density_renderers.append((factor_col, factor, p.line('x', 'y', color=factor_colors.get(factor, 'gray'), source=cds)))
                max_densities[factor_col] = np.max(list(dens) + [max_densities.get(factor_col, 0)])

        p.legend.visible = False
        p.grid.visible = False
        p.outline_line_color = None
        return p, density_renderers, max_densities

    xp = figure(title=title, height=int(height / 3), width=width, x_range=sp.x_range)
    xp, x_renderers, x_max_densities = get_density_plot_items(source_pd, x[0], xp, x_axis=True, colors=sp_color_mappers, continuous_cols=continuous_cols, factor_cols=factor_cols)
    xp.xaxis.visible = False
    yp = figure(height=height, width=int(width / 3), y_range=sp.y_range)
    yp, y_renderers, y_max_densities = get_density_plot_items(source_pd, y[0], yp, x_axis=False, colors=sp_color_mappers, continuous_cols=continuous_cols, factor_cols=factor_cols)
    yp.yaxis.visible = False
    density_renderers = x_renderers + y_renderers
    first_row = [xp]

    if not legend:
        sp_legend.visible = False
        sp_color_bar.visible = False

    # If multiple labels, create JS call back selector
    if len(label_cols) > 1:

        for factor_col, factor, renderer in density_renderers:
            renderer.visible = factor_col == label_cols[0]

        if label_cols[0] in factor_cols:
            xp.y_range.start = 0
            xp.y_range.end = x_max_densities[label_cols[0]]
            yp.x_range.start = 0
            yp.x_range.end = y_max_densities[label_cols[0]]

        callback_args = dict(
            scatter_renderers=sp_scatter_renderers,
            color_mappers=sp_color_mappers,
            density_renderers=x_renderers + y_renderers,
            x_range = xp.y_range,
            x_max_densities=x_max_densities,
            y_range=yp.x_range,
            y_max_densities=y_max_densities
        )

        callback_code="""
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
            callback_args.update(dict(
                legend_items=sp_legend_items,
                legend=sp_legend,
                color_bar=sp_color_bar
            ))
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

    return gridplot(first_row, [sp, yp])


@typecheck(pvals=oneof(sequenceof(numeric), expr_float64), collect_all=bool, n_divisions=int)
def qq(pvals, collect_all=False, n_divisions=500):
    """Create a Quantile-Quantile plot. (https://en.wikipedia.org/wiki/Q-Q_plot)

    Parameters
    ----------
    pvals : List[float] or :class:`.Float64Expression`
        P-values to be plotted.
    collect_all : bool
        Whether to collect all values or downsample before plotting.
        This parameter will be ignored if pvals is a Python object.
    n_divisions : int
        Factor by which to downsample (default value = 500). A lower input results in fewer output datapoints.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
    """
    if isinstance(pvals, Expression):
        source = pvals._indices.source
        if source is not None:
            if collect_all:
                pvals = pvals.collect()
                spvals = sorted(filter(lambda x: x and not(isnan(x)), pvals))
                exp = [-log(float(i) / len(spvals), 10) for i in np.arange(1, len(spvals) + 1, 1)]
                obs = [-log(p, 10) for p in spvals]
            else:
                if isinstance(source, Table):
                    ht = source.select(pval=pvals).key_by().persist().key_by('pval')
                else:
                    ht = source.select_rows(pval=pvals).rows().key_by().select('pval').persist().key_by('pval')
                n = ht.count()
                ht = ht.select(idx=hail.scan.count())
                ht = ht.annotate(expected_p=(ht.idx + 1) / n)
                pvals = ht.aggregate(
                    aggregators.downsample(-hail.log10(ht.expected_p), -hail.log10(ht.pval), n_divisions=n_divisions))
                exp = [point[0] for point in pvals if not isnan(point[1])]
                obs = [point[1] for point in pvals if not isnan(point[1])]
        else:
            return ValueError('Invalid input: expression has no source')
    else:
        spvals = sorted(filter(lambda x: x and not(isnan(x)), pvals))
        exp = [-log(float(i) / len(spvals), 10) for i in np.arange(1, len(spvals) + 1, 1)]
        obs = [-log(p, 10) for p in spvals]

    p = figure(
        title='Q-Q Plot',
        x_axis_label='Expected p-value (-log10 scale)',
        y_axis_label='Observed p-value (-log10 scale)')
    p.scatter(x=exp, y=obs, color='black')
    bound = max(max(exp), max(obs)) * 1.1
    p.line([0, bound], [0, bound], color='red')
    return p


@typecheck(pvals=expr_float64, locus=nullable(expr_locus()), title=nullable(str),
           size=int, hover_fields=nullable(dictof(str, expr_any)), collect_all=bool, n_divisions=int, significance_line=nullable(numeric))
def manhattan(pvals, locus=None, title=None, size=4, hover_fields=None, collect_all=False, n_divisions=500, significance_line=5e-8):
    """Create a Manhattan plot. (https://en.wikipedia.org/wiki/Manhattan_plot)

    Parameters
    ----------
    pvals : :class:`.Float64Expression`
        P-values to be plotted.
    locus : :class:`.LocusExpression`
        Locus values to be plotted.
    title : str
        Title of the plot.
    size : int
        Size of markers in screen space units.
    hover_fields : Dict[str, :class:`.Expression`]
        Dictionary of field names and values to be shown in the HoverTool of the plot.
    collect_all : bool
        Whether to collect all values or downsample before plotting.
    n_divisions : int
        Factor by which to downsample (default value = 500). A lower input results in fewer output datapoints.
    significance_line : float, optional
        p-value at which to add a horizontal, dotted red line indicating
        genome-wide significance.  If ``None``, no line is added.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
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
        n_divisions=None if collect_all else n_divisions
    )
    source_pd['p_value'] = [10 ** (-p) for p in source_pd['_pval']]
    source_pd['_contig'] = [locus.split(":")[0] for locus in source_pd['locus']]

    observed_contigs = set(source_pd['_contig'])
    observed_contigs = [contig for contig in ref.contigs.copy() if contig in observed_contigs]
    contig_ticks = hail.eval([hail.locus(contig, int(ref.lengths[contig]/2)).global_position() for contig in observed_contigs])
    color_mapper = CategoricalColorMapper(factors=ref.contigs, palette= palette[:2] * int((len(ref.contigs)+1)/2))

    p = figure(title=title, x_axis_label='Chromosome', y_axis_label='P-value (-log10 scale)', width=1000)
    p, _, legend, _, _, _ = _get_scatter_plot_elements(
        p, source_pd, x_col='_global_locus', y_col='_pval',
        label_cols=['_contig'], colors={'_contig': color_mapper},
        size=size
    )
    legend.visible = False
    p.xaxis.ticker = contig_ticks
    p.xaxis.major_label_overrides = dict(zip(contig_ticks, observed_contigs))
    p.select_one(HoverTool).tooltips = [t for t in p.select_one(HoverTool).tooltips if not t[0].startswith('_')]

    if significance_line is not None:
        p.renderers.append(Span(location=-log10(significance_line),
                                dimension='width',
                                line_color='red',
                                line_dash='dashed',
                                line_width=1.5))

    return p
