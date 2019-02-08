from math import log, isnan, log10

import numpy as np
import bokeh
import bokeh.io
from bokeh.models import *
from bokeh.plotting import figure
from itertools import cycle

from hail.expr import aggregators
from hail.expr.expressions import *
from hail.expr.expressions import Expression
from hail.typecheck import *
from hail import Table
import hail

palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def output_notebook():
    """Configure the Bokeh output state to generate output in notebook
    cells when :func:`show` is called.  Calls
    :func:`bokeh.io.output_notebook`.

    """
    bokeh.io.output_notebook()

def show(obj):
    """Immediately display a Bokeh object or application.  Calls
    :func:`bokeh.io.show`.

    Parameters
    ----------
    obj
        A Bokeh object to display.
    """
    bokeh.io.show(obj)

@typecheck(data=oneof(hail.utils.struct.Struct, expr_float64), range=nullable(sized_tupleof(numeric, numeric)),
           bins=int, legend=nullable(str), title=nullable(str))
def histogram(data, range=None, bins=50, legend=None, title=None):
    """Create a histogram.

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
                finite_data = hail.bind(lambda x: hail.case().when(hail.is_finite(x), x).or_missing(), data)
                start, end = agg_f((aggregators.min(finite_data),
                                    aggregators.max(finite_data)))
                if start is None and end is None:
                    raise ValueError(f"'data' contains no values that are defined and finite")
            data = agg_f(aggregators.hist(data, start, end, bins))
        else:
            return ValueError('Invalid input')

    p = figure(title=title, x_axis_label=legend, y_axis_label='Frequency', background_fill_color='#EEEEEE')
    p.quad(
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
    return p


@typecheck(data=oneof(hail.utils.struct.Struct, expr_float64), range=nullable(sized_tupleof(numeric, numeric)),
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


@typecheck(x=oneof(sequenceof(numeric), expr_float64), y=oneof(sequenceof(numeric), expr_float64),
           label=oneof(nullable(str), expr_str, sequenceof(str)), title=nullable(str),
           xlabel=nullable(str), ylabel=nullable(str), size=int, legend=bool,
           source_fields=nullable(dictof(str, sequenceof(anytype))), collect_all=nullable(bool), n_divisions=int)
def scatter(x, y, label=None, title=None, xlabel=None, ylabel=None, size=4, legend=True,
            collect_all=False, n_divisions=500, source_fields=None):
    """Create a scatterplot.

    Parameters
    ----------
    x : List[float] or :class:`.Float64Expression`
        List of x-values to be plotted.
    y : List[float] or :class:`.Float64Expression`
        List of y-values to be plotted.
    label : List[str] or :class:`.StringExpression`
        List of labels for x and y values, used to assign each point a label (e.g. population)
    title : str
        Title of the scatterplot.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    size : int
        Size of markers in screen space units.
    legend : bool
        Whether or not to show the legend in the resulting figure.
    collect_all : bool
        Whether to collect all values or downsample before plotting.
        This parameter will be ignored if x and y are Python objects.
    n_divisions : int
        Factor by which to downsample (default value = 500). A lower input results in fewer output datapoints.
    source_fields : Dict[str, List[Any]]
        Extra fields for the ColumnDataSource of the plot.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
    """
    if isinstance(x, Expression) and isinstance(y, Expression):
        agg_f = x._aggregation_method()
        if isinstance(label, Expression):
            if collect_all:
                res = hail.tuple([x, y, label]).collect()
                label = [point[2] for point in res]
            else:
                res = agg_f(aggregators.downsample(x, y, label=label, n_divisions=n_divisions))
                label = [point[2][0] for point in res]

            x = [point[0] for point in res]
            y = [point[1] for point in res]
        else:
            if collect_all:
                res = hail.tuple([x, y]).collect()
            else:
                res = agg_f(aggregators.downsample(x, y, n_divisions=n_divisions))

            x = [point[0] for point in res]
            y = [point[1] for point in res]
    elif isinstance(x, Expression) or isinstance(y, Expression):
        raise TypeError('Invalid input: x and y must both be either Expressions or Python Lists.')
    else:
        if isinstance(label, Expression):
            label = label.collect()

    p = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, background_fill_color='#EEEEEE')
    if label is not None:
        fields = dict(x=x, y=y, label=label)
        if source_fields is not None:
            for key, values in source_fields.items():
                fields[key] = values

        source = ColumnDataSource(fields)

        if legend:
            leg = 'label'
        else:
            leg = None

        factors = list(set(label))
        if len(factors) > len(palette):
            color_gen = cycle(palette)
            colors = []
            for i in range(0, len(factors)):
                colors.append(next(color_gen))
        else:
            colors = palette[0:len(factors)]

        color_mapper = CategoricalColorMapper(factors=factors, palette=colors)
        p.circle('x', 'y', alpha=0.5, source=source, size=size,
                 color={'field': 'label', 'transform': color_mapper}, legend=leg)
    else:
        p.circle(x, y, alpha=0.5, size=size)
    return p


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
    def get_contig_index(x, starts):
        left = 0
        right = len(starts) - 1
        while left <= right:
            mid = (left + right) // 2
            if x < starts[mid]:
                if x >= starts[mid - 1]:
                    return mid - 1
                right = mid
            elif x >= starts[mid+1]:
                left = mid + 1
            else:
                return mid

    if locus is None:
        locus = pvals._indices.source.locus

    if hover_fields is None:
        hover_fields = {}

    hover_fields['locus'] = hail.str(locus)

    pvals = -hail.log10(pvals)

    if collect_all:
        res = hail.tuple([locus.global_position(), pvals, hail.struct(**hover_fields)]).collect()
        hf_struct = [point[2] for point in res]
        for key in hover_fields:
            hover_fields[key] = [item[key] for item in hf_struct]
    else:
        agg_f = pvals._aggregation_method()
        res = agg_f(aggregators.downsample(locus.global_position(), pvals,
                                           label=hail.array([hail.str(x) for x in hover_fields.values()]),
                                           n_divisions=n_divisions))
        fields = [point[2] for point in res]
        for idx, key in enumerate(list(hover_fields.keys())):
            hover_fields[key] = [field[idx] for field in fields]

    x = [point[0] for point in res]
    y = [point[1] for point in res]
    y_linear = [10 ** (-p) for p in y]
    hover_fields['p_value'] = y_linear

    ref = locus.dtype.reference_genome

    total_pos = 0
    start_points = []
    for i in range(0, len(ref.contigs)):
        start_points.append(total_pos)
        total_pos += ref.lengths.get(ref.contigs[i])
    start_points.append(total_pos)  # end point of all contigs

    observed_contigs = set()
    label = []
    for element in x:
        contig_index = get_contig_index(element, start_points)
        label.append(str(contig_index % 2))
        observed_contigs.add(ref.contigs[contig_index])

    labels = ref.contigs.copy()
    num_deleted = 0
    mid_points = []
    for i in range(0, len(ref.contigs)):
        if ref.contigs[i] in observed_contigs:
            length = ref.lengths.get(ref.contigs[i])
            mid = start_points[i] + length / 2
            if mid % 1 == 0:
                mid += 0.5
            mid_points.append(mid)
        else:
            del labels[i - num_deleted]
            num_deleted += 1

    p = scatter(x, y, label=label, title=title, xlabel='Chromosome', ylabel='P-value (-log10 scale)',
                size=size, legend=False, source_fields=hover_fields)

    p.xaxis.ticker = mid_points
    p.xaxis.major_label_overrides = dict(zip(mid_points, labels))
    p.width = 1000

    tooltips = [(key, "@{}".format(key)) for key in hover_fields]
    p.add_tools(HoverTool(
        tooltips=tooltips
    ))

    if significance_line is not None:
        p.renderers.append(Span(location=-log10(significance_line),
                                dimension='width',
                                line_color='red',
                                line_dash='dashed',
                                line_width=1.5))

    return p
