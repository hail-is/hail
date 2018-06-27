import numpy as np
from math import log, isnan
from hail.typecheck import *
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Category10


def histogram(data, legend=None, title=None):
    """Create a histogram.

    Parameters
    ----------
    data : :class:`.Struct`
        Sequence of data to plot.
    legend : str
        Label of data on the x-axis.
    title : str
        Title of the histogram.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
    """
    p = figure(title=title, x_axis_label=legend, y_axis_label='Frequency', background_fill_color='#EEEEEE')
    p.quad(
        bottom=0, top=data.bin_freq,
        left=data.bin_edges[:-1], right=data.bin_edges[1:],
        legend=legend, line_color='black')
    return p


def scatter(x, y, label=None, title=None, xlabel=None, ylabel=None, size=4):
    """Create a scatterplot.

    Parameters
    ----------
    x : List[float]
        List of x-values to be plotted.
    y : List[float]
        List of y-values to be plotted.
    label : List[str]
        List of labels for x and y values, used to assign each point a label (e.g. population)
    title : str
        Title of the scatterplot.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    size : int
        Size of markers in screen space units.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
    """
    p = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, background_fill_color='#EEEEEE')
    if label is not None:
        source = ColumnDataSource(dict(x=x, y=y, label=label))
        factors = list(set(label))
        color_mapper = CategoricalColorMapper(factors=factors, palette=Category10[len(factors)])
        p.circle('x', 'y', alpha=0.5, source=source, size=size,
                 color={'field': 'label', 'transform': color_mapper}, legend='label')
    else:
        p.circle(x, y, alpha=0.5, size=size)
    return p


def qq(pvals):
    """Create a Quantile-Quantile plot. (https://en.wikipedia.org/wiki/Q-Q_plot)

    Parameters
    ----------
    pvals : List[float]
        P-values to be plotted.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
    """
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
