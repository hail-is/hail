import numpy as np
from math import log, isnan
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Category10


def histogram(data, label=None, title=None, ylabel='Frequency'):
    p = figure(title=title, x_axis_label=label, y_axis_label=ylabel, background_fill_color='#EEEEEE')
    p.quad(
        bottom=0, top=data.bin_freq,
        left=data.bin_edges[:-1], right=data.bin_edges[1:],
        legend=label, line_color='black')
    return p


def scatter(x, y, label=None, title=None, xlabel=None, ylabel=None, size=8):
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


# plots is a 2D array of plots that the user organizes, e.g. [[p1, p2], [p3, p4]]
def grid(plots, total_width=400, total_height=400):
    g = gridplot(plots, plot_width=total_width, plot_height=total_height)
    return g


def qq(pvals):
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
