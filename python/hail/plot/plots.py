import numpy as np
from math import log, isnan
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Span


def histogram(data, legend=None, title=None, xlabel=None, ylabel='Frequency', background_fill_color='#EEEEEE'):
    p = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, background_fill_color=background_fill_color, )
    p.quad(
        bottom=0, top=data.bin_freq,
        left=data.bin_edges[:-1], right=data.bin_edges[1:],
        legend=legend, line_color='black')
    show(p)


def scatter(x, y, title=None, xlabel=None, ylabel=None,
            background_fill_color='#EEEEEE', alpha=0.5,
            legend=None, color='blue', size=8,
            indicate_outlier_cuts=False, threshold_x=None, threshold_y=None):
    p = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, background_fill_color=background_fill_color, )
    p.scatter(x, y, alpha=alpha, color=color, legend=legend, size=size)
    if indicate_outlier_cuts:
        p.renderers.extend(
            [Span(location=threshold_y, dimension='width', line_color='black', line_width=1),
             Span(location=threshold_x, dimension='height', line_color='black', line_width=1)])
    show(p)


# plots is a 2D array of plots that the user organizes, e.g. [[p1, p2], [p3, p4]]
def grid(plots, total_width=400, total_height=400):
    g = gridplot(plots, plot_width=total_width, plot_height=total_height)
    show(g)


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
    show(p)