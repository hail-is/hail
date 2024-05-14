from hailtop import is_notebook

if is_notebook():
    from bokeh.io import output_notebook

    output_notebook()

from .plots import (
    output_notebook,
    show,
    histogram,
    cumulative_histogram,
    histogram2d,
    scatter,
    joint_plot,
    qq,
    manhattan,
    smoothed_pdf,
    pdf,
    cdf,
    set_font_size,
    visualize_missingness,
)

__all__ = [
    'output_notebook',
    'show',
    'histogram',
    'cumulative_histogram',
    'scatter',
    'joint_plot',
    'histogram2d',
    'qq',
    'manhattan',
    'pdf',
    'smoothed_pdf',
    'cdf',
    'set_font_size',
    'visualize_missingness',
]
