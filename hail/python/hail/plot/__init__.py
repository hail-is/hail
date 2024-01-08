from hailtop import is_notebook

if is_notebook():
    from bokeh.io import output_notebook

    output_notebook()

from .plots import (
    cdf,
    cumulative_histogram,
    histogram,
    histogram2d,
    joint_plot,
    manhattan,
    output_notebook,
    pdf,
    qq,
    scatter,
    set_font_size,
    show,
    smoothed_pdf,
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
