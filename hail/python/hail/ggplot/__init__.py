from hailtop import is_notebook

if is_notebook():
    from plotly.io import renderers

    renderers.default = 'iframe'

from .coord_cartesian import coord_cartesian
from .ggplot import ggplot, GGPlot  # noqa F401
from .aes import aes, Aesthetic  # noqa F401
from .geoms import (
    FigureAttribute,
    geom_line,
    geom_point,
    geom_text,
    geom_bar,
    geom_histogram,
    geom_density,
    geom_func,
    geom_hline,
    geom_vline,
    geom_tile,
    geom_col,
    geom_area,
    geom_ribbon,
)  # noqa F401
from .labels import ggtitle, xlab, ylab, labs
from .scale import (
    scale_x_continuous,
    scale_y_continuous,
    scale_x_discrete,
    scale_y_discrete,
    scale_x_genomic,
    scale_x_log10,
    scale_y_log10,
    scale_x_reverse,
    scale_y_reverse,
    scale_color_discrete,
    scale_color_hue,
    scale_color_identity,
    scale_color_manual,
    scale_color_continuous,
    scale_fill_discrete,
    scale_fill_hue,
    scale_fill_identity,
    scale_fill_continuous,
    scale_fill_manual,
    scale_shape_manual,
    scale_shape_auto,
)
from .facets import vars, facet_wrap

__all__ = [
    "aes",
    "ggplot",
    "geom_point",
    "geom_line",
    "geom_text",
    "geom_bar",
    "geom_col",
    "geom_histogram",
    "geom_density",
    "geom_hline",
    "geom_func",
    "geom_vline",
    "geom_tile",
    "geom_area",
    "geom_ribbon",
    "ggtitle",
    "xlab",
    "ylab",
    "labs",
    "coord_cartesian",
    "scale_x_continuous",
    "scale_y_continuous",
    "scale_x_discrete",
    "scale_y_discrete",
    "scale_x_genomic",
    "scale_x_log10",
    "scale_y_log10",
    "scale_x_reverse",
    "scale_y_reverse",
    "scale_color_continuous",
    "scale_color_identity",
    "scale_color_discrete",
    "scale_color_hue",
    "scale_color_manual",
    "scale_fill_continuous",
    "scale_fill_identity",
    "scale_fill_discrete",
    "scale_fill_hue",
    "scale_fill_manual",
    "scale_shape_manual",
    "scale_shape_auto",
    "facet_wrap",
    "vars",
]
