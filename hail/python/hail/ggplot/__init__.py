from hailtop import is_notebook

if is_notebook():
    from plotly.io import renderers

    renderers.default = 'iframe'

from .aes import Aesthetic, aes  # noqa F401
from .coord_cartesian import coord_cartesian
from .facets import facet_wrap, vars
from .geoms import (
    FigureAttribute,  # noqa F401
    geom_area,
    geom_bar,
    geom_col,
    geom_density,
    geom_func,
    geom_histogram,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_text,
    geom_tile,
    geom_vline,
)
from .ggplot import GGPlot, ggplot  # noqa F401
from .labels import ggtitle, labs, xlab, ylab
from .scale import (
    scale_color_continuous,
    scale_color_discrete,
    scale_color_hue,
    scale_color_identity,
    scale_color_manual,
    scale_fill_continuous,
    scale_fill_discrete,
    scale_fill_hue,
    scale_fill_identity,
    scale_fill_manual,
    scale_shape_auto,
    scale_shape_manual,
    scale_x_continuous,
    scale_x_discrete,
    scale_x_genomic,
    scale_x_log10,
    scale_x_reverse,
    scale_y_continuous,
    scale_y_discrete,
    scale_y_log10,
    scale_y_reverse,
)

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
