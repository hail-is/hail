from typeguard import install_import_hook

install_import_hook("hail.ggplot2")

# These imports need to be placed after the import hook in order for typechecking to work.
# https://typeguard.readthedocs.io/en/stable/userguide.html#using-the-import-hook
from .altair_wrapper import ChartWrapper  # noqa: E402
from .ggplot2 import (  # noqa: E402
    aes,
    extend,
    geom_histogram,
    geom_line,
    geom_point,
    ggplot,
    show,
    undo,
)

__all__ = [
    "ChartWrapper",
    "aes",
    "extend",
    "geom_point",
    "geom_line",
    "geom_histogram",
    "ggplot",
    "undo",
    "show",
]
