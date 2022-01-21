-----------------
Plotting Overview
-----------------

.. warning::
    Plotting functionality is in early stages and is experimental.

The `hl.gg` module is designed based on R's `ggplot2` library. This module provides a subset of `ggplot2`'s
functionality to allow users to generate plots in much the same way they would in `ggplot2`.

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.gg

.. rubric:: Core functions

.. autosummary::
    :nosignatures:

    ggplot
    aes
    coord_cartesian

.. autofunction:: ggplot
.. autofunction:: aes

.. rubric:: Geoms

.. autosummary::
    :nosignatures:

    geom_point
    geom_line
    geom_text
    geom_bar
    geom_histogram

.. rubric:: Scales

.. autosummary::
    :nosignatures

    scale_x_continuous
    scale_x_discrete
    scale_x_genomic
    scale_y_continuous
    scale_y_discrete
    scale_color_continuous
    scale_color_discrete
    scale_color_identity
    scale_fill_continuous
    scale_fill_discrete
    scale_fill_identity

.. rubric:: Labels

.. autosummary::
    :nosignatures

    xlab
    ylab
    ggtitle
