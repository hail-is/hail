-----------------
Plotting Overview
-----------------

.. warning::
    Plotting functionality is in early stages and is experimental.

The ``hl.gg`` module is designed based on R's ``ggplot2`` library. This module provides a subset of ``ggplot2``'s
functionality to allow users to generate plots in much the same way they would in ``ggplot2``.

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
.. autofunction:: coord_cartesian

.. rubric:: Geoms

.. autosummary::
    :nosignatures:

    geom_point
    geom_line
    geom_text
    geom_bar
    geom_histogram
    geom_hline
    geom_vline

.. autofunction:: geom_point
.. autofunction:: geom_line
.. autofunction:: geom_text
.. autofunction:: geom_bar
.. autofunction:: geom_histogram
.. autofunction:: geom_hline
.. autofunction:: geom_vline


.. rubric:: Scales

.. autosummary::
    :nosignatures

    scale_x_continuous
    scale_x_discrete
    scale_x_genomic
    scale_x_log10
    scale_x_reverse
    scale_y_continuous
    scale_y_discrete
    scale_y_log10
    scale_y_reverse
    scale_color_continuous
    scale_color_discrete
    scale_color_identity
    scale_fill_continuous
    scale_fill_discrete
    scale_fill_identity

.. autofunction:: scale_x_continuous
.. autofunction:: scale_x_discrete
.. autofunction:: scale_x_genomic
.. autofunction:: scale_x_log10
.. autofunction:: scale_x_reverse
.. autofunction:: scale_y_continuous
.. autofunction:: scale_y_discrete
.. autofunction:: scale_y_log10
.. autofunction:: scale_y_reverse
.. autofunction:: scale_color_continuous
.. autofunction:: scale_color_discrete
.. autofunction:: scale_color_identity
.. autofunction:: scale_fill_continuous
.. autofunction:: scale_fill_discrete
.. autofunction:: scale_fill_identity

.. rubric:: Labels

.. autosummary::
    :nosignatures

    xlab
    ylab
    ggtitle

.. autofunction:: xlab
.. autofunction:: ylab
.. autofunction:: ggtitle

.. rubric:: Classes

.. autoclass:: GGPlot
.. autoclass:: Aesthetic
.. autoclass:: Geom
.. autoclass:: Labels
.. autoclass:: Scale
.. autoclass:: CoordCartesian