Plot
====

.. warning::
    Plotting functionality is in early stages and is experimental. Interfaces will change regularly.

Plotting in Hail is easy. Hail's plot functions utilize Bokeh plotting libraries to create attractive,
interactive figures. Plotting functions in this module return a Bokeh Figure, so you can call
a method to plot your data and then choose to extend the plot however you like by interacting
directly with Bokeh.

Plot functions in Hail accept data in the form of either Python objects or :class:`.Table` and :class:`.MatrixTable` fields.

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.plot

.. autosummary::
    :nosignatures:

    histogram
    scatter
    qq

.. autofunction:: histogram
.. autofunction:: scatter
.. autofunction:: qq