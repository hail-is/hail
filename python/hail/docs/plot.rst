Plot
====

.. warning::
    Plotting functionality is in early stages and is experimental. Interfaces will change regularly.

Plotting in Hail is easy. Hail's plot functions utilize Bokeh plotting libraries to create interactive,
pretty figures. Plotting functions in this module return a Bokeh Figure, so you can call
a method to plot your data and then choose to extend the plot however you like by interacting
directly with Bokeh.

Plot functions in Hail currently accept data in the form of Python objects, but they will soon be able
to accept :class:`.Table` and :class:`.MatrixTable` fields as well.

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