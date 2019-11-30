Plot
====

.. warning::
    Plotting functionality is in early stages and is experimental. Interfaces will change regularly.

Plotting in Hail is easy. Hail's plot functions utilize Bokeh plotting libraries to create attractive,
interactive figures. Plotting functions in this module return a Bokeh Figure, so you can call
a method to plot your data and then choose to extend the plot however you like by interacting
directly with Bokeh. See the GWAS tutorial for examples.

Plot functions in Hail accept data in the form of either Python objects or :class:`.Table` and :class:`.MatrixTable` fields.

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.plot

.. autosummary::
    :nosignatures:

    cdf
    pdf
    smoothed_pdf
    histogram
    cumulative_histogram
    histogram2d
    scatter
    qq
    manhattan
    output_notebook
    visualize_missingness

.. autofunction:: cdf
.. autofunction:: pdf
.. autofunction:: smoothed_pdf
.. autofunction:: histogram
.. autofunction:: cumulative_histogram
.. autofunction:: histogram2d
.. autofunction:: scatter
.. autofunction:: qq
.. autofunction:: manhattan
.. autofunction:: output_notebook
.. autofunction:: visualize_missingness