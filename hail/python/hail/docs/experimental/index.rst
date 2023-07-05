Experimental
============

This module serves two functions: as a staging area for extensions of Hail
not ready for inclusion in the main package, and as a library of lightly reviewed
community submissions.

At present, the experimental module is organized into a few freestanding
modules, linked immediately below, and many freestanding functions, documented
on this page.

.. toctree::
    :maxdepth: 1

    ldscsim

Contribution Guidelines
-----------------------
Submissions from the community are welcome! The criteria for inclusion in the
experimental module are loose and subject to change:

1. Function docstrings are required. Hail uses
   `NumPy style docstrings <https://www.sphinx-doc.org/en/stable/usage/extensions/example_numpy.html>`__.
2. Tests are not required, but are encouraged. If you do include tests, they must
   run in no more than a few seconds. Place tests as a class method on ``Tests`` in
   ``python/tests/experimental/test_experimental.py``
3. Code style is not strictly enforced, aside from egregious violations. We do
   recommend using `autopep8 <https://pypi.org/project/autopep8/>`__ though!

.. currentmodule:: hail.experimental

Annotation Database
-------------------

.. rubric:: Classes

.. autosummary::
    :nosignatures:
    :toctree: ./
    :template: class.rst

    hail.experimental.DB

Genetics Methods
----------------

.. autosummary::

    load_dataset
    ld_score
    ld_score_regression
    write_expression
    read_expression
    filtering_allele_frequency
    hail_metadata
    plot_roc_curve
    phase_by_transmission
    phase_trio_matrix_by_transmission
    explode_trio_matrix
    import_gtf
    get_gene_intervals
    export_entries_by_col
    pc_project

`dplyr`-inspired Methods
------------------------

.. autosummary::

    gather
    separate
    spread

Functions
---------

.. autofunction:: load_dataset
.. autofunction:: ld_score
.. autofunction:: ld_score_regression
.. autofunction:: write_expression
.. autofunction:: read_expression
.. autofunction:: hail_metadata
.. autofunction:: plot_roc_curve
.. autofunction:: filtering_allele_frequency
.. autofunction:: phase_by_transmission
.. autofunction:: phase_trio_matrix_by_transmission
.. autofunction:: explode_trio_matrix
.. autofunction:: import_gtf
.. autofunction:: get_gene_intervals
.. autofunction:: export_entries_by_col
.. autofunction:: gather
.. autofunction:: separate
.. autofunction:: spread
.. autofunction:: full_outer_join_mt
.. autofunction:: strftime
.. autofunction:: strptime
.. autofunction:: pc_project
.. autofunction:: loop
