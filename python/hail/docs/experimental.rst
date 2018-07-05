Experimental
============

This module serves two functions: as a staging area for extensions of Hail
not ready for inclusion in the main package, and as a library of lightly reviewed
community submissions.

Contribution Guidelines
-----------------------
Submissions from the community are welcome! The criteria for inclusion in the
experimental module are loose and subject to change:

1. Function docstrings are required. Hail uses
   `NumPy style docstrings <http://www.sphinx-doc.org/en/stable/ext/example_numpy.html#example-numpy>`__.
2. Tests are not required, but are encouraged. If you do include tests, they must
   run in no more than a few seconds. Place tests as a class method on ``Tests`` in
   ``python/tests/experimental/test_experimental.py``
3. Code style is not strictly enforced, aside from egregious violations. We do
   recommend using `autopep8 <https://pypi.org/project/autopep8/>`__ though!

.. currentmodule:: hail.experimental

Genetics Methods
----------------

.. autosummary::

    ld_score

.. autofunction:: ld_score
