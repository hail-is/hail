.. _sec-scan:

Scans
===========

The ``scan`` module is exposed as ``hl.scan``, e.g. ``hl.scan.sum``.

The functions in this module perform rolling aggregations along the desired axis;
For example, the ``count`` aggregator can be used as ``hl.scan.count`` to add an
index along the rows of a `Table` or the rows or columns of a `MatrixTable`.

See the ``aggregators``<aggregators> module for documentation on the behavior of
specific aggregators.

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.expr.scan