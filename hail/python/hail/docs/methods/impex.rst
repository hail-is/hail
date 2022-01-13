.. _methods_impex:

Import / Export
===============

.. _methods_impex_export:

.. currentmodule:: hail.methods

.. toctree::
    :maxdepth: 2

This page describes functionality for moving data in and out of Hail.

Hail has a suite of functionality for importing and exporting data to and from
general-purpose, genetics-specific, and high-performance native file formats.

Native file formats
-------------------

.. _methods_impex_read:

When saving data to disk with the intent to later use Hail, we highly recommend
that you use the native file formats to store :class:`.Table` and
:class:`.MatrixTable` objects. These binary formats not only smaller than other formats
(especially textual ones) in most cases, but also are significantly faster to
read into Hail later.

These files can be created with methods on the :class:`.Table` and
:class:`.MatrixTable` objects:

- :meth:`.Table.write`
- :meth:`.MatrixTable.write`

These files can be read into a Hail session later using the following methods:

.. autosummary::

    read_matrix_table
    read_table

Import
------

General purpose
~~~~~~~~~~~~~~~

The :func:`.import_table` function is widely-used to import textual data
into a Hail :class:`.Table`. :func:`.import_matrix_table` is used to import
two-dimensional matrix data in textual representations into a Hail
:class:`.MatrixTable`. Finally, it is possible to create a Hail Table
from a :mod:`pandas` DataFrame with :meth:`.Table.from_pandas`.

.. autosummary::

    import_table
    import_matrix_table
    import_lines

Genetics
~~~~~~~~

Hail has several functions to import genetics-specific file formats into Hail
:class:`.MatrixTable` or :class:`.Table` objects:

.. autosummary::

    import_vcf
    import_plink
    import_bed
    import_bgen
    index_bgen
    import_gen
    import_fam
    import_locus_intervals
    import_gvcfs

Export
------

General purpose
~~~~~~~~~~~~~~~

Some of the most widely-used export functionality is found as class methods
on the :class:`.Table` and :class:`.Expression` objects:

- :meth:`.Table.export`: Used to write a Table to a text table (TSV).
- :meth:`.Expression.export`: Used to write an expression to a text file. For
  one-dimensional expressions (table row fields, matrix table row or column fields),
  this is very similar to :meth:`.Table.export`. For two-dimensional expressions
  (entry expressions on matrix tables), a text matrix representation that can be
  imported with :func:`.import_matrix_table` will be produced.
- :meth:`.Table.to_pandas`: Used to convert a Hail table to a :mod:`pandas`
  DataFrame.

Genetics
~~~~~~~~

Hail can export to some of the genetics-specific file formats:

.. autosummary::

    export_vcf
    export_bgen
    export_plink
    export_gen
    export_elasticsearch
    get_vcf_metadata


Reference documentation
-----------------------

.. autofunction:: read_matrix_table
.. autofunction:: read_table
.. autofunction:: import_bed
.. autofunction:: import_bgen
.. autofunction:: index_bgen
.. autofunction:: import_fam
.. autofunction:: import_gen
.. autofunction:: import_locus_intervals
.. autofunction:: import_matrix_table
.. autofunction:: import_plink
.. autofunction:: import_table
.. autofunction:: import_lines
.. autofunction:: import_vcf
.. autofunction:: import_gvcfs
.. autofunction:: export_vcf
.. autofunction:: export_elasticsearch
.. autofunction:: export_bgen
.. autofunction:: export_gen
.. autofunction:: export_plink
.. autofunction:: get_vcf_metadata
