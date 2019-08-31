.. _methods_impex:

Export / Import
---------------

.. _methods_impex_export:

Export
~~~~~~

Export data from a Hail format into a non-Hail format. There is also
a :meth:`.Table.export` method for exporting a table to a .tsv file.

.. currentmodule:: hail.methods

.. toctree::
    :maxdepth: 2

.. autosummary::

    export_elasticsearch
    export_gen
    export_plink
    export_vcf
    get_vcf_metadata


.. _methods_impex_import:

Import
~~~~~~

Import data from a non-Hail format into a Hail format.

.. autosummary::

    import_bed
    import_bgen
    index_bgen
    import_fam
    import_gen
    import_locus_intervals
    import_matrix_table
    import_plink
    import_table
    import_vcf

.. _methods_impex_read:

Read
~~~~

Read data from a Hail format.

.. autosummary::

    read_matrix_table
    read_table

.. autofunction:: export_elasticsearch
.. autofunction:: export_gen
.. autofunction:: export_bgen
.. autofunction:: export_plink
.. autofunction:: export_vcf
.. autofunction:: get_vcf_metadata
.. autofunction:: import_bed
.. autofunction:: import_bgen
.. autofunction:: index_bgen
.. autofunction:: import_fam
.. autofunction:: import_gen
.. autofunction:: import_locus_intervals
.. autofunction:: import_matrix_table
.. autofunction:: import_plink
.. autofunction:: import_table
.. autofunction:: import_vcf
.. autofunction:: read_matrix_table
.. autofunction:: read_table
