.. _how_to_basics:

Basic Methods for Working with Hail Data
========================================

Get Data Into and Out of Hail
-----------------------------

Import
~~~~~~

Import data from a non-Hail format into a Hail format, using
one of the :ref:`import_* <methods_impex_import>` methods.

:**description**: Import a .tsv file as a table.

:**code**:

    .. code-block:: python

        >>> table = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')
        >>> table.show()
        +-------+-------+-----+-------+-------+-------+-------+-------+
        |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |
        +-------+-------+-----+-------+-------+-------+-------+-------+
        | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 |
        +-------+-------+-----+-------+-------+-------+-------+-------+
        |     1 |    65 | "M" |     5 |     4 |     2 |    50 |     5 |
        |     2 |    72 | "M" |     6 |     3 |     2 |    61 |     1 |
        |     3 |    70 | "F" |     7 |     3 |    10 |    81 |    -5 |
        |     4 |    60 | "F" |     8 |     2 |    11 |    90 |   -10 |
        +-------+-------+-----+-------+-------+-------+-------+-------+

:**dependencies**: :func:`.import_table`


Export
~~~~~~

Export Hail data to a non-Hail format, using one of the
:ref:`export_* <methods_impex_export>` methods.

:**description**: Export a matrix table as a VCF.

:**code**:

    >>> hl.export_vcf(mt, 'output/example.vcf.bgz') # doctest: +SKIP

:**dependencies**: :func:`.export_vcf`

Write
~~~~~

Write data in a Hail format to disk using one of
the write() methods, e.g. :meth:`.Table.write` or :meth:`.MatrixTable.write`.

:**description**: Write a matrix table to disk.

:**code**:

    >>> mt.write('output/example.mt') # doctest: +SKIP

:**dependencies**:  :meth:`.MatrixTable.write`

Read
~~~~

If you wrote a table or matrix table to disk using one of Hail's write()
methods, you can read it using one of the
:ref:`read <methods_impex_read>` methods.

:**description**: Read a table from disk.

:**code**:

    >>> ht = hl.read_table('data/example.ht') # doctest: +SKIP

:**dependencies**: :func:`.read_table`

Examine your data
-----------------

Explore the schema
~~~~~~~~~~~~~~~~~~

Matrix Table
............

:**description**: Get information about the fields and keys of a matrix table.

:**code**:

    .. code-block:: python

        >>> mt.describe()  # doctest: +SKIP_OUTPUT_CHECK
        ----------------------------------------
        Global fields:
            'populations': array<str>
        ----------------------------------------
        Column fields:
            's': str
            'is_case': bool
            'pheno': struct {
                is_case: bool,
                is_female: bool,
                age: float64,
                height: float64,
                blood_pressure: float64,
                cohort_name: str
            }
        ----------------------------------------
        Row fields:
            'locus': locus<GRCh37>
            'alleles': array<str>
            'rsid': str
            'qual': float64
        ----------------------------------------
        Entry fields:
            'GT': call
            'AD': array<int32>
            'DP': int32
            'GQ': int32
            'PL': array<int32>
        ----------------------------------------
        Column key: ['s']
        Row key: ['locus', 'alleles']
        Partition key: ['locus']
        ----------------------------------------

:**dependencies**: :meth:`.MatrixTable.describe`

Table
.....

:**description**: Get information about the fields and keys of a table.

:**code**:

    .. code-block:: python

        >>> ht.describe()  # doctest: +SKIP_OUTPUT_CHECK
        ----------------------------------------
        Global fields:
            None
        ----------------------------------------
        Row fields:
            'locus': locus<GRCh37>
            'alleles': array<str>
        ----------------------------------------
        Key: ['locus', 'alleles']
        ----------------------------------------

:**dependencies**: :meth:`.Table.describe`

Expression
..........

:**description**: Get information about a specific field in a table or matrix table.

:**code**:

    .. code-block:: python

        >>> mt.s.describe()  # doctest: +SKIP_OUTPUT_CHECK
        --------------------------------------------------------
        Type:
            str
        --------------------------------------------------------
        Source:
            <hail.matrixtable.MatrixTable object at 0x60e42f518>
        Index:
            ['column']
        --------------------------------------------------------

:**dependencies**: :meth:`.Expression.describe`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            We can select fields from a table or matrix table with an expression like
            ``mt.s``. Then we can call the :meth:`.Expression.describe` method on the
            expression to get information about the expression's type, indices, and source.

View your data locally
~~~~~~~~~~~~~~~~~~~~~~

Table
.....

:**description**: View the first n rows of a table.

:**code**:

    >>> ht.show(5)
    +-------+-------+-----+-------+-------+-------+-------+-------+
    |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |
    +-------+-------+-----+-------+-------+-------+-------+-------+
    | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 |
    +-------+-------+-----+-------+-------+-------+-------+-------+
    |     1 |    65 | "M" |     5 |     4 |     2 |    50 |     5 |
    |     2 |    72 | "M" |     6 |     3 |     2 |    61 |     1 |
    |     3 |    70 | "F" |     7 |     3 |    10 |    81 |    -5 |
    |     4 |    60 | "F" |     8 |     2 |    11 |    90 |   -10 |
    +-------+-------+-----+-------+-------+-------+-------+-------+

:**dependencies**: :meth:`.Table.show`

Matrix Table
............

:**description**: View the columns, rows, or entries of a matrix table.

:**code**:

    >>> mt.rows().show()
    >>> mt.cols().show()
    >>> mt.entries().show()

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            Unlike tables, matrix tables do not have a ``show`` method, but you can call
            :meth:`.Table.show` on the :meth:`.MatrixTable.rows` table,
            :meth:`.MatrixTable.cols` table, or :meth:`.MatrixTable.entries` table of your
            matrix table.

:**dependencies**: :meth:`.Table.show`, :meth:`.MatrixTable.rows`, :meth:`.MatrixTable.cols`, :meth:`.MatrixTable.entries`

Expression
..........

:**description**: View an expression.

:**code**:

    >>> mt.rsid.show()
    +---------------+------------+---------------+
    | locus         | alleles    | rsid          |
    +---------------+------------+---------------+
    | locus<GRCh37> | array<str> | str           |
    +---------------+------------+---------------+
    | 20:12990057   | ["T","A"]  | "rs3761894"   |
    | 20:13029862   | ["C","T"]  | "rs919604"    |
    | 20:13074235   | ["G","A"]  | "rs708937"    |
    | 20:13140720   | ["G","A"]  | "rs61738161"  |
    | 20:13695498   | ["G","A"]  | "rs6079146"   |
    | 20:13714384   | ["A","C"]  | "rs41275402"  |
    | 20:13765944   | ["C","G"]  | NA            |
    | 20:13765954   | ["C","T"]  | "rs113805278" |
    | 20:13845987   | ["C","T"]  | "rs761811"    |
    | 20:16223957   | ["T","C"]  | "rs1000121"   |
    +---------------+------------+---------------+
    showing top 10 rows
    <BLANKLINE>

:**dependencies**: :meth:`.Expression.show`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            ``mt.rsid`` is an expression that references a field of ``mt``. We
            can call :meth:`.Expression.show` to display the first n values
            referenced by the expression. Since ``mt.rsid`` is indexed by row,
            the row key fields ``locus`` and ``alleles`` will also be displayed.
