.. _how_to_basics:

Basic Methods for Working with Hail Data
========================================

Get Data Into and Out of Hail
-----------------------------

Import
~~~~~~

To import data in a non-Hail format into a Hail format (e.g. a Table,
MatrixTable), use one of the
:ref:`import_* <methods_impex_import>` methods.

>>> table = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')

Export
~~~~~~

To export Hail data to a non-Hail format, use one of the
:ref:`export_* <methods_impex_export>` methods.

>>> hl.export_vcf(mt, 'output/example') # doctest: +SKIP

Write
~~~~~

Once you have your data in a Hail format, you can write it to disk using one of
the write() methods, e.g. :meth:`.Table.write` or :meth:`.MatrixTable.write`.

>>> mt.write('output/example.mt') # doctest: +SKIP

Read
~~~~

If you wrote a Hail Table or MatrixTable to disk using one of Hail's write()
methods, you can read it using one of the
:ref:`read <methods_impex_read>` methods.

>>> ht = hl.read_table('data/example.ht') # doctest: +SKIP


Examine your data
-----------------

Explore the schema
~~~~~~~~~~~~~~~~~~

Get information about the fields and keys of a matrix table using
:meth:`.MatrixTable.describe`.

.. code-block:: python

    >>> mt.describe()
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

Get information about the fields and keys of a table using
:meth:`.Table.describe`.

.. code-block:: python

    >>> ht.describe()
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

We can also select fields from a table or matrix table with an expression like
``mt.s``. Then we can call the :meth:`.Expression.describe` method on the
expression to get information about the expression's type, indices, and source:

.. code-block:: python

    >>> mt.s.describe()
    --------------------------------------------------------
    Type:
        str
    --------------------------------------------------------
    Source:
        <hail.matrixtable.MatrixTable object at 0x60e42f518>
    Index:
        ['column']
    --------------------------------------------------------

View your data locally
~~~~~~~~~~~~~~~~~~~~~~

The :meth:`.Table.show` method can be used to view the first n rows of a
dataset. The default number of rows shown is 10. You should only try to view
the entire table if your table is small.

>>> ht.show(5)
+-------+-------+-----+-------+-------+-------+-------+-------+
|    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |
+-------+-------+-----+-------+-------+-------+-------+-------+
| int32 | int32 | str | int32 | int32 | int32 | int32 | int32 |
+-------+-------+-----+-------+-------+-------+-------+-------+
|     1 |    65 | M   |     5 |     4 |     2 |    50 |     5 |
|     2 |    72 | M   |     6 |     3 |     2 |    61 |     1 |
|     3 |    70 | F   |     7 |     3 |    10 |    81 |    -5 |
|     4 |    60 | F   |     8 |     2 |    11 |    90 |   -10 |
+-------+-------+-----+-------+-------+-------+-------+-------+

Matrix Tables do not have a ``show`` method, but you can call
:meth:`.Table.show` on the :meth:`.MatrixTable.rows` table,
:meth:`.MatrixTable.cols` table, or :meth:`.MatrixTable.entries` table of your
matrix table:

>>> mt.rows().show()
>>> mt.cols().show()
>>> mt.entries().show()

The :meth:`.Expression.show` method can also be called on an expression that
references fields from a table or matrix table like so:

>>> mt.rsid.show()
+---------------+--------------+-------------+
| locus         | alleles      | rsid        |
+---------------+--------------+-------------+
| locus<GRCh37> | array<str>   | str         |
+---------------+--------------+-------------+
| 20:10579373   | ["C","T"]    | rs78689061  |
| 20:13695607   | ["T","G"]    | rs34414644  |
| 20:13698129   | ["G","A"]    | rs78509779  |
| 20:14306896   | ["G","A"]    | rs6042672   |
| 20:14306953   | ["G","T"]    | rs6079391   |
| 20:15948325   | ["AG","A"]   | NA          |
| 20:15948326   | ["GAAA","G"] | NA          |
| 20:17479423   | ["T","C"]    | rs185188648 |
| 20:17600357   | ["G","A"]    | rs11960     |
| 20:17640833   | ["A","C"]    | NA          |
+---------------+--------------+-------------+


