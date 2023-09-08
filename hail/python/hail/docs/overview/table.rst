--------------
Table Overview
--------------

A :class:`.Table` is the Hail equivalent of a SQL table, a Pandas Dataframe, an
R Dataframe, a dyplr Tibble, or a Spark Dataframe. It consists of rows of data
conforming to a given schema where each column (row field) in the dataset is of
a specific type.

Import
======

Hail has functions to create tables from a variety of data sources.
The most common use case is to load data from a TSV or CSV file, which can be
done with the :func:`.import_table` function.

    >>> ht = hl.import_table("data/kt_example1.tsv", impute=True)

Examples of genetics-specific import methods are
:func:`.import_locus_intervals`, :func:`.import_fam`, and :func:`.import_bed`.
Many Hail methods also return tables.

An example of a table is below. We recommend `ht` as a variable name for
tables, referring to a "Hail table".

    >>> ht.show()
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

Global Fields
=============

In addition to row fields, Hail tables also have global fields. You can think of
globals as extra fields in the table whose values are identical for every row.
For example, the same table above with the global field ``G = 5`` can be thought
of as

.. code-block:: text

    +-------+-------+-----+-------+-------+-------+-------+-------+-------+
    |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |     G |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+
    | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 | int32 |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+
    |     1 |    65 | M   |     5 |     4 |     2 |    50 |     5 |     5 |
    |     2 |    72 | M   |     6 |     3 |     2 |    61 |     1 |     5 |
    |     3 |    70 | F   |     7 |     3 |    10 |    81 |    -5 |     5 |
    |     4 |    60 | F   |     8 |     2 |    11 |    90 |   -10 |     5 |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+

but the value ``5`` is only stored once for the entire dataset and NOT once per
row of the table. The output of :meth:`.Table.describe` lists what all of the row
fields and global fields are.

    >>> ht.describe()  # doctest: +SKIP_OUTPUT_CHECK
    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'ID': int32
        'HT': int32
        'SEX': str
        'X': int32
        'Z': int32
        'C1': int32
        'C2': int32
        'C3': int32
    ----------------------------------------
    Key:
        None
    ----------------------------------------

Keys
====

Row fields can be specified to be the key of the table with the method
:meth:`.Table.key_by`. Keys are important for joining tables together (discussed
below).

Referencing Fields
==================

Each :class:`.Table` object has all of its row fields and global fields as
attributes in its namespace. This means that the row field `ID` can be accessed
from table `ht` with ``ht.Sample`` or ``ht['Sample']``. If `ht` also had a
global field `G`, then it could be accessed by either ``ht.G`` or ``ht['G']``.
Both row fields and global fields are top level fields. Be aware that accessing
a field with the dot notation will not work if the field name has spaces or
special characters in it. The Python type of each attribute is an
:class:`.Expression` that also contains context about its type and source, in
this case a row field of table `ht`.

    >>> ht  # doctest: +SKIP_OUTPUT_CHECK
    <hail.table.Table at 0x110791a20>

    >>> ht.ID  # doctest: +SKIP_OUTPUT_CHECK
    <Int32Expression of type int32>


Updating Fields
===============

Add or remove row fields from a Table with :meth:`.Table.select` and
:meth:`.Table.drop`.

    >>> ht.drop('C1', 'C2')
    >>> ht.drop(*['C1', 'C2'])

    >>> ht.select(ht.ID, ht.SEX)
    >>> ht.select(*['ID', 'C3'])

Use :meth:`.Table.annotate` to add new row fields or update the values of
existing row fields and use :meth:`.Table.filter` to either keep or remove
rows based on a condition:

    >>> ht_new = ht.filter(ht['C1'] >= 10)
    >>> ht_new = ht_new.annotate(id_times_2 = ht_new.ID * 2)


Aggregation
===========

To compute an aggregate statistic over the rows of
a dataset, Hail provides an :meth:`.Table.aggregate` method which can be passed
a wide variety of aggregator functions (see :ref:`sec-aggregators`):

    >>> ht.aggregate(hl.agg.fraction(ht.SEX == 'F'))
    0.5

We also might want to compute the mean value of `HT` for each sex. This is
possible with a combination of :meth:`.Table.group_by` and
:meth:`.GroupedTable.aggregate`:

    >>> ht_agg = (ht.group_by(ht.SEX)
    ...             .aggregate(mean = hl.agg.mean(ht.HT)))
    >>> ht_agg.show()
    +-----+----------+
    | SEX |     mean |
    +-----+----------+
    | str |  float64 |
    +-----+----------+
    | "F" | 6.50e+01 |
    | "M" | 6.85e+01 |
    +-----+----------+

Note that the result of ``ht.group_by(...).aggregate(...)`` is a new
:class:`.Table` while the result of ``ht.aggregate(...)`` is a Python value.

Joins
=====

To join the row fields of two tables together, Hail provides a
:meth:`.Table.join` method with options for how to join the rows together (left,
right, inner, outer). The tables are joined by the row fields designated as
keys. The number of keys and their types must be identical between the two
tables. However, the names of the keys do not need to be identical. Use the
:attr:`.Table.key` attribute to view the current table row keys and the
:meth:`.Table.key_by` method to change the table keys. If top level row field
names overlap between the two tables, the second table's field names will be
appended with a unique identifier "_N".

    >>> ht = ht.key_by('ID')
    >>> ht2 = hl.import_table("data/kt_example2.tsv", impute=True).key_by('ID')

    >>> ht_join = ht.join(ht2)
    >>> ht_join.show(width=120)
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+----------+
    |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |     A | B        |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+----------+
    | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 | int32 | str      |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+----------+
    |     1 |    65 | "M" |     5 |     4 |     2 |    50 |     5 |    65 | "cat"    |
    |     2 |    72 | "M" |     6 |     3 |     2 |    61 |     1 |    72 | "dog"    |
    |     3 |    70 | "F" |     7 |     3 |    10 |    81 |    -5 |    70 | "mouse"  |
    |     4 |    60 | "F" |     8 |     2 |    11 |    90 |   -10 |    60 | "rabbit" |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+----------+
    <BLANKLINE>

In addition to the :meth:`.Table.join` method, Hail provides another
join syntax using Python's bracket indexing syntax. The syntax looks like
``right_table[left_table.key]``, which will return an :class:`.Expression`
instead of a :class:`.Table`. This expression is a dictionary mapping the
keys in the left table to the rows in the right table.
We can annotate the left table with this expression to perform a left join:
``left_table.annotate(x = right_table[left_table.key].x]``. For example, below
we add the field 'B' from `ht2` to `ht`:

    >>> ht1 = ht.annotate(B = ht2[ht.ID].B)
    >>> ht1.show(width=120)
    +-------+-------+-----+-------+-------+-------+-------+-------+----------+
    |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 | B        |
    +-------+-------+-----+-------+-------+-------+-------+-------+----------+
    | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 | str      |
    +-------+-------+-----+-------+-------+-------+-------+-------+----------+
    |     1 |    65 | "M" |     5 |     4 |     2 |    50 |     5 | "cat"    |
    |     2 |    72 | "M" |     6 |     3 |     2 |    61 |     1 | "dog"    |
    |     3 |    70 | "F" |     7 |     3 |    10 |    81 |    -5 | "mouse"  |
    |     4 |    60 | "F" |     8 |     2 |    11 |    90 |   -10 | "rabbit" |
    +-------+-------+-----+-------+-------+-------+-------+-------+----------+

Interacting with Tables Locally
===============================

Hail has many useful methods for interacting with tables locally such as in an
Jupyter notebook. Use the :meth:`.Table.show` method to see the first few rows
of a table.

:meth:`.Table.take` will collect the first `n` rows of a table into a local
Python list:

    >>> first3 = ht.take(3)
    >>> first3
    [Struct(ID=1, HT=65, SEX='M', X=5, Z=4, C1=2, C2=50, C3=5),
     Struct(ID=2, HT=72, SEX='M', X=6, Z=3, C1=2, C2=61, C3=1),
     Struct(ID=3, HT=70, SEX='F', X=7, Z=3, C1=10, C2=81, C3=-5)]

Note that each element of the list is a :class:`.Struct` whose elements can be
accessed using Python's get attribute or get item notation:

    >>> first3[0].ID
    1

    >>> first3[0]['ID']
    1

The :meth:`.Table.head` method is helpful for testing pipelines. It subsets a
table to the first `n` rows, causing downstream operations to run much more
quickly.

:meth:`.Table.describe` is a useful method for showing all of the fields of the
table and their types. The types themselves can be accessed using the fields
(e.g. ``ht.ID.dtype``), and the full row and global types can be accessed with
``ht.row.dtype`` and ``ht.globals.dtype``. The row fields that are part of the
key can be accessed with :attr:`.Table.key`. The :meth:`.Table.count` method
returns the number of rows.
