.. _sec-overview:

.. py:currentmodule:: hail

========
Overview
========

Hail is...

  - a library for analyzing structured tabular and matrix data
  - a collection of primitives for operating on data in parallel
  - a suite of functionality for processing genetic data
  - *not* an acronym

-----
Types
-----

In Python, ``5`` is of type :obj:`int` while ``"hello"`` is of type :obj:`str`.
Python is a dynamically-typed language, meaning that a function like:

    >>> def add_x_and_y(x, y):
    ...     return x + y

can be called on any two objects which can be added, like numbers, strings, or
:mod:`numpy` arrays.

Types are very important in Hail, because the fields of :class:`.Table` and
:class:`.MatrixTable` objects have data types.

Hail has basic data types for numeric and string objects:

 - :py:data:`.tstr` - Text string.
 - :py:data:`.tbool` - Boolean (``True`` or ``False``) value.
 - :py:data:`.tint32` - 32-bit integer.
 - :py:data:`.tint64` - 64-bit integer.
 - :py:data:`.tfloat32` - 32-bit floating point number.
 - :py:data:`.tfloat64` - 64-bit floating point number.

Hail has genetics-specific types:

 - :py:data:`.tcall` - Genotype calls.
 - :class:`.tlocus` - Genomic locus, parameterized by reference genome.

Hail has container types:

 - :class:`.tarray` - Ordered collection of homogenous objects.
 - :class:`.tset` - Unordered collection of distinct homogenous objects.
 - :class:`.tdict` - Key-value map. Keys and values are both homogenous.
 - :class:`.ttuple` - Tuple of heterogeneous values.
 - :class:`.tstruct` - Structure containing named fields, each with its own
   type.

Homogenous collections are a change from standard Python collections.
While the list ``['1', 2, 3.0]`` is a perfectly valid Python list,
a Hail array could not contain both :py:data:`.tstr` and :py:data:`.tint32`
objects. Likewise, a the :obj:`dict` ``{'a': 1, 2: 'b'}`` is a valid Python
dictionary, but a Hail dictionary cannot contain keys of different types.
An example of a valid dictionary is ``{'a': 1, 'b': 2}``, where the keys are all
strings and the values are all integers. The type of this dictionary would be
``dict<str, int32>``.

The :class:`.tstruct` type is used to compose types together to form nested
structures. The :class:`.tstruct` is an ordered mapping from field name to field
type. Each field name must be unique.


-----------
Expressions
-----------

The Python language allows users to specify their computations using expressions.
For example, a simple expression is ``5 + 6``. This will be evaluated and return
``11``. You can also assign expressions to variables and then add variable expressions
together such as ``x = 5; y = 6; x + y``.

Throughout Hail documentation and tutorials, you will see Python code like this:

    >>> ht2 = ht.annotate(C4 = ht.C3 + 3 * ht.C2 ** 2)

However, Hail is not running Python code on your data. Instead, Hail is keeping
track of the computations applied to your data, then compiling these computations
into native code and running them in parallel.

This happens using the :class:`.Expression` class. Hail expressions operate much
like Python objects of the same type: for example, an :class:`.Int32Expression`
can be used in arithmetic with other integers or expressions in much the same
way a Python :obj:`int` can. However, you will be unable to use these
expressions with other modules, like :mod:`numpy` or :mod:`scipy`.

:class:`.Expression` objects keep track of their data type. This can be accessed
with :meth:`.Expression.dtype`:

    >>> i = hl.int32(100)
    >>> i.dtype
    dtype('int32')

The Hail equivalent of the Python example above would be as follows:

    >>> x = hl.int32(5)
    >>> y = hl.int32(6)

We can print `x` in a Python interpreter and see that `x` is an :class:`.Int32Expression`.
This makes sense because `x`  is a Python :obj:`int`.

    >>> x
    <Int32Expression of type int32>

We can add two :class:`.Int32Expression` objects together just like with Python
:obj:`int` objects. ``x + y`` returns another :class:`.Int32Expression` representing
the computation of ``x + y`` and not an actual value.

    >>> z = x + y
    >>> z
    <Int32Expression of type int32>

To peek at the value of this computation, there are two options:
:meth:`.Expression.value`, which returns a Python value, and
:meth:`.Expression.show`, which prints a human-readable representation of an
expression.

    >>> z.value
    11
    >>> z.show()
    +--------+
    | <expr> |
    +--------+
    |  int32 |
    +--------+
    |     11 |
    +--------+

Expressions like to bring Python objects into the world of expressions as well.
For example, we can add a Python :obj:`int` to an :class:`.Int32Expression`.

    >>> x + 3
    <Int32Expression of type int32>

Addition is commutative, so we can also add an :class:`.Int32Expression` to an
:obj:`int`.

    >>> 3 + x
    <Int32Expression of type int32>

Hail has many subclasses of :class:`.Expression` -- one for each Hail type. Each
subclass defines possible methods and operations that can be applied. For example,
if we have a list of Python integers, we can convert this to a Hail
:class:`.ArrayNumericExpression` with either :func:`.array` or :func:`.literal`:

    >>> a = hl.array([1, 2, -3, 0, 5])
    >>> a
    <ArrayNumericExpression of type array<int32>>

    >>> a.dtype
    dtype('array<int32>')

Hail arrays can be indexed and sliced like Python lists or :mod:`numpy` arrays:

    >>> a[1]
    >>> a[1:-1]


Boolean Logic
=============

Unlike Python, a Hail :class:`.BooleanExpression` cannot be used with ``and``,
``or``, and ``not``. The equivalents are ``&``, ``|``, and ``~``.

    >>> s1 = x == 3
    >>> s2 = x != 4

    >>> s1 & s2 # s1 and s2
    >>> s1 | s2 # s1 or s2
    >>> ~s1 # not s1

.. caution::

    The operator precedence of ``&`` and ``|`` is different from ``and`` and
    ``or``. You will need parentheses around expressions like this:

    >>> (x == 3) & (x != 4)

Conditionals
============

Python ``if`` / ``else`` do not work with Hail expressions. Instead, you must
use the :func:`.cond`, :func:`.case`, and :func:`.switch` functions.

A conditional expression has three components: the condition to evaluate, the
consequent value to return if the condition is ``True``, and the alternate to
return if the condition is ``False``. For example:

.. code-block:: python

    if (x > 0):
        return 1
    else:
        return 0


In the above conditional, the condition is ``x > 0``, the consequent is ``1``,
and the alternate is ``0``.

Here is the Hail expression equivalent with :func:`.cond`:

    >>> hl.cond(x > 0, 1, 0)
     <Int32Expression of type int32>

This example returns an :class:`.Int32Expression` which can be used in more
computations:

    >>> a + hl.cond(x > 0, 1, 0)
    <ArrayNumericExpression of type array<int32>>

More complicated conditional statements can be constructed with :func:`.case`.
For example, we might want to emit ``1`` if ``x < -1``, ``2`` if
``-1 <= x <= 2`` and ``3`` if ``x > 2``.

    >>> (hl.case()
    ...   .when(x < -1, 1)
    ...   .when((x >= -1) & (x <= 2), 2)
    ...   .when(x > 2, 3)
    ...   .or_missing())
    <Int32Expression of type int32>

Finally, Hail has the :func:`.switch` function to build a conditional tree based
on the value of an expression. In the example below, `csq` is a
:class:`.StringExpression` representing the functional consequence of a
mutation. If `csq` does not match one of the cases specified by
:meth:`.SwitchBuilder.when`, it is set to missing with
:meth:`.SwitchBuilder.or_missing`. Other switch statements are documented in the
:class:`.SwitchBuilder` class.

    >>> csq = hl.str('nonsense')

    >>> (hl.switch(csq)
    ...    .when("synonymous", False)
    ...    .when("intron", False)
    ...    .when("nonsense", True)
    ...    .when("indel", True)
    ...    .or_missing())
    <BooleanExpression of type bool>


Missingness
===========

In Hail, all expressions can be missing.
An expression representing a missing value of a given type can be generated with
the :func:`.null` function, which takes the type as its single argument. An
example of generating a :class:`.Float64Expression` that is missing is:

    >>> hl.null('float64')

These can be used with conditional statements to set values to missing if they
don't satisfy a condition:

    >>> hl.cond(x > 2.0, x, hl.null(hl.tfloat))

The result of method calls on a missing value is ``None``. For example, if
we define ``cnull`` to be a missing value with type :class:`.tcall`, calling
the method `is_het` will return ``None`` and not ``False``.

    >>> cnull = hl.null('call')
    >>> cnull.is_het().value
    None


Binding Variables
=================

Hail inlines function calls each time an expression appears. This can result
in unexpected behavior when random values are used. For example, let `x` be
a random number generated with the function :func:`.rand_unif`:

    >>> x = hl.rand_unif(0, 1)

The value of `x` changes with each evaluation:

    >>> x.value
    0.4678132874101748

    >>> x.value
    0.9097632224065403

If we create a list with x repeated 3 times, we'd expect to get an array with identical
values. However, instead we see a list of 3 random numbers.

    >>> hl.array([x, x, x]).value
    [0.8846327207915881, 0.14415148553468504, 0.8202677741734825]

To solve this problem, we can use the :func:`.bind` function to bind an expression to a
value before applying it in a function.

    >>> expr = hl.bind(lambda x: [x, x, x], hl.rand_unif(0, 1))

    >>> expr.value
    [0.5562065047992025, 0.5562065047992025, 0.5562065047992025]


Functions
=========

In addition to the methods exposed on each :class:`.Expression`, Hail also has
numerous functions that can be applied to expressions, which also return an expression.

Take a look at the :ref:`sec-functions` page for full documentation.

-----
Table
-----

A :class:`.Table` is the Hail equivalent of a SQL table, a Pandas Dataframe, an
R Dataframe, a dyplr Tibble, or a Spark Dataframe. It consists of rows of data
conforming to a given schema where each column (row field) in the dataset is of
a specific type.

Import
======

Hail has functions to create tables from a variety of data sources.
The most common use case is to load data from a TSV or CSV file, which can be
done with the :func:`import_table` function.

    ht = hl.import_table("data/kt_example1.tsv", impute=True)

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
    |     1 |    65 | M   |     5 |     4 |     2 |    50 |     5 |
    |     2 |    72 | M   |     6 |     3 |     2 |    61 |     1 |
    |     3 |    70 | F   |     7 |     3 |    10 |    81 |    -5 |
    |     4 |    60 | F   |     8 |     2 |    11 |    90 |   -10 |
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

    >>> ht.describe()
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

    >>> ht
    <hail.table.Table at 0x110791a20>

    >>> ht.ID
    <Int32Expression of type int32>


Common Operations
=================

The main operations on a table are :meth:`.Table.select` and :meth:`.Table.drop` to add or remove row fields,
:meth:`.Table.filter` to either keep or remove rows based on a condition, and :meth:`.Table.annotate` to add
new row fields or update the values of existing row fields. For example:

    >>> ht_new = ht.filter(ht['C1'] >= 10)
    >>> ht_new = ht_new.annotate(id_times_2 = ht_new.ID * 2)


Aggregation
===========

A commonly used operation is to compute an aggregate statistic over the rows of
the dataset. Hail provides an :meth:`.Table.aggregate` method along with many
aggregator functions (see :ref:`sec-aggregators`) to return the result of a
query:

    >>> ht.aggregate(agg.fraction(ht.SEX == 'F'))
    0.5

We also might want to compute the mean value of `HT` for each sex. This is
possible with a combination of :meth:`Table.group_by` and
:meth:`.GroupedTable.aggregate`:

    >>> ht_agg = (ht.group_by(ht.SEX)
    ...             .aggregate(mean = agg.mean(ht.HT)))
    >>> ht_agg.show()
    +-----+-------------+
    | SEX |        mean |
    +-----+-------------+
    | str |     float64 |
    +-----+-------------+
    | M   | 6.85000e+01 |
    | F   | 6.50000e+01 |
    +-----+-------------+


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
    >>> ht_join.show()
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+--------+
    |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |     A | B      |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+--------+
    | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 | int32 | str    |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+--------+
    |     3 |    70 | F   |     7 |     3 |    10 |    81 |    -5 |    70 | mouse  |
    |     4 |    60 | F   |     8 |     2 |    11 |    90 |   -10 |    60 | rabbit |
    |     2 |    72 | M   |     6 |     3 |     2 |    61 |     1 |    72 | dog    |
    |     1 |    65 | M   |     5 |     4 |     2 |    50 |     5 |    65 | cat    |
    +-------+-------+-----+-------+-------+-------+-------+-------+-------+--------+

In addition to using the :meth:`.Table.join` method, Hail provides an additional
join syntax using Python's bracket notation. This syntax does a left join, like
looking up values in a dictionary. Instead of returning a :class:`.Table`, this
syntax returns an :class:`.Expression` which can be used in expressions of the
left table. For example, below we add the field 'B' from `ht2` to `ht`:

    >>> ht1 = ht.annotate(B = ht2[ht.ID].B)
    >>> ht1.show()
    +-------+-------+-----+-------+-------+-------+-------+-------+--------+
    |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 | B      |
    +-------+-------+-----+-------+-------+-------+-------+-------+--------+
    | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 | str    |
    +-------+-------+-----+-------+-------+-------+-------+-------+--------+
    |     3 |    70 | F   |     7 |     3 |    10 |    81 |    -5 | mouse  |
    |     4 |    60 | F   |     8 |     2 |    11 |    90 |   -10 | rabbit |
    |     2 |    72 | M   |     6 |     3 |     2 |    61 |     1 | dog    |
    |     1 |    65 | M   |     5 |     4 |     2 |    50 |     5 | cat    |
    +-------+-------+-----+-------+-------+-------+-------+-------+--------+

Interacting with Tables Locally
===============================

Hail has many useful methods for interacting with tables locally such as in an
Jupyter notebook. Use the :meth:`.Table.show` method to see the first few rows
of a table.

:meth:`.Table.take` will collect the first `n` rows of a table into a local
Python list:

    >>> first3 = ht.take(3)
    >>> first3
    [Struct(ID=3, HT=70, SEX=F, X=7, Z=3, C1=10, C2=81, C3=-5),
     Struct(ID=4, HT=60, SEX=F, X=8, Z=2, C1=11, C2=90, C3=-10),
     Struct(ID=2, HT=72, SEX=M, X=6, Z=3, C1=2, C2=61, C3=1)]

Note that each element of the list is a :class:`.Struct` whose elements can be
accessed using Python's get attribute or get item notation:

    >>> first3[0].ID
    3

    >>> first3[0]['ID']
    3

The :meth:`.Table.head` method is helpful for testing pipelines. It subsets a
table to the first `n` rows, causing downstream operations to run much more
quickly.

:meth:`.Table.describe` is a useful method for showing all of the fields of the
table and their types. The types themselves can be accessed using the fields
(e.g. ``ht.ID.dtype``), and the full row and global types can be accessed with
``ht.row.dtype`` and ``ht.globals.dtype``. The row fields that are part of the
key can be accessed with :attr:`.Table.key`. The :meth:`.Table.count` method
returns the number of rows.

Export
======

Hail provides multiple methods to export data to other formats. Tables can be
exported to TSV files with the :meth:`.Table.export` method or written to disk
in Hail's on-disk format with :meth:`.Table.write` (these files may be read in
with :func:`.read_table`). Tables can also be exported to :mod:`pandas`
DataFrames with :meth:`.Table.to_pandas` or to :mod:`.pyspark` Dataframes with
:meth:`.Table.to_spark`.

-----------
MatrixTable
-----------

A :class:`.MatrixTable` is a distributed two-dimensional dataset consisting of
four components: a two-dimensional matrix where each entry is indexed by row
key(s) and column key(s), a corresponding rows table that stores all of the row
fields which are constant for every column in the dataset, a corresponding
columns table that stores all of the column fields that are constant for every
row in the dataset, and a set of global fields that are constant for every entry
in the dataset.

Unlike a :class:`.Table` which has two field groups (row fields and global
fields), a matrix table has four field groups: global fields, row fields, column
fields, entry fields.

In addition, there are different operations on the matrix for each field group.
For instance, :class:`.Table` has :meth:`.Table.select` and
:meth:`.Table.select_globals`, and :class:`.MatrixTable` has
:meth:`.MatrixTable.select_rows`, :meth:`.MatrixTable.select_cols`,
:meth:`.MatrixTable.select_entries`, and :meth:`.MatrixTable.select_globals`.

It is possible to represent matrix data by coordinate in a table , storing one
record per entry of the matrix. However, the :class:`.MatrixTable` represents
this data far more efficiently and exposes natural interfaces for computing on
it.

The :meth:`.MatrixTable.rows` and :meth:`.MatrixTable.cols` methods return the
row and column fields as separate tables. The :meth:`.MatrixTable.entries`
method returns the matrix as a table in coordinate form -- use this object with
caution.

Keys
====

Matrix tables have keys just as tables do. However, instead of one key, matrix
tables have two keys: a row key and a column key. Row fields are indexed by the
row key, column fields are indexed by the column key, and entry fields are
indexed by the row key and the column key. The key structs can be accessed with
:attr:`.MatrixTable.row_key` and :attr:`.MatrixTable.col_key`. It is possible to
change the key with :meth:`.MatrixTable.key_rows_by` and
:meth:`.MatrixTable.key_cols_by`.

Note that changing the row key, however, may be an expensive operation.

Hail matrix tables are natively distributed objects, and as such have another
key: a partition key. This key is used for specifying the ordering of the matrix
table along the row dimension, which is important for performance. Access this
with :attr:`.MatrixTable.partition_key`

Referencing Fields
==================

All fields (row, column, global, entry) are top-level and exposed as attributes
on the :class:`.MatrixTable` object. For example, if the matrix table `mt` had a
row field `locus`, this field could be referenced with either ``mt.locus`` or
``mt['locus']``. The former access pattern does not work with field names with
spaces or punctuation.

The result of referencing a field from a matrix table is an :class:`.Expression`
which knows its type and knows its source as well as whether it is a row field,
column field, entry field, or global field. Hail uses this context to know which
operations are allowed for a given expression.

When evaluated in a Python interpreter, we can see ``mt.locus`` is a
:class:`.LocusExpression` with type ``locus<GRCh37>`` and it is a row field of
the MatrixTable `mt`.

    >>> mt
    <hail.matrixtable.MatrixTable at 0x1107e54a8>

    >>> mt.locus
    <LocusExpression of type locus<GRCh37>>

Likewise, ``mt.DP`` would be an :class:`.Int32Expression` with type ``int32``
and is an entry field of `mt`. It is indexed by both rows and columns as denoted
by its indices when describing the expression:

    >>> mt.DP.describe()
    --------------------------------------------------------
    Type:
        int32
    --------------------------------------------------------
    Source:
        <class 'hail.matrixtable.MatrixTable'>
    Index:
        ['row', 'column']
    --------------------------------------------------------

Import
======

Text files may be imported with :func:`.import_matrix_table`. Additionally, Hail
provides functions to import genetic datasets as matrix tables from a
variety of file formats: :func:`.import_vcf`, :func:`.import_plink`,
:func:`.import_bgen`, and :func:`.import_gen`.

    >>> mt = hl.import_vcf('data/sample.vcf.bgz')

The :meth:`.MatrixTable.describe` method prints all fields in the table and
their types, as well as the keys.

    >>> mt.describe()
    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Column fields:
        's': str
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'rsid': str
        'qual': float64
        'filters': set<str>
        'info': struct {
            NEGATIVE_TRAIN_SITE: bool,
            AC: array<int32>,
            ...
            DS: bool
        }
    ----------------------------------------
    Entry fields:
        'GT': call
        'AD': array<int32>
        'DP': int32
        'GQ': int32
        'PL': array<int32>
    ----------------------------------------
    Column key:
        's': str
    Row key:
        'locus': locus<GRCh37>
        'alleles': array<str>
    Partition key:
        'locus': locus<GRCh37>
    ----------------------------------------

Common Operations
=================

Like tables, Hail provides a number of useful methods for manipulating data in a
matrix table.

**Filter**

:class:`.MatrixTable` has three methods to filter based on expressions:

- :meth:`.MatrixTable.filter_rows`
- :meth:`.MatrixTable.filter_cols`
- :meth:`.MatrixTable.filter_entries`

Filter methods take a :class:`.BooleanExpression` argument. These expressions
are generated by applying computations to the fields of the matrix table:

    >>> filt_mt = mt.filter_rows(hl.len(mt.alleles) == 2)

    >>> filt_mt = mt.filter_cols(hl.agg.mean(mt.GQ) < 20)

    >>> filt_mt = mt.filter_entries(mt.DP < 5)

These expressions can compute arbitrarily over the data: the :meth:`.MatrixTable.filter_cols`
example above aggregates entries per column of the matrix table to compute the
mean of the `GQ` field, and removes columns where the result is smaller than 20.

**Annotate**

:class:`.MatrixTable` has four methods to add new fields or update existing fields:

- :meth:`.MatrixTable.annotate_rows`
- :meth:`.MatrixTable.annotate_cols`
- :meth:`.MatrixTable.annotate_entries`
- :meth:`.MatrixTable.annotate_globals`

Annotate methods take keyword arguments where the key is the name of the new
field to add and the value is an expression specifying what should be added.

The simplest example is adding a new global field `foo` that just contains the constant
5.

    >>> mt_new = mt.annotate_globals(foo = 5)
    >>> print(mt.globals.dtype.pretty())
    struct {
        foo: int32
    }

Another example is adding a new row field `call_rate` which computes the fraction
of non-missing entries `GT` per row:

    >>> mt_new = mt.annotate_rows(call_rate = hl.agg.fraction(hl.is_defined(mt.GT)))

Annotate methods are also useful for updating values. For example, to update the
GT entry field to be missing if `GQ` is less than 20, we can do the following:

    >>> mt_new = mt.annotate_entries(GT = hl.case()
    ...                                     .when(mt.GQ >= 20, mt.GT)
    ...                                     .or_missing())

**Select**

Select is used to create a new schema for a dimension of the matrix table. For
example, following the matrix table schemas from importing a VCF file (shown above),
to create a hard calls dataset where each entry only contains the `GT` field
one can do the following:

    >>> mt_new = mt.select_entries('GT')
    >>> print(mt_new.entry.dtype.pretty())
    struct {
        GT: call
    }

:class:`.MatrixTable` has four select methods that select and create new fields:

- :meth:`.MatrixTable.select_rows`
- :meth:`.MatrixTable.select_cols`
- :meth:`.MatrixTable.select_entries`
- :meth:`.MatrixTable.select_globals`

Each method can take either strings referring to top-level fields, an attribute
reference (useful for accessing nested fields), as well as keyword arguments
``KEY=VALUE`` to compute new fields. The Python unpack operator ``**`` can be
used to specify that all fields of a Struct should become top level fields.
However, be aware that all top-level field names must be unique. In this
example, `**mt['info']` would fail because `DP` already exists as an entry
field.

The example below will keep the row keys `locus` and `alleles` as well as add
two new fields: `AC` is making the subfield `AC` into a top level field and
`n_filters` is a new computed field.

    >>> mt_new = mt.select_rows(AC = mt.info.AC,
    ...                         n_filters = hl.len(mt['filters']))

The order of the fields entered as arguments will be maintained in the new
matrix table.

**Drop**

The complement of `select` methods, :meth:`.MatrixTable.drop` can remove any top
level field. An example of removing the `GQ` entry field is:

    >>> mt_new = mt.drop('GQ')

**Explode**

Explode operations can is used to unpack a row or column field that is of type array or
set.

- :meth:`.MatrixTable.explode_rows`
- :meth:`.MatrixTable.explode_cols`

One use case of explode is to duplicate rows:

    >>> mt_new = mt.annotate_rows(replicate_num = [1, 2])
    >>> mt_new = mt_new.explode_rows(mt_new['replicate_num'])
    >>> mt.count_rows()
    346
    >>> mt_new.count_rows()
    692

    >>> mt_new.replicate_num.show()
    +---------------+------------+---------------+
    | locus         | alleles    | replicate_num |
    +---------------+------------+---------------+
    | locus<GRCh37> | array<str> |         int32 |
    +---------------+------------+---------------+
    | 20:10019093   | ["A","G"]  |             1 |
    | 20:10019093   | ["A","G"]  |             2 |
    | 20:10026348   | ["A","G"]  |             1 |
    | 20:10026348   | ["A","G"]  |             2 |
    | 20:10026357   | ["T","C"]  |             1 |
    | 20:10026357   | ["T","C"]  |             2 |
    | 20:10030188   | ["T","A"]  |             1 |
    | 20:10030188   | ["T","A"]  |             2 |
    | 20:10030452   | ["G","A"]  |             1 |
    | 20:10030452   | ["G","A"]  |             2 |
    +---------------+------------+---------------+

Aggregation
===========

:class:`.MatrixTable` has three methods to compute aggregate statistics.

- :meth:`.MatrixTable.aggregate_rows`
- :meth:`.MatrixTable.aggregate_cols`
- :meth:`.MatrixTable.aggregate_entries`

These methods take an aggregated expression and evaluate it, returning
a Python value.

An example of querying entries is to compute the global mean of field `GQ`:

    >>> mt.aggregate_entries(hl.agg.mean(mt.GQ))
    67.73196915777027

It is possible to compute multiple values simultaneously (and encouraged,
because grouping two computations together will run twice as fast!) by
creating a tuple or struct:

    >>> mt.aggregate_entries((agg.stats(mt.DP), agg.stats(mt.GQ)))
    (Struct(mean=41.83915800445897, stdev=41.93057654787303, min=0.0, max=450.0, n=34537, sum=1444998.9999999995),
    Struct(mean=67.73196915777027, stdev=29.80840934057741, min=0.0, max=99.0, n=33720, sum=2283922.0000000135))

See the :ref:`sec-aggregators` page for the complete list of aggregator
functions.

Group-By
========

Matrix tables can be aggregated along the row or column axis to produce a new
matrix table.

- :meth:`.MatrixTable.group_rows_by`
- :meth:`.MatrixTable.group_cols_by`

First let's add a random phenotype as a new column field `case_status` and then
compute statistics about the entry field `GQ` for each grouping of `case_status`.

    >>> mt_ann = mt.annotate_cols(case_status = hl.cond(hl.rand_bool(0.5),
    ...                                                 "CASE",
    ...                                                 "CONTROL"))

Next we group the columns by `case_status` and aggregate:

    >>> mt_grouped = (mt_ann.group_cols_by(mt_ann.case_status)
    ...                 .aggregate(gq_stats = agg.stats(mt_ann.GQ)))

    >>> print(mt_grouped.entry.dtype.pretty())
    struct {
        gq_stats: struct {
            mean: float64,
            stdev: float64,
            min: float64,
            max: float64,
            n: int64,
            sum: float64
        }
    }

    >>> print(mt_grouped.col.dtype)
    struct{status: str}

Joins
=====

Joins on two-dimensional data are significantly more complicated than joins
in one dimension, and Hail does not yet support the full range of
joins on both dimensions of a matrix table.

:class:`.MatrixTable` has methods for concatenating rows or columns:

- :meth:`.MatrixTable.union_cols`
- :meth:`.MatrixTable.union_rows`

:meth:`.MatrixTable.union_cols` joins matrix tables together by performing an
inner join on rows while concatenating columns together (similar to `paste` in
Unix). Likewise, :meth:`.MatrixTable.union_rows` performs an inner join on
columns while concatenating rows together (similar to `cat` in Unix).

In addition, Hail provides support for joining data from multiple sources together
if the keys of each source are compatible (same order and type, but the names do
not need to be identical) using Python's bracket notation ``[]``. The arguments
inside the brackets are the destination key as a single value or a tuple if there
are multiple destination keys.

For example, we can annotate rows with row fields from another matrix table or
table. Let `gnomad_data` be a :class:`.Table` keyed by two row fields with type
``locus`` and ``array<str>``, which matches the row keys of `mt`:

    >>> mt_new = mt.annotate_rows(gnomad_ann = gnomad_data[mt.locus, mt.alleles])

If we only cared about adding one new row field such as `AF` from `gnomad_data`,
we could do the following:

    >>> mt_new = mt.annotate_rows(gnomad_af = gnomad_data[mt.locus, mt.alleles]['AF'])

To add all fields as top-level row fields, the following syntax unpacks the gnomad_data
row as keyword arguments to :meth:`.MatrixTable.annotate_rows`:

    >>> mt_new = mt.annotate_rows(**gnomad_data[mt.locus, mt.alleles])


Interacting with Matrix Tables Locally
======================================

Some useful methods to interact with matrix tables locally are
:meth:`.MatrixTable.describe`, :meth:`.MatrixTable.head`, and
:meth:`.MatrixTable.sample`. `describe` prints out the schema for all row
fields, column fields, entry fields, and global fields as well as the row keys,
column keys, and the partition key. `head` returns a new matrix table with only
the first N rows. `sample` returns a new matrix table where the rows are
randomly sampled with frequency `p`.


To get the dimensions of the matrix table, use :meth:`.MatrixTable.count_rows`
and :meth:`.MatrixTable.count_cols`.


Export
======

To save a matrix table to a file, use the :meth:`.MatrixTable.write`. These
files can be read with :func:`.read_matrix_table`.

--------------
Linear Algebra
--------------
This section coming soon!

--------
Genetics
--------
This section coming soon!

-------------
Common errors
-------------
This section coming soon!

--------------------------
Performance Considerations
--------------------------
This section coming soon!

