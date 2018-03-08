.. _sec-overview:

.. py:currentmodule:: hail

========
Overview
========

-------------
Why use Hail?
-------------

Hail is...

  - what problems can Hail solve
  - what problems can't Hail solve

-----
Types
-----

In Python, ``5`` is of type `int` while ``"hello"`` is of type `string`. Hail has
basic types such as :class:`.TString`, :class:`.TBoolean`, :class:`.TInt32`,
:class:`.TInt64`, :class:`.TFloat32`, :class:`.TFloat64`, as well
as two genetics-specific types: :class:`.TCall` for the genotype call
and :class:`.TLocus` for the genomic locus.

Hail also has container types such as :class:`.TArray`, :class:`.TSet`, and
:class:`.TDict`, which each have an element type as a parameter. For example, a
list of integers in Python would have the type `TArray[TInt32]` where `TInt32` is
the element type. Unlike dictionaries in Python, all keys in a Hail dictionary
must have the same type and the same for values. A dictionary of ``{"a": 1, 2: "b"}`` would be an invalid
dictionary in Hail, but ``{"a": 1, "b": 2}`` would be because the keys are all
strings and the values are all ints. The type of this dictionary would be
`TDict[TString, TInt32]`.

One way to combine types together to form more complicated types is with the
:class:`.TStruct` type. TStruct is a container with an ordered set of fields. A
field has a name and a type. The field names in a TStruct
must be unique. An example valid TStruct is
TStruct["a": TString, "b": TBoolean, "c": TInt32].

More complex types can be created by nesting TStructs. For example, the type
TStruct["a": TStruct["foo": TInt, "baz": TString], "b": TStruct["bar": TArray[TInt32]]] consists
of two fields "a" and "b" each with the type TStruct, but with different fields.
Hail uses TStructs to create complex schemas representing
the structure of data.

-------
Structs
-------

A :class:`.Struct` object corresponds to the type :class:`.TStruct`. It is a
container object for named values, similar to Python's OrderedDict class.
An example is a Struct with two fields `a` and
`b` with the corresponding values 3 and "hello".

    >>> s = Struct(a=3, b="hello")

To access the value ``3``, you can either reference the field `a` as a method or
as an attribute with bracket notation:

    >>> s.a
    3

    >>> s['a']
    3

Be aware that accessing the field as a method will not work if the field name
has periods or special characters in it.

-----------
Expressions
-----------

The Python language allows users to specify their computations using expressions.
For example, a simple expression is ``5 + 6``. This will be evaluated and return
``11``. You can also assign expressions to variables and then add variable expressions
together such as ``x = 5; y = 6; x + y``.

The equivalent of a Python expression in Hail is the :class:`.Expression` class.
Hail expressions are used to specify what computations should be executed on a
dataset such as :class:`.Table`s or :class:`.MatrixTable`s. An expression can represent a single value
such as the int value ``5`` or they can represent the composition of multiple expressions
together and function application. All expressions have a type. For example, an :class:`.StringExpression` would
have the Hail type :class:`.TString`.

The Hail equivalent of the Python example above would be as follows:

    >>> x = hl.capture(5)
    >>> y = hl.capture(6)

The `capture` function is used to convert basic Python objects such as strings, ints,
lists, and dictionaries into their corresponding Hail expression objects.
`capture` also can be applied to Hail objects such as :class:`.Struct`, :class:`.Locus`,
:class:`.Interval`, and :class:`.Call`. The `broadcast` function has the same
functionality as `capture`, but should be used for larger objects such as
a large list or dictionary.

We can print ``x`` in a Python interpreter and see that ``x`` is an :class:`.Int32Expression`.
This makes sense because ``5`` is a Python :obj:`int`.

    >>> x
    <hail.expr.expression.Int32Expression object at 0x10cb5fb50>
      Type: Int32
      Index: None

We can add two :class:`.Int32Expression` objects together just like with Python
:obj:`int`s. Unlike Python, ``x + y`` returns another :class:`.Int32Expression` representing the computation
of ``x + y`` and not an actual value.

    >>> x + y
    <hail.expr.expression.Int32Expression object at 0x10cb5b110>
      Type: Int32
      Index: None

To obtain an actual value, Hail has the `eval_expr` function which will execute the
expression on the input data and return a value. `eval_expr_typed` does the same thing
but also returns the Hail type corresponding to the value.

    >>> hl.eval_expr(x + y)
    11
    >>> hl.eval_expr_typed(x + y)
    (11, TInt32())

We can also add Python :obj:`int` to an :class:`.Int32Expression`.

    >>> x + 3
    <hail.expr.expression.Int32Expression object at 0x10cb218d0>
      Type: Int32
      Index: None

Addition is cumutative, so we can also add an :class:`.Int32Expression` to an
:obj:`int`.

    >>> 3 + x
    <hail.expr.expression.Int32Expression object at 0x10cb4d8d0>
      Type: Int32
      Index: None

Hail has many subclasses of :class:`.Expression` -- one for each Hail type. Each
subclass defines possible methods and operations that can be applied. For example,
if we have a list of :obj:`int` in Python, we can convert this to a Hail :class:`.ArrayInt32Expression`.

    >>> a = hl.capture([1, 2, -3, 0, 5])
    >>> a
    <hail.expr.expression.ArrayInt32Expression object at 0x10cb64390>
      Type: Array[Int32]
      Index: None

:class:`.ArrayInt32Expression` has many methods that are documented `here`. We
can obtain the ith element using Python's index notation with ``a[i]``. The resultant
expression will be a :class:`.Int32Expression` because each element of the array is
an integer.

    >>> a[1]
    <hail.expr.expression.Int32Expression object at 0x10bbdd450>
      Type: Int32
      Index: None

Likewise, if we `sort` the array, the resultant expression is a :class:`.ArrayInt32Expression`.

    >>> a.sort()
    <hail.expr.expression.ArrayInt32Expression object at 0x10bbddd50>
      Type: Array[Int32]
      Index: None


Boolean Logic
=============

Unlike Python, Hail :class:`.BooleanExpression`s cannot be combined with ``and``, ``or``,
and ``not``. The equivalents are ``&``, ``|``, and ``~``.

    >>> s1 = hl.capture(x == 3)
    >>> s2 = hl.capture(x != 4)

    >>> s1 & s2 # s1 and s2
    >>> s1 | s2 # s1 or s2
    >>> ~s1 # not s1

In addition, parantheses are required if the boolean expression is not a single variable
because the precedence of the ``&` and ``|`` operators are lower than ``and`` and ``or``
in Python.

    >>> (x == 3) & (x != 4)

Conditionals
============

A conditional expression has three components: the condition to evaluate, the consequent
value to return if the condition is ``True``, and the alternative to return if the
condition is ``False``. The Python equivalent of this is `if-else` statements. For example,
a trivial example is

.. code-block:: python

    if (x > 0):
        return 1
    else:
        return 0

where the condition is ``x > 0``, the consequent is ``1``, and the alternative is ``0``.

The Hail equivalent of this is with the `cond` function.

    >>> hl.cond(x > 0, 1, 0)
    <hail.expr.expression.Int32Expression object at 0x10cb630d0>
      Type: Int32
      Index: None


The condition statement must be a :obj:`boolean` or a :class:`.BooleanExpression`.
The type of evaluating this function is an :class:`.Int32Expression` because both the
consequent and alternative are :obj:`int`. **The types of the consequent and alternative
must always be the same.** This conditional expression can be used in composing
larger expressions where :class:`.Int32Expression`s can be used. For example, we
can add the result of the conditional statement to ``a`` which was defined above.

    >>> a + hl.cond(x > 0, 1, 0)
    <hail.expr.expression.ArrayInt32Expression object at 0x10cb668d0>
      Type: Array[Int32]
      Index: None

More complicated conditional statements can be constructed with `case`. For example,
we might want to emit ``1`` if ``x < -1``, ``2`` if ``-1 <= x <= 2`` and ``3`` if ``x > 2``.

    >>> hl.case()
    ...   .when(x < -1, 1)
    ...   .when(x >= -1 & x <= 2, 2)
    ...   .when(x > 2, 3)

Default values can also be specified if no match is made with ``.default(...)``.

    >>> hl.case()
    ...   .when(x >= -1 & x <= 2, 1)
    ...   .when(x > 2 & x < 5, 2)
    ...   .default(0)


Lastly, Hail has a `switch` function to build a conditional tree based on the
value of an expression. In the example below, `csq` is a :class:`.StringExpression`
representing the functional consequence of a mutation. If `csq` does not match
one of the cases specified by `when`, it is set to missing with `or_missing`. Other
switch statements are documented in the :class:`.SwitchBuilder` class.

.. code-block:: python

    is_damaging = (hl.switch(csq)
                     .when("synonymous", False)
                     .when("intron", False)
                     .when("nonsense", True)
                     .when("indel", True)
                     .or_missing())


Missingness
===========

An expression representing a missing value of a given type can be generated with
the `null` function which takes the type as its single argument. An example of
generating a :class:`.Float64Expression` that is missing is

    >>> hl.null(TFloat64())

These can be used with conditional statements to set values to missing if they
don't satisfy a condition:

    >>> hl.cond(x > 2.0, x, hl.null(TFloat64()))

The result of method calls on a missing value is ``None``. For example, if
we define ``cnull`` to be a missing value with type :class:`.TCall`, calling
the method `is_het` will return ``None`` and not ``False``.

    >>> cnull = hl.null(TCall())
    >>> cnull.is_het()
    None


Binding Variables
=================

Hail inlines function calls each time an expression appears. This can result
in unexpected behavior when random values are used. For example, let ``x`` be
a random number generated with the function `rand_unif`.

    >>> x = hl.rand_unif(0, 1)

If we create a list with x repeated 3 times, we'd expect to get an array with identical
values. However, instead we see a list of 3 random numbers.

    >>> hl.eval_expr([x, x, x])
    [0.8846327207915881, 0.14415148553468504, 0.8202677741734825]

To solve this problem, we can use the `bind` function to bind an expression to a
value before applying it in a function.

    >>> expr = hl.bind(hl.rand_unif(0, 1), lambda x: [x, x, x])
    >>> hl.eval_expr(expr)
    [0.5562065047992025, 0.5562065047992025, 0.5562065047992025]


Functions
=========

In addition to the methods exposed on each :class:`.Expression`, Hail also has
numerous functions that can be applied to expressions, which also return an expression.
We have already seen examples of the functions `capture`, `cond`, `switch`, `case`, `bind`,
`rand_unif`, and `null`. Some examples of other commonly used functions are

**Conditionals**

- `cond`
- `switch`
- `case`
- `or_else`
- `or_missing`

**Missingness**

- `is_defined`
- `is_missing`
- `is_nan`

**Mathematical Operations**

- `exp`
- `log`
- `log10`

**Manipulating Structs**

- `select`
- `merge`
- `drop`

**Constructors**

Construct a missing value of a given type:

- `null`

Construct expressions from input arguments:

- `Dict`
- `locus`
- `interval`
- `call`

Parse strings to construct expressions:

- `parse_variant`
- `parse_locus`
- `parse_interval`
- `parse_call`

**Random Number Generators**

- `rand_bool`
- `rand_norm`
- `rand_pois`
- `rand_unif`

**Statistical Tests**

- `chisq`
- `fisher_exact_test`
- `hardy_weinberg_p`

See the full `API` for a list of all functions and their documentation.


-----
Table
-----

A :class:`~hail.Table` is the Hail equivalent of a SQL table, a Pandas Dataframe, an R Dataframe,
a dyplr Tibble, or a Spark Dataframe. It consists of rows of data conforming to
a given schema where each column (row field) in the dataset is of a specific type.

An example of a table is below:

+---------+---------+-------+
| Sample  | Status  | qPhen |
+---------+---------+-------+
| String  | String  | Int32 |
+---------+---------+-------+
| HG00096 | CASE    | 27704 |
| HG00097 | CASE    | 16636 |
| HG00099 | CASE    |  7256 |
| HG00100 | CASE    | 28574 |
| HG00101 | CASE    | 12088 |
| HG00102 | CASE    | 19740 |
| HG00103 | CASE    |  1861 |
| HG00105 | CASE    | 22278 |
| HG00106 | CASE    | 26484 |
| HG00107 | CASE    | 29726 |
+---------+---------+-------+

It's schema is

.. code-block::text

    TStruct(Sample=TString, Status=TString, qPhen = TInt32)


Global Fields
=============

In addition to row fields, Hail tables also have global fields. You can think of globals as
extra fields in the table whose values are identical for every row. For example,
the same table above with the global field ``X = 5`` can be thought of as

+---------+---------+-------+-------+
| Sample  | Status  | qPhen |     X |
+---------+---------+-------+-------+
| String  | String  | Int32 | Int32 |
+---------+---------+-------+-------+
| HG00096 | CASE    | 27704 |     5 |
| HG00097 | CASE    | 16636 |     5 |
| HG00099 | CASE    |  7256 |     5 |
| HG00100 | CASE    | 28574 |     5 |
| HG00101 | CASE    | 12088 |     5 |
| HG00102 | CASE    | 19740 |     5 |
| HG00103 | CASE    |  1861 |     5 |
| HG00105 | CASE    | 22278 |     5 |
| HG00106 | CASE    | 26484 |     5 |
| HG00107 | CASE    | 29726 |     5 |
+---------+---------+-------+-------+

but the value ``5`` is only stored once for the entire dataset and NOT once per
row of the table. The output of `describe` lists what all of the row
fields and global fields are.

.. code-block::text

    Global fields:
        'X': Int32

    Row fields:
        'Sample': String
        'Status': String
        'qPhen': Int32


Keys
====

Row fields can be specified to be the keys of the table with the method `key_by`.
Keys are important for joining tables together (discussed below).

Referencing Fields
==================

Each :class:`.Table` object has all of its row fields and global fields as
attributes in its namespace. This means that the row field `Sample` can be accessed
from table `t` with ``t.Sample`` or ``t['Sample']``. If `t` also had a global field `X`,
then it could be accessed by either ``t.X`` or ``t['X']``. Both row fields and global
fields are top level fields. Be aware that accessing a field with the `dot` notation will not work
if the field name has special characters or periods in it. The Python type of each
attribute is an :class:`.Expression` that also contains context about its type and source,
in this case a row field of table `t`.

    >>> t

.. code-block:: text

    is.hail.table.Table@42dd544f

    >>> t.Sample

.. code-block:: text

    <hail.expr.expression.StringExpression object at 0x10b498290>
      Type: String
      Index:
        row of is.hail.table.Table@42dd544f

Import
======

Hail has functions to create tables from a variety of data sources.
The most common use case is to load data from a TSV or CSV file, which can be
done with the `import_table` function.

.. doctest::

    t = methods.import_table("data/kt_example1.tsv", impute=True)

A table can also be created from Python
objects with `parallelize`. For example, a table with only the first two rows
above could be created from Python objects.

.. doctest::

    rows = [{"Sample": "HG00096", "Status": "CASE", "qPhen": 27704},
            {"Sample": "HG00097", "Status": "CASE", "qPhen": 16636}]

    schema = TStruct(["Sample", "Status", "qPhen"], [TString(), TString(), TInt32()])

    t_new = Table.parallelize(rows, schema)

Examples of genetics-specific import methods are
`import_interval_list`, `import_fam`, and `import_bed`. Many Hail methods also
return tables.

Common Operations
=================

The main operations on a table are `select` and `drop` to add or remove row fields,
`filter` to either keep or remove rows based on a condition, and `annotate` to add
new row fields or update the values of existing row fields. For example, extending
the example table above, we can filter the table to only contain rows where
``qPhen < 15000``, add a new row field `SampleInt` which is the integer component of the row
field `Sample`, add a new global field `foo`, and select only the row fields `SampleInt` and
`qPhen` as well as define a new row field `bar` which is the product of `qPhen` and `SampleInt`.
Lastly, we can use `show` to view the first 10 rows of the new table.


# FIXME: add transmute and explode

.. doctest::

    t_new = t.filter(t['qPhen'] < 15000)
    t_new = t_new.annotate(SampleInt = t.Sample.replace("HG", "").to_int32())
    t_new = t_new.annotate_globals(foo = 131)
    t_new = t_new.select(t['SampleInt'], t['qPhen'], bar = t['qPhen'] * t['SampleInt'])
    t_new.show()

The final output is

.. code-block:: text

    +-----------+-------+---------+
    | SampleInt | qPhen |     bar |
    +-----------+-------+---------+
    |     Int32 | Int32 |   Int32 |
    +-----------+-------+---------+
    |        99 |  7256 |  718344 |
    |       101 | 12088 | 1220888 |
    |       103 |  1861 |  191683 |
    |       113 |  8845 |  999485 |
    |       116 | 12742 | 1478072 |
    |       121 |  4832 |  584672 |
    |       124 |  2691 |  333684 |
    |       125 | 14466 | 1808250 |
    |       127 | 10224 | 1298448 |
    |       128 |  2807 |  359296 |
    +-----------+-------+---------+

with the following schema:

.. code-block:: text

    Global fields:
        'foo': Int32

    Row fields:
        'SampleInt': Int32
        'qPhen': Int32
        'bar': Int32

Aggregations
============

A commonly used operation is to compute an aggregate statistic over the rows of
the dataset. Hail provides an `aggregate` method along with many
`aggregator functions` to return the result of a query.
For example, to compute the fraction of rows with ``Status == "CASE"`` and the
mean value for `qPhen`, we can run the following command:

.. doctest::

    t.aggregate(frac_case = agg.fraction(t.Status == "CASE"),
                mean_qPhen = agg.mean(t.qPhen))

.. code-block:: text

    Struct(frac_case=0.41, mean_qPhen=17594.625)

We also might want to compute the mean value of `qPhen` for each unique value of `Status`.
To do this, we need to first create a :class:`.GroupedTable` using the `group_by` method. This
will expose the method `aggregate` which can be used to compute new row fields
over the grouped-by rows.

.. doctest::

    t_agg = (t.group_by('Status')
              .aggregate(mean = agg.mean(t['qPhen'])))
    t_agg.show()


.. code-block:: text

    +--------+-------------+
    | Status |        mean |
    +--------+-------------+
    | String |     Float64 |
    +--------+-------------+
    | CASE   | 1.83183e+04 |
    | CTRL   | 1.70995e+04 |
    +--------+-------------+

Note that the result of `t.group_by(...).aggregate(...)` is a new :class:`.Table`
while the result of `t.aggregate(...)` is either a single value or a :class:`.Struct`.

Joins
=====

To join the row fields of two tables together, Hail provides a `join` method with
options for how to join the rows together (left, right, inner, outer). The tables are
joined by the row fields designated as keys. The number of keys and their types
must be identical between the two tables. However, the names of the keys do not
need to be identical. Use the `key` attribute to view the current
table row keys and the `key_by` method to change the table keys. If top level
row field names overlap between the two tables, the second table's field names
will be appended with a unique identifier "_N".

.. doctest::

    t1 = t.key_by('Sample')
    t2 = (functions.import_table("data/kt_example2.tsv", impute=True)
                   .key_by('Sample'))

    t_join = t1.join(t2)
    t_join.show()

.. code-block:: text

    +---------+--------+-------+-------------+--------+
    | Sample  | Status | qPhen |      qPhen2 | qPhen3 |
    +---------+--------+-------+-------------+--------+
    | String  | String | Int32 |     Float64 |  Int32 |
    +---------+--------+-------+-------------+--------+
    | HG00097 | CASE   | 16636 | 3.32720e+03 |  16626 |
    | HG00128 | CASE   |  2807 | 5.61400e+02 |   2797 |
    | HG00111 | CASE   | 30065 | 6.01300e+03 |  30055 |
    | HG00122 | CASE   |    NA | 0.00000e+00 |    -10 |
    | HG00107 | CASE   | 29726 | 5.94520e+03 |  29716 |
    | HG00136 | CASE   | 12348 | 2.46960e+03 |  12338 |
    | HG00113 | CASE   |  8845 | 1.76900e+03 |   8835 |
    | HG00103 | CASE   |  1861 | 3.72200e+02 |   1851 |
    | HG00120 | CASE   | 19599 | 3.91980e+03 |  19589 |
    | HG00114 | CASE   | 31255 | 6.25100e+03 |  31245 |
    +---------+--------+-------+-------------+--------+

In addition to using the `join` method, Hail provides an additional join syntax
using Python's bracket notation. For example, below we add the column `qPhen2` from table
2 to table 1 by joining on the row field `Sample`:

.. doctest::

    t1 = t1.annotate(qPhen2 = t2[t.Sample].qPhen2)
    t1.show()

.. code-block:: text

    +---------+--------+-------+-------------+
    | Sample  | Status | qPhen |      qPhen2 |
    +---------+--------+-------+-------------+
    | String  | String | Int32 |     Float64 |
    +---------+--------+-------+-------------+
    | HG00180 | CTRL   | 27337 |          NA |
    | HG00160 | CTRL   | 29590 |          NA |
    | HG00141 | CTRL   | 25689 |          NA |
    | HG00097 | CASE   | 16636 | 3.32720e+03 |
    | HG00145 | CTRL   |  7641 |          NA |
    | HG00158 | CTRL   | 12369 |          NA |
    | HG00243 | CTRL   | 18065 |          NA |
    | HG00128 | CASE   |  2807 | 5.61400e+02 |
    | HG00234 | CTRL   | 18268 |          NA |
    | HG00111 | CASE   | 30065 | 6.01300e+03 |
    +---------+--------+-------+-------------+

The general format of the key word argument to `annotate` is

.. code-block:: text

    new_field_name = <other table> [<this table's keys >].<field to insert>

Note that both `t1` and `t2` have been keyed by the column `Sample` with the same
type TString. This syntax for joining can be extended to add new row fields
from many tables simultaneously.

If both `t1` and `t2` have the same schema, but different rows, the rows
of the two tables can be combined with `union`.


Interacting with Tables Locally
===============================

Hail has many useful methods for interacting with tables locally such as in an
iPython notebook. Use the `show` method to see the first 10 rows of a table.

`take` will collect the first `n` rows of a table into a local Python list

.. doctest::

    x = t.take(3)
    x

.. code-block:: text

    [Struct(Sample=HG00096, Status=CASE, qPhen=27704),
     Struct(Sample=HG00097, Status=CASE, qPhen=16636),
     Struct(Sample=HG00099, Status=CASE, qPhen=7256)]

Note that each element of the list is a Struct whose elements can be accessed using
Python's get attribute notation

.. doctest::

    x[0].qPhen

.. code-block:: text

    27704

When testing pipelines, it is helpful to subset the dataset to the first `n` rows
with the `head` method. The result of `head` is a new Table rather than a local
list of Struct elements as with `take` or a printed representation with `show`.
`sample` will return a randomly sampled fraction of the dataset. This is useful
for having a smaller, but random subset of the data.

`describe` is a useful method for showing all of the fields of the table and their
types. The complete table schemas can be accessed with `schema` and `global_schema`.
The row fields that are keys can be accessed with `key`. Lastly, the `num_columns`
attribute returns the number of row fields and the `count` method returns the
number of rows in the table.

Export
======

Hail provides multiple functions to export data to other formats. Tables
can be exported to TSV files with the `export` method or written to disk in Hail's
on-disk format with `write` and read back in with `read_table`. Tables can also be exported to Pandas tables with
`to_pandas` or to Spark Dataframes with `to_spark`. Lastly, tables can be converted
to a Hail :class:`.MatrixTable` with `to_matrix_table`, which is the subject of the next
section.

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

Unlike a :class:`.Table` which has two schemas, a matrix table has four schemas
that define the structure of the dataset. The rows table has a `row_schema`, the
columns table has a `col_schema`, each entry in the matrix follows the schema
defined by `entry_schema`, and the global fields have a `global_schema`.

In addition, there are different operations on the matrix for each dimension
of the data. For example, instead of just `filter` for tables, matrix tables
have `filter_rows`, `filter_cols`, and `filter_entries`.

One equivalent way of representing this data is in one combined table encompassing
all row, column, and global fields with one row in the table per entry in the matrix (coordinate form).
Hail does not store the data in this format as it is inefficient when computing
results and the on-disk representation would be massive as constant values are
repeated per entry in the dataset.

Keys
====

Analogous to tables, matrix tables also have keys. However, instead of one key, matrix
tables have two keys: one for the rows table and the other for the columns table.  Entries
are indexed by both the row keys and column keys. The keys
can be accessed with the attributes `row_key` and `col_key` and set with the methods
`key_rows_by` and `key_cols_by`. Keys are used for joining tables together (discussed below).

In addition, each matrix table has a `partition_key`. This key is used for specifying
the ordering of the matrix table along the row dimension, which is important for
performance.


Referencing Fields
==================

All fields (row, column, global, entry)
are top-level and exposed as attributes on the :class:`.MatrixTable` object.
For example, if the matrix table `mt` had a row field `locus`, this field
could be referenced with either ``mt.locus`` or ``mt['locus']``. The former
access pattern does not work with field names with special characters or periods
in it.

The result of referencing a field from a matrix table is an :class:`Expression` which knows its type
and knows its source as well as whether it is a row field, column field, entry field, or global field.
Hail uses this context to know which operations are allowed for a given expression.

When evaluated in a Python interpreter, we can see ``mt.locus`` is a :class:`.LocusExpression`
with type `Locus(GRCh37)` and it is a row field of the MatrixTable `mt`.

    >>> mt

.. code-block:: text

    <hail.matrixtable.MatrixTable at 0x10a6a3e50>

    >>> mt.locus

.. code-block:: text

    <hail.expr.expression.LocusExpression object at 0x10b17f790>
      Type: Locus(GRCh37)
      Index:
        row of <hail.matrixtable.MatrixTable object at 0x10a6a3e50>

Likewise, ``mt.DP`` would be an :class:`.Int32Expression` with type `Int32` and
is an entry field of `mt`. It is indexed by both rows and columns as denoted
by its indices when printing the expression.

    >>> mt.DP

.. code-block:: text

    <hail.expr.expression.Int32Expression object at 0x10b2cec10>
      Type: Int32
      Indices:
        column of <hail.matrixtable.MatrixTable object at 0x10a6a3e50>
        row of <hail.matrixtable.MatrixTable object at 0x10a6a3e50>


Import
======

Hail provides four functions to import genetic datasets as matrix tables from a
variety of file formats: `import_vcf`, `import_plink`, `import_bgen`, and
`import_gen`. We will be adding a function to import a matrix table from a TSV
file in the future.

An example of importing data from a VCF file to a matrix table follows:

    >>> mt = methods.import_vcf('data/example2.vcf.bgz')

The `describe` method shows the schemas for the global fields, column fields,
row fields, entry fields, as well as the column key(s), the row key(s), and the
partition key.

    >>> mt.describe()
    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Column fields:
        's': String
    ----------------------------------------
    Row fields:
        'locus': Locus(GRCh37)
        'alleles': Array[String]
        'rsid': String
        'qual': Float64
        'filters': Set[String]
        'info': Struct {
            NEGATIVE_TRAIN_SITE: Boolean,
            HWP: Float64,
            AC: Array[Int32],
            culprit: String,
            .
            .
            .
        }
    ----------------------------------------
    Entry fields:
        'GT': Call
        'AD': Array[+Int32]
        'DP': Int32
        'GQ': Int32
        'PL': Array[+Int32]
    ----------------------------------------
    Column key:
        's': String
    Row key:
        'locus': Locus(GRCh37)
        'alleles': Array[String]
    Partition key:
        'locus': Locus(GRCh37)
    ----------------------------------------


Common Operations
=================

Like tables, Hail provides a number of useful methods for manipulating data in a
matrix table.

**Filter**

Hail has three methods to filter a matrix table based on a condition:

- `filter_rows`
- `filter_cols`
- `filter_entries`

Filter methods take a `boolean expression` as its argument. The simplest boolean
expression is ``False``, which will remove all rows, or ``True``, which will
keep all rows.

Just filtering out all rows, columns, or entries isn't particularly useful. Often,
we want to filter parts of a dataset based on a condition the elements satisfy.
A commonly used application in genetics is to only keep rows where the number of
alleles is two (biallelic). This can be expressed as follows:

    >>> mt_biallelic = mt.filter_rows(mt['alleles'].length() == 2)

So what is going on here? The reference to the row field `alleles` returns an
expression of type `Array[String] :class:`.ArrayStringExpression`. Array expressions
have multiple methods on them including `length` which returns the number of elements
in the array. This expression representing the length of the row field `alleles`
is compared to the number 2 with the `==` comparison operator to return a boolean expression.
Note that the expression `mt['alleles'].length() == 2` is not actually a value
in Python. Rather it represents a recipe for computation that is then used by
Hail to evaluate each row in the matrix table for whether the condition is met.

More complicated expressions can be written with a combination of Hail's functions.
An example of filtering columns where the fraction of non-missing elements for
the entry field `GT` is greater than 0.95 utilizes the function `is_defined` and
the aggregator function `fraction`.

    >>> mt_new = mt.filter_cols(agg.fraction(functions.is_defined(mt.GT)) >= 0.95)
    >>> mt.count_cols()
    100
    >>> mt_new.count_cols()
    91

In this case, the expression ``mt.GT`` is an aggregable because the function context
is an operation on columns (`filter_cols`). This means for each column in the
matrix table, we have N `GT` entries where N is the number of rows in the dataset.
Aggregables cannot be realized as an actual value, so we must use an aggregator
function to reduce the aggregable to an actual value.

In the example above, `functions.is_defined` is applied to each element of the aggregable ``mt.GT``
to transform it from an Aggregable[Call] to an Aggregable[Boolean] where ``True``
means the value `GT` was defined or ``False`` for missing. `agg.fraction` requires
an Aggregable[Boolean] for its input, which it then reduces to a single value by computing the
number of ``True`` values divided by `N`, the length of the aggregable. The result
of `fraction` is a single value per column, which can then be compared
to the value `0.95` with the `>=` comparison operator.

Hail also provides two methods to filter columns or rows based on an input list
of values. This is useful if you have a known subset of the dataset you want to
subset to.

- `filter_rows_list`
- `filter_cols_list`


**Annotate**

Hail provides four methods to add fields to a matrix table or update existing fields:

- `annotate_rows`
- `annotate_cols`
- `annotate_entries`
- `annotate_globals`

Annotate methods take key-word arguments where the key is the name of the new
field to add and the value is an expression specifying what should be added.

The simplest example is adding a new global field `foo` that just contains the constant
5.

    >>> mt_new = mt.annotate_globals(foo = 5)
    >>> mt.global_schema.pretty()
    Struct {
        foo: Int32
    }

Another example is adding a new row field `call_rate` which computes the fraction
of non-missing entries `GT` per row. This is similar to the filter example described
above, except the result of `agg.fraction(functions.is_defined(mt.GT))` is stored
as a new row field in the matrix table and the operation is performed over rows
rather than columns.

    >>> mt_new = mt.annotate_rows(call_rate = agg.fraction(functions.is_defined(mt.GT)))

Annotate methods are also useful for updating values. For example, to update the
GT entry field to be missing if `GQ` is less than 20, we can do the following:

    >>> mt_new = mt.annotate_entries(GT = functions.cond(mt.GQ < 20,
    ...                                                  functions.null(TCall()),
    ...                                                  mt.GT))

**Select**

Select is used to create a new schema for a dimension of the matrix table. For
example, following the matrix table schemas from importing a VCF file (shown above),
to create a hard calls dataset where each entry only contains the `GT` field
one can do the following:

    >>> mt_new = mt.select_entries('GT')
    >>> mt_new.entry_schema.pretty()
    Struct {
        GT: Call
    }

Hail has four select methods that correspond to modifying the schema of the row
fields, the column fields, the entry fields, and the global fields.

- `select_rows`
- `select_cols`
- `select_entries`
- `select_globals`

Each method can take either strings referring to top-level fields, an attribute
reference (useful for accessing nested fields), as well as key word arguments
``KEY=VALUE`` to compute new fields. The Python unpack operator ``**`` can be used
to specify that all fields of a Struct should become top level fields. However,
be aware that all field names must be unique across rows, columns, entries, and globals.
So in this example, `**mt['info']` would fail because `DP` already exists as an entry field.

The example below will keep
the row fields `locus` and `alleles` as well as add two new fields: `AC` is making
the subfield `AC` into a top level field and `n_filters` is a new computed field.

.. doctest::

    mt_new = mt.select_rows('locus',
                            'alleles',
                            AC = mt['info']['AC'],
                            n_filters = mt['filters'].length())

    mt_new.row_schema.pretty()

.. code-block:: text

    Struct {
        locus: Locus(GRCh37),
        alleles: Array[String],
        AC: Array[Int32],
        n_filters: Int32
    }

The order of the fields entered as arguments will be maintained in the new
matrix table.

**Drop**

Analogous to `select`, `drop` will remove any top level field. An example of
removing the `GQ` entry field is

    >>> mt_new = mt.drop('GQ')
    >>> mt_new.entry_schema.pretty()
    Struct {
        GT: Call,
        AD: Array[+Int32],
        DP: Int32,
        PL: Array[+Int32]
    }

Hail also has two methods to drop all rows or all columns from the matrix table:
`drop_rows` and `drop_cols`.

**Explode**

Explode is used to unpack a row or column field that is of type array or
set.

- `explode_rows`
- `explode_cols`

One use case of explode is to duplicate rows:

    >>> mt_new = mt.annotate_rows(replicate_num = [1, 2])
    >>> mt_new = mt_new.explode_rows(mt_new['replicate_num'])
    >>> mt.count_rows()
    7
    >>> mt_new.count_rows()
    14

    >>> mt_new.rows_table().select('locus', 'alleles', 'replicate_num').show()

.. code-block:: text

    +---------------+-----------------+---------------+
    | locus         | alleles         | replicate_num |
    +---------------+-----------------+---------------+
    | Locus(GRCh37) | Array[String]   |         Int32 |
    +---------------+-----------------+---------------+
    | 20:12990057   | ["T","A"]       |             1 |
    | 20:12990057   | ["T","A"]       |             2 |
    | 20:13090733   | ["A","AT"]      |             1 |
    | 20:13090733   | ["A","AT"]      |             2 |
    | 20:13695824   | ["CAA","C"]     |             1 |
    | 20:13695824   | ["CAA","C"]     |             2 |
    | 20:13839933   | ["T","C"]       |             1 |
    | 20:13839933   | ["T","C"]       |             2 |
    | 20:15948326   | ["GAAAAAA","G"] |             1 |
    | 20:15948326   | ["GAAAAAA","G"] |             2 |
    +---------------+-----------------+---------------+

Aggregations
============

Like :class:`Table`, Hail provides three methods to compute aggregate statistics.

- `aggregate_rows`
- `aggregate_cols`
- `aggregate_entries`

These methods take key word arguments where the key is the name of the value to
compute and the value is the expression for what to compute. The return value
of aggregate is either a single value or a :class:`.Struct` depending
on the number of values to compute.

An example of querying entries is to compute the fraction of values where `GT`
is defined across the entire dataset (call rate):

    >>> mt.aggregate_entries(call_rate = agg.fraction(functions.is_defined(mt.GT)))
    0.9871428571428571

We can also compute multiple global statistics simulatenously by supplying multiple
key-word arguments:

    >>> result = mt.aggregate_entries(dp_stats = agg.stats(mt.DP),
    ...                               gq_stats = agg.stats(mt.GQ))

    >>> result.dp_stats
    Struct(min=5.0, max=161.0, sum=22587.0, stdev=17.7420068551, nNotMissing=699, mean=32.313304721)

Hail provides many aggregator functions which are documented `here`.

Group-By
========

Hail provides two methods to group data by either a row field or a column field
and compute an aggregated statistic for each grouping which then becomes the
entry fields of a new :class:`.MatrixTable`.

- `group_rows_by`
- `group_cols_by`

First let's add a random phenotype
as a new column field `Status` and then compute statistics about the entry field `GQ`
for each grouping of `Status`.

    >>> mt_ann = mt.annotate_cols(Status = functions.cond(functions.rand_bool(0.5),
    ...                                                   "CASE",
    ...                                                   "CONTROL"))

Next we group the columns by `Status` and specify the new entry field will be
stats on `GQ` that are computed for each grouping of `Status`:

    >>> mt_grouped = (mt_ann.group_cols_by(mt_ann['Status'])
    ...                 .aggregate(gq_stats = agg.stats(mt_ann.GQ)))

    >>> mt_grouped.entry_schema().pretty()
    Struct {
        gq_stats: Struct {
            mean: Float64,
            stdev: Float64,
            min: Float64,
            max: Float64,
            nNotMissing: Int64,
            sum: Float64
        }
    }

    >>> mt_grouped.col_schema().pretty()
    Struct {
        Status: String
    }

Joins
=====

Hail provides two methods to join :class:`.MatrixTable`s together:

- `union_join_cols`
- `union_join_rows`

`union_join_cols` joins matrix tables together by performing an inner join
on rows while concatenating columns together (similar to `paste` in Unix).
Likewise, `union_join_rows` performs an inner join on columns while concatenating
rows together (similar to `cat` in Unix).

In addition, Hail provides support for joining data from multiple sources together
if the keys of each source are compatible (same order and type, but the names do
not need to be identical) using Python's bracket notation ``[]``. The arguments
inside the brackets are the destination key as a single value or a tuple if there
are multiple destination keys.

For example, we can annotate rows with row fields from another matrix table or table.
Let `gnomad_data` be a :class:`.Table` keyed by two row fields with type TLocus and
TArray(TString), which matches the row keys of `mt`:

    >>> mt_new = mt.annotate_rows(gnomad_ann = gnomad_data[(mt.locus, mt.alleles)])

This command will add a new field `gnomad_ann` which is the result of joining
between the `locus` and `alleles` row fields of `gnomad_data` and the `locus`
row field of the matrix table `mt`. For every row in which the keys intersect,
a new row field `gnomad_ann` which is of type TStruct with fields equal to the
row fields of `gnomad_data`. For rows where the keys do not intersect, a Struct is
added with field names equal to the row fields of `gnomad_data`, but whose values
are all set to missing.

If we only cared about adding one new row field such as `AF` from `gnomad_data`,
we could do the following:

    >>> mt_new = mt.annotate_rows(gnomad_af = gnomad_data[(mt.locus, mt.alleles)]['AF'])

Analogously, we can add new column fields from a table. In this example, `pheno_data`
is a table with one key of type TString, which matches the column key of the matrix
table `mt`. A new column field `phenotypes` will be added which is a Struct containing
the row fields of the table `pheno_data`.

    >>> mt_new = mt.annotate_cols(phenotypes = pheno_data[mt.s])

This implicit join syntax can also be used to add fields from one matrix table
to another matrix table.

    >>> mt_new = mt.annotate_cols(phenotypes = mt1[mt.s]['SampleID2'])


Interacting with MatrixTables Locally
=====================================

Some useful methods to interact with matrix tables locally are `describe`,
`head`, and `sample`. `describe` prints out the schema for all row fields, column
fields, entry fields, and global fields as well as the row keys, column keys, and
the partition key. `head` returns a new matrix table with only the first N
rows. `sample` returns a new matrix table where the rows are randomly sampled
with frequency `p`.

To get the dimensions of the matrix table, use `count_rows` and `count_cols`.

Export
======

To save a matrix table to a file, use the `write` command and subsequently `read_matrix_table`
to read the file again.

In addition, Hail provides three methods to convert matrix tables to tables, which can then
be printed with :meth:`~hail.Table.show` or exported to a file:

- `rows_table`
- `cols_table`
- `entries_table`

The rows table contains a :class:`.Table` with all row fields and the columns table
contains a :class:`.Table` with all column fields. Likewise, the entries table is
a :class:`.Table` that contains a row for every element in the matrix along with the row
and column fields. The entries table is extremely big because it contains
a row for every element in the matrix as well as the corresponding row and column fields.
The entries table should never be saved to disk with `write`.

    >>> mt.rows_table().select('locus', 'alleles', 'rsid').show()
    >>> mt.cols_table().select('s').show()

A common idiom is to compute ... 

Methods
-------



--------------------------
Other Hail Data Structures
--------------------------
- linear algebra
- block matrix


---------------------
Where's the Genetics?
---------------------
  - genetics specific
    - import vcf, gen, bgen
    - export vcf, gen, etc.
    - call stats, inbreeding, hwe aggregators
    - alternate alleles
- tdt
- genetics objects
- genetics types

---------------------
Python Considerations
---------------------
  - chaining methods together => not referring to correct dataset in future operations
  - varargs vs. keyword args
  - how to access attributes (square brackets vs. method accessor)
  - how to work with fields with special chars or periods in name **{'a.b': 5}


--------------------------
Performance Considerations
--------------------------

-----
Other
-----
  - hadoop_open, etc.

