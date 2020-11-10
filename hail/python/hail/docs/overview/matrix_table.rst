--------------------
MatrixTable Overview
--------------------

A :class:`.MatrixTable` is a distributed two-dimensional extension of a
:class:`.Table`.

Unlike a table, which has two field groups (row fields and global
fields), a matrix table consists of four components:

1. a two-dimensional matrix of **entry fields** where each entry is indexed by
   row key(s) and column key(s)
2. a corresponding rows table that stores all of the **row fields** that are
   constant for every column in the dataset
3. a corresponding columns table that stores all of the **column fields** that
   are constant for
   every row in the dataset
4. a set of **global fields** that are constant for every entry in the dataset

There are different operations on the matrix for each field group.
For instance, :class:`.Table` has :meth:`.Table.select` and
:meth:`.Table.select_globals`, while :class:`.MatrixTable` has
:meth:`.MatrixTable.select_rows`, :meth:`.MatrixTable.select_cols`,
:meth:`.MatrixTable.select_entries`, and :meth:`.MatrixTable.select_globals`.

It is possible to represent matrix data by coordinate in a table , storing one
record per entry of the matrix. However, the :class:`.MatrixTable` represents
this data far more efficiently and exposes natural interfaces for computing on
it.

The :meth:`.MatrixTable.rows` and :meth:`.MatrixTable.cols` methods return the
row and column fields as separate tables. The :meth:`.MatrixTable.entries`
method returns the matrix as a table in coordinate form -- use this object with
caution, because this representation is costly to compute and is significantly
larger in memory

Keys
====

Matrix tables have keys just as tables do. However, instead of one key, matrix
tables have two keys: a row key and a column key. Row fields are indexed by the
row key, column fields are indexed by the column key, and entry fields are
indexed by the row key and the column key. The key structs can be accessed with
:attr:`.MatrixTable.row_key` and :attr:`.MatrixTable.col_key`. It is possible to
change the keys with :meth:`.MatrixTable.key_rows_by` and
:meth:`.MatrixTable.key_cols_by`.

Due to the data representation of a matrix table, changing a row key is often an
expensive operation.

Referencing Fields
==================

All fields (row, column, global, entry) are top-level and exposed as attributes
on the :class:`.MatrixTable` object. For example, if the matrix table `mt` had a
row field `locus`, this field could be referenced with either ``mt.locus`` or
``mt['locus']``. The former access pattern does not work with field names with
spaces or punctuation.

The result of referencing a field from a matrix table is an :class:`.Expression`
which knows its type, its source matrix table, and whether it is a row field,
column field, entry field, or global field. Hail uses this context to know which
operations are allowed for a given expression.

When evaluated in a Python interpreter, we can see ``mt.locus`` is a
:class:`.LocusExpression` with type ``locus<GRCh37>``.

    >>> mt  # doctest: +SKIP_OUTPUT_CHECK
    <hail.matrixtable.MatrixTable at 0x1107e54a8>

    >>> mt.locus  # doctest: +SKIP_OUTPUT_CHECK
    <LocusExpression of type locus<GRCh37>>

Likewise, ``mt.DP`` is an :class:`.Int32Expression` with type ``int32``
and is an entry field of ``mt``.

Hail expressions can also :meth:`.Expression.describe` themselves, providing
information about their source matrix table or table and which keys index the
expression, if any. For example, ``mt.DP.describe()`` tells us that ``mt.DP``
has type ``int32`` and is an entry field of ``mt``, since it is indexed
by both rows and columns:

    >>> mt.DP.describe()  # doctest: +SKIP_OUTPUT_CHECK
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

    >>> mt.describe()  # doctest: +SKIP_OUTPUT_CHECK
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
    ----------------------------------------

Common Operations
=================

Like tables, Hail provides a number of methods for manipulating data in a
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

- :meth:`.MatrixTable.annotate_globals`
- :meth:`.MatrixTable.annotate_rows`
- :meth:`.MatrixTable.annotate_cols`
- :meth:`.MatrixTable.annotate_entries`

Annotate methods take keyword arguments where the key is the name of the new
field to add and the value is an expression specifying what should be added.

The simplest example is adding a new global field `foo` that just contains the constant
5.

    >>> mt_new = mt.annotate_globals(foo = 5)
    >>> print(mt_new.globals.dtype.pretty())
    struct {
        foo: int32
    }

Another example is adding a new row field `call_rate` which computes the fraction
of non-missing entries `GT` per row:

    >>> mt_new = mt.annotate_rows(call_rate = hl.agg.fraction(hl.is_defined(mt.GT)))

Annotate methods are also useful for updating values. For example, to update the
GT entry field to be missing if `GQ` is less than 20, we can do the following:

    >>> mt_new = mt.annotate_entries(GT = hl.or_missing(mt.GQ >= 20, mt.GT))

**Select**

Select is used to create a new schema for a dimension of the matrix table. Key
fields are always preserved even when not selected. For example, following the
matrix table schemas from importing a VCF file (shown above),
to create a hard calls dataset where each entry only contains the `GT` field
we can do the following:

    >>> mt_new = mt.select_entries('GT')
    >>> print(mt_new.entry.dtype.pretty())
    struct {
        GT: call
    }

:class:`.MatrixTable` has four select methods that select and create new fields:

- :meth:`.MatrixTable.select_globals`
- :meth:`.MatrixTable.select_rows`
- :meth:`.MatrixTable.select_cols`
- :meth:`.MatrixTable.select_entries`

Each method can take either strings referring to top-level fields, an attribute
reference (useful for accessing nested fields), as well as keyword arguments
``KEY=VALUE`` to compute new fields. The Python unpack operator ``**`` can be
used to specify that all fields of a Struct should become top level fields.
However, be aware that all top-level field names must be unique. In the
following example, `**mt['info']` would fail if `DP` already exists as an entry
field.

    >>> mt_new = mt.select_rows(**mt['info']) # doctest: +SKIP

The example below adds two new row fields. Keys are always preserved, so the
row keys ``locus`` and ``alleles`` will also be present in the new table.
``AC = mt.info.AC`` turns the subfield ``AC`` into a top-level field.

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

    >>> mt_new.replicate_num.show() # doctest: +SKIP_OUTPUT_CHECK
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
    showing top 10 rows

Aggregation
===========

:class:`.MatrixTable` has three methods to compute aggregate statistics.

- :meth:`.MatrixTable.aggregate_rows`
- :meth:`.MatrixTable.aggregate_cols`
- :meth:`.MatrixTable.aggregate_entries`

These methods take an aggregated expression and evaluate it, returning
a Python value.

An example of querying entries is to compute the global mean of field `GQ`:

    >>> mt.aggregate_entries(hl.agg.mean(mt.GQ))  # doctest: +SKIP_OUTPUT_CHECK
    67.73196915777027

It is possible to compute multiple values simultaneously by
creating a tuple or struct. This is encouraged, because grouping two
computations together is far more efficient by traversing the dataset only once
rather than twice.

    >>> mt.aggregate_entries((hl.agg.stats(mt.DP), hl.agg.stats(mt.GQ)))  # doctest: +SKIP_OUTPUT_CHECK
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

    >>> mt_ann = mt.annotate_cols(case_status = hl.if_else(hl.rand_bool(0.5),
    ...                                                    "CASE",
    ...                                                    "CONTROL"))

Next we group the columns by `case_status` and aggregate:

    >>> mt_grouped = (mt_ann.group_cols_by(mt_ann.case_status)
    ...                 .aggregate(gq_stats = hl.agg.stats(mt_ann.GQ)))
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
    struct{case_status: str}

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
if the keys of each source are compatible. Keys are compatible if they are the
same type, and share the same ordering in the case where tables have multiple keys.

If the keys are compatible, joins can then be performed using Python's bracket
notation ``[]``. This looks like ``right_table[left_table.key]``. The argument
inside the brackets is the key of the destination (left) table as a single value, or a
tuple if there are multiple destination keys.

For example, we can join a matrix table and a table in order to annotate the
rows of the matrix table with a field from the table. Let `gnomad_data` be a
:class:`.Table` keyed by two row fields with type
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
:meth:`.MatrixTable.sample_rows`. `describe` prints out the schema for all row
fields, column fields, entry fields, and global fields as well as the row keys
and column keys. `head` returns a new matrix table with only the first N rows.
`sample_rows` returns a new matrix table where the rows are randomly sampled with
frequency `p`.


To get the dimensions of the matrix table, use :meth:`.MatrixTable.count_rows`
and :meth:`.MatrixTable.count_cols`.
