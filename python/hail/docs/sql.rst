.. _sec-sql:

==================
Querying using SQL
==================

Since Hail uses the Parquet file format for data storage, a Hail VDS can be queried using
Hadoop SQL tools, like Hive or Impala. This mode of access may be convenient for users
who have ad hoc queries that they are able to express in SQL.

Note that SQL access is *read-only*: it is not possible to write Hail datasets using
SQL at the current time.

------
Impala
------

Each VDS should be registered in the Hive metastore to allow Impala to query it (Impala uses Hive's metastore to store table metadata). This is done by creating an external table in Hive, the "external" part means that the data is managed by an entity outside Hive (and
Impala). The table schema is read from one of the Parquet files in the VDS file
hierarchy.

To generate a Hive file:

    1. Copy a VCF file into HDFS

    .. code-block:: text

        $ hadoop fs -put src/test/resources/sample.vcf.bgz sample.vcf.bgz

    2. Convert the VCF file into a VDS using Hail::

        >>> hc.import_vcf("sample.vcf.bgz").write("sample.vds")

    3. Register the VDS as a Hive table

    .. code-block:: text

        $ PARQUET_DATA_FILE=$(hadoop fs -stat '%n' hdfs:///user/$USER/sample.vds/rdd.parquet/*.parquet | head -1)
        $ impala-shell -q "CREATE EXTERNAL TABLE variants LIKE PARQUET 'hdfs:///user/$USER/sample.vds/rdd.parquet/$PARQUET_DATA_FILE' STORED AS PARQUET LOCATION 'hdfs:///user/$USER/sample.vds/rdd.parquet'"


It is good practice to run Impala's ``COMPUTE STATS`` command on the newly-created table, so that subsequent queries run efficiently.

.. code-block:: text

    $ impala-shell -q "COMPUTE STATS variants"


Before running any queries it's worth understanding the table schema, which is easily
done by calling ``DESCRIBE`` on the table:

.. code-block:: text

    $ impala-shell -q "DESCRIBE variants"

.. code-block:: text

    +-------------+----------------------------------+-----------------------------+
    | name        | type                             | comment                     |
    +-------------+----------------------------------+-----------------------------+
    | variant     | struct<                          | Inferred from Parquet file. |
    |             |   contig:string,                 |                             |
    |             |   start:int,                     |                             |
    |             |   ref:string,                    |                             |
    |             |   altalleles:array<struct<       |                             |
    |             |     ref:string,                  |                             |
    |             |     alt:string                   |                             |
    |             |   >>                             |                             |
    |             | >                                |                             |
    | annotations | struct<                          | Inferred from Parquet file. |
    |             |   rsid:string,                   |                             |
    |             |   qual:double,                   |                             |
    |             |   filters:array<string>,         |                             |
    |             |   pass:boolean,                  |                             |
    |             |   info:struct<                   |                             |
    |             |     negative_train_site:boolean, |                             |
    |             |     hwp:double,                  |                             |
    |             |     ac:array<int>,               |                             |
    ...
    |             |   >                              |                             |
    |             | >                                |                             |
    | gs          | array<struct<                    | Inferred from Parquet file. |
    |             |   gt:int,                        |                             |
    |             |   ad:array<int>,                 |                             |
    |             |   dp:int,                        |                             |
    |             |   gq:int,                        |                             |
    |             |   px:array<int>,                 |                             |
    |             |   fakeref:boolean,               |                             |
    |             |   isdosage:boolean               |                             |
    |             | >>                               |                             |
    +-------------+----------------------------------+-----------------------------+

Notice that the schema is nested. The ``annotations`` type corresponds to the variant
annotation schema that is returned by :py:meth:`~hail.VariantDataset.variant_schema`:

.. code-block:: python

    >>> print(hc.read("sample.vds").variant_schema)

.. code-block:: text

    Struct {
        rsid: String,
        qual: Double,
        filters: Set[String],
        pass: Boolean,
        info: Struct {
            NEGATIVE_TRAIN_SITE: Boolean,
            HWP: Double,
            AC: Array[Int],
            ...
        }
    }

Here is an example query to find variants in a given interval. Note the way that the
array of alternate alleles is joined with the main table, and the use of the
``item`` keyword to refer to the value of the array element. Working with complex types
is explained in detail in the `Impala documentation <http://www.cloudera.com/documentation/enterprise/5-5-x/topics/impala_complex_types.html>`_.

.. code-block:: text

    $ impala-shell -q "SELECT variant.contig, variant.start, variant.ref, altalleles.item.alt, annotations.rsid FROM variants, variants.variant.altalleles WHERE variant.start > 13090000 AND variant.start < 13100000"

.. code-block:: text

    +----------------+---------------+-------------+----------+------------------+
    | variant.contig | variant.start | variant.ref | item.alt | annotations.rsid |
    +----------------+---------------+-------------+----------+------------------+
    | 20             | 13090728      | A           | T        | rs6109712        |
    | 20             | 13090733      | A           | AT       | .                |
    | 20             | 13090733      | AT          | A        | .                |
    | 20             | 13090745      | G           | C        | rs2236126        |
    | 20             | 13098135      | T           | C        | rs150175260      |
    +----------------+---------------+-------------+----------+------------------+

Here is another example showing how you can query the genotype information. Notice that
each genotype is represented by a whole row in the results. The ``genotype_pos`` column is
the index of the genotype for the variant.

.. code-block:: text

    $ impala-shell -q "SELECT variant.contig, variant.start, variant.ref, gs.pos AS genotype_pos, gs.item.gt AS gt FROM variants, variants.gs WHERE variant.start = 13090728 AND gs.pos >= 20 AND gs.pos < 25;"

.. code-block:: text

    +----------------+---------------+-------------+--------------+----+
    | variant.contig | variant.start | variant.ref | genotype_pos | gt |
    +----------------+---------------+-------------+--------------+----+
    | 20             | 13090728      | A           | 20           | 1  |
    | 20             | 13090728      | A           | 21           | 0  |
    | 20             | 13090728      | A           | 22           | 0  |
    | 20             | 13090728      | A           | 23           | 0  |
    | 20             | 13090728      | A           | 24           | 0  |
    +----------------+---------------+-------------+--------------+----+

We can also retrieve the values from the AD (Allelic Depths) array by doing a nested
query that returns one row per genotype and per AD value. The ``ad_pos`` column is
the index of the value in the AD array.

.. code-block:: text

    $ impala-shell -q "SELECT variant.contig, variant.start, variant.ref, gs.pos AS genotype_pos, gs.item.gt AS gt, ad.pos AS ad_pos, ad.item AS ad FROM variants, variants.gs, gs.ad WHERE variant.start = 13090728 LIMIT 6;"

.. code-block:: text

    +----------------+---------------+-------------+--------------+----+--------+----+
    | variant.contig | variant.start | variant.ref | genotype_pos | gt | ad_pos | ad |
    +----------------+---------------+-------------+--------------+----+--------+----+
    | 20             | 13090728      | A           | 0            | 0  | 0      | 28 |
    | 20             | 13090728      | A           | 0            | 0  | 1      | 0  |
    | 20             | 13090728      | A           | 1            | 0  | 0      | 20 |
    | 20             | 13090728      | A           | 1            | 0  | 1      | 0  |
    | 20             | 13090728      | A           | 2            | 0  | 0      | 11 |
    | 20             | 13090728      | A           | 2            | 0  | 1      | 0  |
    +----------------+---------------+-------------+--------------+----+--------+----+

If you no longer need to use SQL you can delete the table definition. Since the table
was registered as an external table the underlying data is *not* affected, so you can
still access the VDS from Hail.

.. code-block:: text

    $ impala-shell -q "DROP TABLE variants"
    $ hadoop fs -ls sample.vds
