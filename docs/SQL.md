# Querying using SQL

Since Hail uses the Parquet file format for data storage, a Hail VDS can be queried using 
Hadoop SQL tools, like Hive or Impala. This mode of access may be convenient for users 
who have ad hoc queries that they are able to express in SQL.

Note that SQL access is _read-only_: it is not possible to write Hail datasets using 
SQL at the current time.


## <a name="impala"></a> Impala

Each VDS should be registered in the Hive metastore to allow Impala to query it. 
(Impala uses Hive's metastore to store table metadata.) This is done by creating an external table in Hive, the "external" part means that the data is managed by an entity outside Hive (and Impala). The table schema is read from the *_metadata* file in the VDS file 
hierarchy.

The following commands copy a VCF file into HDFS, turn it into a VDS, then register it 
as a Hive table:

```
$ hadoop fs -put src/test/resources/sample.vcf.bgz sample.vcf.bgz
$ hail importvcf sample.vcf.bgz write -o sample.vds
$ impala-shell -q "CREATE EXTERNAL TABLE variants LIKE PARQUET 'hdfs:///user/$USER/sample.vds/rdd.parquet/_metadata' STORED AS PARQUET LOCATION 'hdfs:///user/$USER/sample.vds/rdd.parquet'"
```

It is good practice to run Impala's `COMPUTE STATS` command on the newly-created table,
 so that subsequent queries run efficiently.
 
``` 
$ impala-shell -q "COMPUTE STATS variants"
```

Before running any queries it's worth understanding the table schema, which is easily 
done by calling `DESCRIBE` on the table:

```
$ impala-shell -q "DESCRIBE variants"
```

```
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
| gs          | struct<                          | Inferred from Parquet file. |
|             |   decomplen:int,                 |                             |
|             |   bytes:string                   |                             |
|             | >                                |                             |
+-------------+----------------------------------+-----------------------------+
```

Notice that the schema is nested. The `annotations` type corresponds to the variant
 annotation schema that is displayed using Hail's `printschema` command:

```
$ hail read -i sample.vds printschema
```

```
Variant annotation schema:
va: Struct {
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
```

The genotypes (`gs`) are encoded in a compact binary representation, which means they 
cannot be queried using SQL.

Here is an example query to find variants in a given interval. Note the way that the 
array of alternate alleles is joined with the main table, and the use of the 
`item` keyword to refer to the value of the array element. Working with complex types 
is explained in detail in the [Impala documentation](http://www.cloudera.com/documentation/enterprise/5-5-x/topics/impala_complex_types.html).

```
$ impala-shell -q "SELECT variant.contig, variant.start, variant.ref, altalleles.item.alt, annotations.rsid FROM variants, variants.variant.altalleles WHERE variant.start > 13090000 AND variant.start < 13100000"
```

```
+----------------+---------------+-------------+----------+------------------+
| variant.contig | variant.start | variant.ref | item.alt | annotations.rsid |
+----------------+---------------+-------------+----------+------------------+
| 20             | 13090728      | A           | T        | rs6109712        |
| 20             | 13090733      | A           | AT       | .                |
| 20             | 13090733      | AT          | A        | .                |
| 20             | 13090745      | G           | C        | rs2236126        |
| 20             | 13098135      | T           | C        | rs150175260      |
+----------------+---------------+-------------+----------+------------------+
```

If you no longer need to use SQL you can delete the table definition. Since the table 
was registered as an external table the underlying data is *not* affected, so you can 
still access the VDS from Hail.

```
$ impala-shell -q "DROP TABLE variants"
$ hadoop fs -ls sample.vds
```