# Hail columnar format

- **Format identifier:** `hail-columnar`
- **Spec version:** `0.1` (draft)
- **Status:** experimental. Versions `0.x` make no compatibility guarantee. At
  `1.0` the spec freezes, and every `1.x` reader must read all `1.x` data.

The keywords MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119. Terms in
**bold** are defined in [Terminology](#terminology).

## Background and rationale

Hail 3 moves to columnar storage and processing, and Hail 2 must be able to write
data that Hail 3 can process. Hail 2's **native format** is undocumented and
row-oriented, so only Hail can read Hail data, and projecting one field still
decodes every byte.

**The contract is an Arrow schema.** The columnar format is defined by an Arrow
schema, with Hail-specific types expressed as Arrow extension types. It is not
defined by any one file format. Every likely consumer of the data speaks Arrow
natively: DataFusion (the most likely Hail 3 engine), DuckDB, Polars and PyArrow.
This is what lets analyses written outside Hail operate on Hail data. The manifest
records the container of each data file, so other containers can be added without
changing the logical contract.

**The data files are Parquet first.** Vortex was the fastest container on a
DataFusion prototype's workload, but Parquet was chosen for Hail 2's writer:

- Vortex's peak memory was 3-7x Parquet's in the same benchmarks.
- On the JVM, Vortex is a large JNI jar that requires a newer Arrow than Spark
  ships. parquet-java is already on Hail 2's runtime classpath.
- Vortex has no columnar list layout yet, so projecting one field out of a
  list-typed entry field reads the whole list.
- Vortex only guarantees that new readers can read old files, so each writer has
  to pin a format version for its target readers.
- Every external tool reads Parquet.

None of this rules Vortex out for Hail 3, which can read Parquet written by Hail 2
and write Vortex itself.

**Alternatives considered:**

- *Extending the native format* with a new codec. This is the least Hail 2 work,
  but Hail 3 would then have to parse Hail 2's internal specs, type strings and
  byte-offset indexes forever, and the format would stay undocumented.
- *An open table format as the manifest* (Iceberg, Delta, DuckLake). These are
  built for mutable, catalogued tables. Hail's range partitioning on struct keys,
  its structured globals, the columns table and reference genomes would all become
  side metadata. Instead, data files carry Parquet field IDs and standard types,
  and the manifest carries what such a format would need, so conversion stays
  possible later.

**Consequences:**

- Parquet has no columnar layout for a second axis. Reading a subset of a matrix
  table's columns cheaply needs **column blocks**, which sit above the container.
- The format has no separate index. Key-range reads rely on the partition bounds
  in the manifest and on Parquet statistics. This is why loci are stored as
  order-preserving packed integers (section 5.2).

## Terminology

- **Hail 2**: the current release line of Hail (versions 0.2.x).
- **Hail 3**: the planned successor to Hail 2, which moves to columnar storage and
  processing.
- **Native format**: the row-oriented on-disk encoding Hail 2 uses for tables and
  matrix tables.
- **Columnar format**: the format this document specifies.
- **Manifest**: the metadata describing a stored table or matrix table: its schema,
  key, row partitions, and the data files that hold them.
- **Partitions table**: the part of the manifest that lists every row partition
  with its key bounds, row count and data files.
- **Row partition**: a contiguous, key-ordered range of a table's rows that is
  stored and processed as a unit.
- **Row file**: a data file holding the row fields of one row partition.
- **Entry file**: a data file holding the entries of one row partition for one
  column block.
- **Column block**: a contiguous range of a matrix table's columns whose entries
  are stored together, so that a subset of columns can be read without reading
  every entry.
- **Row ID**: a dataset-unique, order-preserving identifier for a stored row,
  derived from its row partition and its position within that partition.
- **Key**: the ordered prefix of a table's row fields (or a matrix table's row or
  column fields) by which its rows are sorted and partitioned.
- **Field**: a named member of a stored schema, such as an Arrow or Parquet schema.
- **Arrow schema**: an [Apache Arrow](https://arrow.apache.org/docs/format/Columnar.html)
  schema, including field metadata.

## 1. Overview

A stored table or matrix table in the **columnar format** is a directory holding:

- a **manifest**, made up of a JSON root (`manifest.json`) and a **partitions
  table** (`partitions.parquet`);
- one **row file** per **row partition**;
- for matrix tables, one **entry file** per (row partition, **column block**), a
  columns file, and a globals file;
- for tables, an optional globals file.

The logical contract is an **Arrow schema**. Every data file is a Parquet file
whose schema is the Parquet translation of that Arrow schema. Hail-specific types
are expressed as Arrow extension types (section 5). A reader recovers every Hail
type from the Arrow schema alone. The manifest stores no Hail type strings.

## 2. Directory layout

```
<path>/
  manifest.json                                  JSON root; written last
  partitions.parquet                             partitions table
  globals.parquet                                globals file (omitted if globals has no fields)
  cols.parquet                                   columns file (matrix tables only)
  rows/part-<i>-<uuid>.parquet                   row files
  entries/part-<i>-block-<c>-<uuid>.parquet      entry files (matrix tables only)
```

- `<i>` is the row partition index and `<c>` the column block index. Both are
  zero-padded decimal.
- `<uuid>` is a random UUID4. It makes a retried task's output distinct from a
  failed attempt's.
- All paths stored in the manifest are relative to `<path>`. Readers MUST NOT
  derive file names from the pattern above; they MUST use the paths in the
  manifest.
- Files under `<path>` that the manifest doesn't reference (orphans from failed
  tasks) MUST be ignored by readers.

### 2.1 Commit protocol

`manifest.json` MUST be the last file written. A dataset is committed if and only
if `manifest.json` exists. Writers MUST NOT write `_SUCCESS` files.

## 3. Manifest root (`manifest.json`)

UTF-8 JSON. Fields not listed here MUST be rejected by `0.x` readers.

```json
{
  "format": "hail-columnar",
  "version": "0.1",
  "kind": "table",
  "rows": {
    "schema": "<base64 Arrow IPC schema>",
    "key": ["locus", "alleles"],
    "distinctly_keyed": false
  },
  "globals": {
    "schema": "<base64 Arrow IPC schema>",
    "file": { "path": "globals.parquet", "container": "parquet" }
  },
  "partitions": { "path": "partitions.parquet", "container": "parquet" },
  "internal_fields": { "row_id": "__row_id" },
  "reference_genomes": []
}
```

A matrix table (`"kind": "matrix_table"`) also has:

```json
{
  "cols": {
    "schema": "<base64 Arrow IPC schema>",
    "key": ["s"],
    "count": 2504,
    "file": { "path": "cols.parquet", "container": "parquet" }
  },
  "entries": {
    "schema": "<base64 Arrow IPC schema>",
    "layout": "dense",
    "column_blocks": [ { "start": 0, "end": 2504 } ]
  },
  "internal_fields": {
    "row_id": "__row_id",
    "col_idx": "__col_idx",
    "entry_valid": "__entry_valid"
  }
}
```

Field rules:

- **`schema`**: the Arrow schema of that component's data files, including the
  internal fields. It is serialized as an encapsulated Arrow IPC `Schema` message
  and base64-encoded (RFC 4648 section 4: standard alphabet, with padding). The
  encapsulated message is the `0xFFFFFFFF` continuation marker, the little-endian
  `int32` metadata length, and the flatbuffer `Message` padded to a multiple of 8
  bytes. It has no body. See [Encapsulated message
  format](https://arrow.apache.org/docs/format/Columnar.html#encapsulated-message-format).
  This is the same encoding Arrow uses for the `ARROW:schema` key in Parquet
  footers (section 4.6). Each component stores its schema here so a reader can
  plan without opening data files, and so that a table with zero row partitions
  still has a schema.
- **Schema consistency:** the `ARROW:schema` in each data file's footer MUST
  describe the same schema as its component's `schema`: the same fields in the same
  order, with the same types, nullability and field metadata, including the
  `PARQUET:field_id` keys (section 4.6). Readers MAY check
  this, and MUST reject the dataset if they find a mismatch.
- **`key`**: the names of the key fields, in key order. An empty list means the
  table is unkeyed.
- **`distinctly_keyed`**: `true` if no two rows share a key.
- **`globals.file`**: `null` if the global struct has no fields. Parquet can't
  represent a file with no fields. `globals.schema` is always present, and is the
  empty schema in that case.
- **`count`**: the number of columns.
- **`layout`**: `"dense"` (section 4.3). The value `"long"` is reserved for a
  future sparse layout and MUST be rejected by `0.x` readers.
- **`column_blocks`**: half-open ranges `[start, end)` of column indices. They
  MUST be in ascending order, contiguous, and together cover `[0, count)`. Writers
  of version `0.1` MUST write exactly one column block.
- **`container`**: `"parquet"`. This is the only value defined in `0.x`.
- **`internal_fields`**: the names of the fields this spec adds (sections 4.2 to
  4.4). A writer chooses each name and MUST ensure it doesn't collide with a user
  field in the same file. The default is the name shown above, with a numeric
  suffix appended until it is unique. Readers MUST look these names up here and
  MUST NOT assume the defaults.
- **`reference_genomes`**: full definitions of every non-builtin reference genome
  used by any locus or interval-of-locus field (section 3.1). Builtin genomes
  (`GRCh37`, `GRCh38`, `GRCm38`, `CanFam3`) MUST NOT be listed.

### 3.1 Reference genome definitions

Each entry has the same shape as Hail's reference genome JSON
(`hail/hail/resources/reference/grch38.json` is an example):

```json
{
  "name": "MyGenome",
  "contigs": [ { "name": "1", "length": 248956422 } ],
  "xContigs": ["X"], "yContigs": ["Y"], "mtContigs": ["MT"],
  "par": [ { "start": { "contig": "X", "position": 10001 },
             "end":   { "contig": "X", "position": 2781480 } } ]
}
```

The order of `contigs` is the genome's contig order.

## 4. Data files

### 4.1 Partitions table (`partitions.parquet`)

One row per row partition, ordered by `partition_index`.

| Field | Arrow type | Notes |
|---|---|---|
| `partition_index` | `int32`, non-null | 0, 1, 2, ... |
| `start` | struct of the key fields, nullable | key of the partition's first row; keyed tables only |
| `end` | struct of the key fields, nullable | key of the partition's last row; keyed tables only |
| `row_count` | `int64`, non-null | |
| `row_file` | `struct<path: utf8, container: utf8>`, non-null | |
| `entry_files` | `list<struct<path: utf8, container: utf8>>`, nullable | matrix tables only; index *c* is column block *c* |

The `start` and `end` structs use exactly the Arrow types of the key fields in the
row schema, including extension types. Bounds are therefore stored in the same
value encoding as the data. Both are `null` for partitions with `row_count = 0`.
The partitions table of an unkeyed table has no `start` or `end` field, because
Parquet can't store a struct with no fields.

**Key invariants** (keyed tables):

- Within a row partition, rows are sorted by key. Missing values sort after all
  non-missing values, field by field, as in Hail.
- In a non-empty partition, `start` is the full key of its first row and `end` the
  full key of its last row. Both bounds are inclusive.
- No two row partitions hold rows with equal keys. For non-empty partitions *i* <
  *j*, `end` of *i* is strictly less than `start` of *j*.

Because of the last invariant, all rows that share a key are in one row partition,
so a heavily repeated key can make one partition arbitrarily large. A writer may
have to repartition its input to satisfy these invariants.

### 4.2 Row files

One per row partition. The fields are:

1. the **Row ID** field (`internal_fields.row_id`): `int64`, non-null;
2. the row fields, in the row schema's order. These include the key fields.

**Row ID**: `(partition_index << 40) | position_in_partition`, where
`position_in_partition` starts at 0. Row IDs are unique within the dataset and
increase in dataset order. They identify a row within one write only; rewriting
the data with different partitioning changes them. This limits a dataset to
2^23 row partitions and 2^40 rows per row partition.

### 4.3 Entry files (dense layout)

One per (row partition, column block). An entry file MUST contain exactly the same
rows, in the same order, as its partition's row file. Let *n* be the width of the
column block (`end - start`). The fields are:

1. the **Row ID** field: `int64`, non-null, equal to the row file's Row ID for the
   same row;
2. the entry validity field (`internal_fields.entry_valid`): `list<bool>`. The
   list is non-null and has exactly *n* elements, each non-null;
3. one field per entry field, in the entry schema's order. Each is a `list<T>`,
   where `T` is the entry field's Arrow type (section 5). The list is non-null and
   has exactly *n* elements. The elements are nullable.

The element at list position *j* belongs to column `start + j`.

- **Missing entry:** `entry_valid[j] = false`, and every entry field's element *j*
  is null.
- **Entry present with a missing field:** `entry_valid[j] = true`, and that field's
  element *j* is null.

### 4.4 Columns file (`cols.parquet`)

One row per column, in column-index order. The fields are:

1. the column index field (`internal_fields.col_idx`): `int32`, non-null, equal to
   the row's position (0, 1, 2, ...);
2. the column fields, in the column schema's order.

The column index field lets generic engines join an entry's list position, plus
its column block's `start`, to the column's fields. It also guarantees that the
file has at least one field.

### 4.5 Globals file (`globals.parquet`)

Exactly one row, holding the global fields. It is omitted when the global struct
has no fields (see `globals.file` in section 3).

### 4.6 Parquet requirements

These apply to every data file:

- **Nested types:** standard three-level `LIST` and standard `MAP` encodings.
  Strings use the `STRING` logical type. No Hail-private physical encodings.
- **Field IDs:** IDs are assigned depth-first (pre-order) to every Arrow field,
  starting at 1, except a map's `entries` field. Each Arrow field records its ID
  in the field metadata key `PARQUET:field_id`, as a decimal string, and the
  corresponding Parquet schema node carries the same `field_id`. The Parquet root
  and the repeated groups of the `LIST` and `MAP` encodings have no field ID. This
  is the convention Arrow's Parquet readers and writers use. The manifest's
  `schema` (section 3) carries the same keys.
- **Arrow schema:** the footer's key-value metadata MUST include `ARROW:schema`,
  holding the file's Arrow schema in the encoding described under `schema` in
  section 3, so Arrow readers recover extension types.
- **Statistics and page index:** writers MUST write column statistics for every
  primitive leaf, and MUST write the ColumnIndex and OffsetIndex.
- **Compression:** zstd. The compression level is not fixed by this spec.

#### 4.6.1 Row groups

- Within a row partition, the row file and all of its entry files MUST have
  identical row-group boundaries: the same number of row groups, with the same row
  count in each.
- **Leaf cap:** within any row group, no single leaf field of an entry file may
  hold more than 2^30 values. Leaf values for an entry field are counted as rows ×
  *n* for a primitive `T`, and more for nested `T` such as `list<int32>`. This
  keeps every row group readable by Arrow readers that use 32-bit list offsets.
- Subject to the leaf cap, writers SHOULD aim for a target row-group byte size. The
  target is not fixed by this spec.

## 5. Type mapping

Nullability: a Hail field or element is non-null in Arrow if and only if it is
required in Hail.

Extension types use the standard Arrow keys `ARROW:extension:name` and
`ARROW:extension:metadata`. Extension metadata is a UTF-8 JSON object, and is `{}`
when there are no parameters.

| Hail type | Arrow storage type | Extension name | Extension metadata |
|---|---|---|---|
| `int32` | `int32` | | |
| `int64` | `int64` | | |
| `float32` | `float32` | | |
| `float64` | `float64` | | |
| `bool` | `bool` | | |
| `str` | `utf8` | | |
| `binary` | `binary` | | |
| `struct{...}` | `struct<...>` | | |
| `tuple(...)` | `struct<"0": ..., "1": ..., ...>` | `hail.tuple` | `{}` |
| `array<T>` | `list<T>` | | |
| `set<T>` | `list<T>`, sorted, distinct | `hail.set` | `{}` |
| `dict<K, V>` | `map<K, V>`, `keysSorted = true`, non-null keys | | |
| `locus<RG>` | `int64` | `hail.locus` | `{"reference_genome": RG, "contigs": [...]}` |
| `interval<T>` | `struct<start: T, end: T, includes_start: bool, includes_end: bool>` | `hail.interval` | `{}` |
| `call` | `int32` | `hail.call` | `{"encoding": 1}` |
| `ndarray<T, N>` | `struct<data: list<T>, shape: fixed_size_list<int32>[N]>` | `arrow.variable_shape_tensor` | per the Arrow canonical extension spec |
| RNG state | not storable | | |

### 5.1 Dicts

Writers MUST fail when they encounter a missing dict key. The write then never
commits (section 2.1). Missing dict values are allowed.

### 5.2 Locus

A locus is stored as `(contig_index << 32) | position`, a signed `int64`:

- `contig_index` is the contig's position in `contigs`, which lists the reference
  genome's contigs in genome order.
- `position` is Hail's 1-based position.

The `contigs` list in the extension metadata is required even for builtin
genomes, so each file can decode its loci on its own. Because of this encoding,
integer order equals Hail's locus order, which makes Parquet min/max statistics
valid for key-range pruning.

### 5.3 Call (`encoding: 1`)

Hail's packed call representation, transcribed from
`hail/hail/src/is/hail/variant/Call.scala`:

| Bits | Meaning |
|---|---|
| 0 | phased flag |
| 1-2 | ploidy (0, 1 or 2; the value 3 is reserved) |
| 3-31 | allele representation (29 bits) |

The allele representation depends on ploidy:

- **Ploidy 0:** 0.
- **Ploidy 1:** the allele index.
- **Ploidy 2, unphased:** for alleles *j* ≤ *k*, `k(k+1)/2 + j`.
- **Ploidy 2, phased:** for alleles *a*|*b*, the same formula with *j* = *a* and
  *k* = *a* + *b*.

### 5.4 NDArray

Data is stored row-major. `shape` has length *N*, the number of dimensions.

## 6. Conformance

A set of fixture datasets is kept with Hail's tests. They are compared by value,
not by bytes. At minimum the fixtures cover:

- a keyed table and an unkeyed table;
- a table with no globals;
- a matrix table that has missing entries and present entries with missing fields;
- every type in section 5, as both required and nullable fields where Hail allows
  it;
- a non-builtin reference genome.

Hail 2's tests MUST read every fixture. Hail 2's writer output MUST pass the
pyarrow conformance tests: schema, extension metadata, field IDs, and reading each
row group whole.
