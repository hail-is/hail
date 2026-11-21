# Hail value boundary

- **Spec version:** `0.1` (draft)
- **Status:** experimental. Versions `0.x` make no compatibility guarantee. The
  Python library and the engine of one Hail release implement the same version.

The keywords MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119. Terms in
**bold** are defined in [Terminology](#terminology). "The columnar format spec"
is [columnar-format-spec.md](columnar-format-spec.md).

## Background and rationale

Hail's Python library sends literals to the query engine and receives the results
of the expressions it asks the engine to evaluate. Hail 2 moves both in the
**Python wire encoding**, a private binary encoding that only Hail's Python
library reads and writes, and that Python decodes one value at a time.

Hail 3's engine will hold and return data as Arrow, in the type mapping the
columnar format already defines. This spec makes the **value boundary** Arrow
too. The Python library can then decode a result from any engine that speaks
Arrow, and an engine that reads columnar files can return what it read without
converting it.

**The boundary uses the columnar format's type mapping as written.** That
includes the parts of the mapping that exist only because Parquet needs them: an
empty struct is a `bool`, and a tensor's `shape` is nullable. Keeping one
mapping means each side has one converter, and data read from a columnar file is
already in the form the boundary carries. The boundary differs from the mapping
in two ways: it adds three forms (section 3), and it refuses some values the
mapping can express (section 4).

**Three forms exist only at the boundary.** Each covers something the stored
mapping can't do or shouldn't do here:

- Hail 2 returns dicts that have a missing key, and an Arrow `map` can't hold
  one.
- Python can't sort a set or a dict in Hail's order, so a literal's sets and
  dicts arrive unsorted.
- The stored form of a locus carries its reference genome's whole contig list,
  which is 86 KB for GRCh38. At the boundary both ends already hold the
  reference genome.

**Alternatives considered:**

- *Parquet bytes as the payload.* This would reuse the columnar reader and
  writer in both directions. It pays for encoding, compression and a footer on
  every call, it can't stream, and a Hail 3 engine won't speak it.
- *A separate type mapping for the boundary*, without the Parquet workarounds.
  The Python library has to read the stored mapping anyway, and an engine would
  have to convert what it read from files before returning it.

## Terminology

- **Value boundary**: where a single Hail value crosses between the Python
  library and the engine, as a result or as a literal.
- **Engine**: the query engine that evaluates expressions. In Hail 2 it is the
  JVM.
- **Result**: the value of an expression the Python library asked the engine to
  evaluate, on its way to Python.
- **Literal**: a Python value that the Python library embeds in an expression, on
  its way to the engine.
- **Sender** and **reader**: the two ends of one crossing. The engine sends a
  result and the Python library reads it. The Python library sends a literal
  and the engine reads it.
- **Session**: the Python library and the engine it is connected to. A reference
  genome is defined in a session under its name, and both ends hold its
  definition.
- **Required** and **optional**: a field, element, key or value is required when
  the sender knows it is never missing, and optional otherwise. The engine
  works this out for each expression. The Python library doesn't, so everything
  it sends is optional.
- **Hail's order**: the order Hail sorts the values of a type in, in which a
  missing value comes after every other value.
- **Distinct**: no two of the elements, or keys, are equal as Hail judges
  equality when it builds a set or a dict. Two missing values are equal there.
  A NaN is not equal to itself there, so a set can hold more than one.
- The columnar format spec relies on Hail's order and on distinctness too, for
  `hail.set` and for sorted `map` keys. Neither spec defines them yet.
- **Python wire encoding**: the private binary encoding in which values cross
  the value boundary in Hail 2 today. This spec does not define it.
- **Type mapping**: the correspondence between Hail types and Arrow types in
  section 5 of the columnar format spec.
- **Hail extension type**: an Arrow extension type that Hail defines, such as
  `hail.locus`.
- **Stream**: an [Arrow IPC
  stream](https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format).

## 1. Overview

A value crosses the **value boundary** as a **stream** that holds exactly that
one value. Results and literals use the same stream (section 2). Its schema uses
the **type mapping**, extended by three forms that are legal only at the
boundary (section 3). Some values fail instead of crossing (section 4).
Sections 5 and 6 say how Hail 2 requests a result and embeds a literal.

## 2. Payload

### 2.1 Stream

The payload is an Arrow IPC stream. A sender MUST write, in order:

1. one Schema message;
2. one RecordBatch message, holding exactly one row;
3. the end-of-stream marker.

A sender MUST also:

- write metadata version `V5`, little-endian;
- write no dictionary batch, and dictionary-encode no field;
- leave the record batch's buffers uncompressed.

A reader MUST reject a stream that has no record batch or more than one, a
record batch whose row count is not one, and a dictionary-encoded field. A
reader MAY reject a stream that breaks any other rule in section 2.1, and
SHOULD NOT depend on the end-of-stream marker.

### 2.2 Schema

The schema MUST have exactly one field, named `value`. Its type is the Arrow
type of the value being sent. A reader MUST reject a schema that has no field,
more than one field, or a field with another name.

- **Nullability.** A field is non-null if and only if it is **required**, which
  is the rule at the start of section 5 of the columnar format spec. Nothing the
  Python library sends is required, so it writes every field nullable. The
  fields whose nullability that section fixes stay as it fixes them, whoever
  the sender is. `hail.dict` sets the nullability of its own fields (section
  3.1).
- **A missing value** is a null in the one row, so its field has to be nullable.
- **Field metadata.** Extension types use the standard keys
  `ARROW:extension:name` and `ARROW:extension:metadata`. A reader MUST ignore
  any other field metadata key. That includes `PARQUET:field_id`, which an
  engine may pass through from a columnar file. A sender need not write field
  IDs.
- **Schema metadata.** This spec defines none. A reader MUST ignore it.

### 2.3 Void

An expression of type void has no value. Its result is an empty payload of zero
bytes, with no stream.

## 3. Type mapping

The type mapping is section 5 of the columnar format spec. Every rule in that
section applies here, and that covers the nullable tensor `shape` (5.4) and
`hail.empty_struct` (5.5).

Three further forms are legal only at the value boundary. They MUST NOT be
written to stored data (columnar format spec, section 5.6).

| Form | In a result | In a literal |
|---|---|---|
| `hail.dict` (3.1) | for a dict whose key type is optional | for any dict |
| `"sorted": false` on `hail.set` and `hail.dict` (3.2) | MUST NOT appear | allowed |
| `hail.locus` with `contigs_sha256` (3.3) | SHOULD be used | SHOULD be used |

The stored forms stay legal in both directions. A reader MUST accept an Arrow
`map` wherever it accepts a `hail.dict`, and a `hail.locus` with `contigs`
wherever it accepts one with `contigs_sha256`.

### 3.1 Dicts with an optional key type (`hail.dict`)

| Hail type | Arrow storage type | Extension name | Extension metadata |
|---|---|---|---|
| `dict<K, V>` | `list<struct<key: K, value: V>>` | `hail.dict` | `{}`, optionally with `sorted` (3.2) |

- The struct is non-null. Its fields are named `key` and `value`, in that order.
- `key` is always nullable. `value` follows section 2.2.
- Keys are distinct, and so at most one entry has a null key.
- Entries are sorted by key in Hail's order, with the null key last, unless the
  metadata says `"sorted": false` (section 3.2).

`hail.dict` and `map` are two Arrow forms of the same Hail type. A sender MUST
NOT use `map` unless it knows that no key of the dict is missing, which is when
the dict's key type is required. The engine SHOULD use `map` in that case, as
stored data does, and MUST use `hail.dict` in every other. A `map` at the
boundary therefore never has a missing key to write, and the failure that
section 5.1 of the columnar format spec requires can't occur.

A `map` MUST have `keysSorted = true`, as in stored data, and a reader MUST
reject one that doesn't. A sender that can't sort a dict's keys uses `hail.dict`
with `"sorted": false`.

### 3.2 Unsorted sets and dicts (`"sorted": false`)

In a literal, the extension metadata of a `hail.set` or a `hail.dict` MAY
contain `"sorted": false`. It means the list is in no particular order, and its
elements, or its keys, need not be distinct. `"sorted": true` means the same as
leaving the key out.

The engine MUST build the value as Hail builds a set from an array, or a dict
from an array of key and value pairs:

- It sorts in Hail's order, with a missing element or key last.
- Elements that are equal become one element. Two NaNs are not equal, so both
  stay.
- Missing elements are equal to one another, and so are missing keys. A set
  keeps at most one missing element, and a dict at most one entry with a missing
  key.
- When two entries of a dict have equal keys, one of them survives. Which one
  is not specified.

A `hail.set` or `hail.dict` that doesn't say `"sorted": false`, and every
`map`, promises that it is sorted in Hail's order and distinct. A sender MUST
NOT break that promise, and a reader MAY rely on it. What a reader does with a
list that breaks it is not specified.

A result MUST NOT use `"sorted": false`.

### 3.3 Locus with a contig digest (`contigs_sha256`)

At the boundary the extension metadata of `hail.locus` SHOULD name the reference
genome's contigs by a digest:

```json
{"reference_genome": "GRCh38",
 "contigs_sha256": "8fb76f415f8756265929578d7c68b59f2e7012796ad22e5f1a1de4acc1ffb915"}
```

The metadata MUST have exactly one of `contigs` and `contigs_sha256`, and a
reader MUST reject metadata that has both or neither. `contigs` is the list of
contig names of section 5.2 of the columnar format spec. The value encoding is
the same for both: `(contig_index << 32) | position`, where `contig_index` is
the contig's position in the reference genome's contig order.

**The digest** is the SHA-256 of the contig names in the reference genome's
contig order. Each name is hashed as the length of its UTF-8 encoding, written
as a 4-byte little-endian unsigned integer, followed by those UTF-8 bytes. The
digest is stored as 64 lowercase hexadecimal digits.

| Contigs | Bytes hashed | `contigs_sha256` |
|---|---|---|
| `["1", "2", "X"]` | `01 00 00 00 31 01 00 00 00 32 01 00 00 00 58` | `a1df3ec4fcb5d3c5a77f0e9f2af4b9bc4d1a0d3ac51f80f2caaeac5851e8cfc2` |
| GRCh37 (84 contigs) | | `8c0f2c4dbdd5a1cb14efbbe874ba994288f61e9975ba4c73212e4bf26e6a1dee` |
| GRCh38 (3366 contigs) | | `8fb76f415f8756265929578d7c68b59f2e7012796ad22e5f1a1de4acc1ffb915` |

A reader MUST look `reference_genome` up in the session, and MUST fail if it
isn't defined there. It MUST then compare the digest, or the `contigs` list,
with the contig names of its own definition, in order, and fail with an error
that names the reference genome if they differ. A contig index means the same
contig at both ends only when this check passes.

### 3.4 Converting a field to a Hail type

A reader converts the `value` field to a Hail type by reading the type mapping
in reverse, with the forms of sections 3.1 to 3.3 added.

- A reader MUST reject a field, at any depth, whose Arrow type the mapping and
  those sections assign to no Hail type. That covers `large_utf8`,
  `large_binary`, `large_list`, the view types, and an extension name the reader
  doesn't know.
- A reader MUST ignore a key it doesn't know in the extension metadata of a Hail
  extension type it does know.
- A reader MUST reject a value it doesn't support for a key it does know, such
  as a `hail.call` whose `encoding` is not 1.
- A sender MUST NOT write the `permutation` key of `arrow.variable_shape_tensor`,
  which would change the layout of `data`, and a reader MUST reject a tensor
  that has it.
- Nullability never changes which Hail type a field converts to.
- `map` and `hail.dict` convert to the same dict type.
- A `hail.locus` converts to the locus type of the reference genome it names.

## 4. Values that cannot cross

The boundary carries fewer values than the type mapping can express. A sender
MUST fail when it is asked to send any of these:

- A value of a type the type mapping marks as not storable.
- A locus whose contig is not in its reference genome, or whose position is
  negative. A packed locus can't represent either.
- An ndarray with a dimension of 2^31 or more (columnar format spec, section
  5.4).
- An ndarray whose element type is not one of `int32`, `int64`, `float32`,
  `float64` and `bool`. This is narrower than the type mapping, which puts no
  limit on an ndarray's element type.
- A value that needs an offset beyond 2^31 - 1 in a `list`, `map`, `utf8` or
  `binary` array.

The error names the field, or the value, that could not be sent. A sender MUST
NOT fall back to the Python wire encoding or to any other form.

For a result, the request fails as any failed request does. For a literal,
building the expression fails in the Python library, before any request is made.

## 5. Results

This section and the next bind the spec to Hail 2's request protocol and IR
text. They don't define either beyond what they add.

The Python library asks for a result with an `Execute` request, whose payload
is a JSON object. The object gains a member, `result_format`:

- `"arrow-ipc"`: the response body is the payload of section 2. The object's
  existing `stream_codec` member, which configures the Python wire encoding, is
  ignored.
- absent: the response is in the Python wire encoding, as before this spec.
- any other value: the engine MUST fail the request.

In a result:

- sets and dicts are sorted, and `"sorted": false` MUST NOT appear;
- a dict is a `map` or a `hail.dict` according to its key type (section 3.1);
- loci SHOULD carry `contigs_sha256`.

The Python library knows the Hail type of the expression it asked for. It MUST
convert the stream's `value` field to a Hail type (section 3.4) and MUST fail
when that is not the type it asked for.

## 6. Literals

A literal is embedded in the IR text as:

```
(ArrowLiteral <type> "<base64>")
```

- `<type>` is the literal's Hail type, written as the IR writes any type.
- `<base64>` is the payload of section 2, base64-encoded (RFC 4648 section 4:
  standard alphabet, with padding).

The engine MUST convert the stream's `value` field to a Hail type (section 3.4)
and MUST fail when that is not `<type>`.

In a literal:

- sets and dicts MAY be unsorted (section 3.2), and a dict MAY be a `hail.dict`
  whatever its key type;
- loci SHOULD carry `contigs_sha256`. The Python library packs each locus
  against the contig order of the reference genome as it holds it, and the
  engine's check (section 3.3) catches a session whose two ends disagree about
  that order.

A primitive value is one of type `int32`, `int64`, `float32`, `float64`, `bool`
or `str`. The IR has its own forms for primitive constants, and the Python
library MAY use them. It MUST embed every other literal as an `ArrowLiteral`
when it uses this spec for literals.

## 7. Python values (informative)

This section records what Hail's Python library returns for a result. It
doesn't bind other readers.

| Hail type | Python value |
|---|---|
| `int32`, `int64` | `int` |
| `float32`, `float64` | `float` |
| `bool` | `bool` |
| `str` | `str` |
| `call` | `hl.Call` |
| `locus<RG>` | `hl.Locus` |
| `interval<T>` | `hl.Interval` |
| `array<T>` | `list` |
| `set<T>` | `set` |
| `dict<K, V>` | `dict` |
| `struct{...}` | `hl.Struct` |
| `tuple(...)` | `tuple` |
| `ndarray<T, N>` | `numpy.ndarray` |

A missing value is `None`, at any depth.

**Freezing.** A dict key and a set element have to be hashable. Inside either
one, at any depth, an array is a `frozenlist`, a set is a `frozenset` and a dict
is a `frozendict`. `frozenlist` and `frozendict` are classes of Hail's Python
library. Everywhere else the values are the plain types in the table.

**ndarrays** are C-contiguous and writable, and own their data. The dtype is the
numpy type of the same name as the element type. A zero-dimensional ndarray and
a shape with a zero in it are both valid.

## 8. Conformance

- The digests in section 3.3 are test vectors. An implementation MUST reproduce
  them.
- A stream the engine writes MUST pass full validation in pyarrow 26 or later,
  the version Hail's Python library requires, and the library MUST convert its
  `value` field to the Hail type of the expression.
- For as long as Hail 2 still has the Python wire encoding, its tests evaluate
  an expression of every type its Python API can build both ways, and require
  equal values of the same Python types. The cases cover a missing value at each
  level, empty containers, containers nested in containers, a dict with a
  missing key, a set with a missing element, and a zero-dimensional ndarray.
- Literal tests cover every type in the same way, and also an unsorted set, an
  unsorted dict and a reference genome whose contig order is not lexicographic.

## 9. Implementation status

This section is informative. It records where Hail 2's implementation falls
short of this spec. Update it as the gaps close.

Hail 2 implements none of this spec yet. Results and literals cross the value
boundary in the Python wire encoding. The type mapping exists on the JVM for the
stored forms only (`is.hail.types.arrow`), and Hail's Python library has no
Arrow types.
