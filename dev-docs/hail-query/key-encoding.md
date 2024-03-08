RFC (will replace with implementing commit if/when implemented)

# Idea
Hail pervasively uses ordering of keys, such as in merge-joins, btree indices, or ordered dictionaries. We originally code-generated comparison functions for key types as needed.

Now, for any key type, we create an encoding to variable-length byte arrays, which preserves the key ordering. This has two main benefits over the code-generated comparison functions:
- Algorithms and data structures which use key comparisons can be written monomorphically, with `memcmp` as the only comparison function needed. As a result, many such algorithms and data structures no longer need to be code-generated, simplifying the implementation and testing.
- The generated comparison of a compound key involves many branches, while `memcmp` is branchless and highly optimized in hardware.

Our implementation was inspired by the [Fast and Memory Efficient Multi-Column Sorts in Apache Arrow Rust](https://arrow.apache.org/blog/2022/11/07/multi-column-sorts-in-arrow-rust-part-2/) blog post. But while they've optimized for vectorized encoding (which we currently can't do), we've preferred simplicity and smaller encodings.

# Design
Type encoders can emit three kinds of output to a byte array buffer:
- byte - simply add a byte to the result, first padding an incomplete byte if necessary
- bit - add a bit to the result, possibly leaving an incomplete byte. We must know statically how many bits are used in the byte.
- pad - add `0`s to pad the last incomplete byte. This is safe (prefix-free) because the number of used bits is a (statically known) constant. We use this to ensure the number of used bits is known statically.
	
Types:
- missingness
  - treat as a type constructor `optional<T>`, i.e. base types don't encode missingness. Emits a single bit in the encoding. Can invert this bit to control whether missing values come first or last in the ordering. If missing, nothing is emitted after.
- sort-order
  - treat reversing the default ordering as a type constructor `reverse<T>`
  - simply inverts the encoding bitwise
- primitive types
  - same as in datafusion, encoding has same size as original type
  - signed integers - flip the sign bit
  - floating point numbers - if sign bit is set, invert all bits, otherwise only flip the sign bit
- arrays
  - before each element and after last element, emit continuation bit (0 if no more elements)
  - pad before each element. This prevents a variable number of missing bits packing into a byte
- strings and byte-arrays
  - simply use null-terminated strings (being careful to do this in a unicode-safe way)
- structs
  - simply concatenate element encodings. safe because codes are prefix-free
- key structs
  - support variable length "interval endpoints"
    - e.g. for a key type `struct<t1, t2>`, the interval `[{a}, {a, b})` contains all keys with first field `a` and second field less than `b`. We break it into two "interval endpoints", `({a}, -1)` and `({a, b}, -1)`, which consist of a struct value which is a prefix of the key struct type, and a "sign". In this case, both endpoints "lean left".
    - needed for working with partitioners at runtime
  - like an array with fixed but heterogenous types and a max length
  - before each element and after last element, emit two continuation bits
    - `00` - end of key, leans left (less than all longer keys with this prefix)
    - `01` - continue, or after last key field of actual key value (not interval endpoint)
      - unambiguous because key value can't terminate early, and can't continue past last key field (max length)
    - `11` - end of key, leans right (greater than all longer keys with this prefix)
  - after each element, pad

# Implementation sketch
Concretely, we define a function `emit(value: SValue, trailingBits: Int)` which appends the encoding of `value` to an output buffer using the interface `emitByte(b: Value[Byte])` (along with `emitInt`, etc), which can only be called with 0 trailing bits, `emitBit(b: Value[Boolean], trailingBits: Int)`, and `pad()`. Note that this interface forces the caller to track the statically known number of trailing bits.

Illustrated using scala-like psuedocode. I pretend conditional and loop statements are virtualized to work on staged values, in which case they generate runtime code. This is straightforward to translate to our `CodeBuilder` interface.

```scala
// top level method, emit a required value of any type
// always leaves 0 trailing bits
emit(value: SValue, trailingBits: Int) {
  // dispatch on type
}

// leaves `trailingBits + 1` trailing bits
emitMissingness(present: Value[Boolean], trailingBits: Int) {
  if (missingFirst) emitBit(present, trailingBits)
  else emitBit(~present, trailingBits)
}

// leaves 0 trailing bits
emitInt(value: Value[Int], _: Int) {
  pad()
  // flip the sign bit
  emitInt(value ^ (1 << 31))
}

// leaves 0 trailing bits
emitArray(value: SIndexable, _: Int) {
  // for simplicity, don't try to pack continuation and missing
  // bits of first element with preceding bits
  pad()
  for ((i, elt) <- value.enumerate) { // runtime loop
    emitBit(true, 0) // continuation bit
    if (value.eltsRequired) { // static if
      emit(elt, 1)
    } else {
      emitMissingness(elt.present, 1)
      emit(elt.get, 2)
    }
  }
  emitBit(false, 0) // continuation bit
  pad() 
}

// leaves 0 trailing bits
emitStruct(value: SBaseStruct, _trailingBits: Int) {
  var trailingBits = _trailingBits
  for (field <- value) { // static loop
    if (field.required) { // static if
      emit(field, trailingBits)
    } else {
      emitMissingness(field.present, trailingBits)
      emit(field, trailingBits + 1)
    }
    trailingBits = 0
  }
}

emitKeyStruct(value: SBaseStruct, length: Value[Int], sign: Value[Int], _trailingBits: Int) {
  var trailingBits = _trailingBits
  for ((field, i) <- value.enumerate) { // static loop
    if (i < length) { // runtime if
      emitBit(false, trailingBits)
      emitBit(true, trailingBits + 1) // continuation bits
      if (field.required) { // static if
        emit(field, trailingBits + 2)
      } else {
        emitMissingness(field.present, trailingBits + 2)
        emit(field, trailingBits + 3)
      }
    } else if (i == length) { // terminating continuation bits
      if (sign < 0) {
        emitBit(false, trailingBits)
        emitBit(false, trailingBits + 1)
      } else if (sign > 0) {
        emitBit(true, trailingBits)
        emitBit(true, trailingBits + 1)
      } else {
        assert(length == value.numFields)
        emitBit(false, trailingBits)
        emitBit(true, trailingBits + 1) // continuation bits
      }
    } else {
      // past end of value, do nothing
    }
    trailingBits = 0
  }
  pad()
}
```

# Examples

Each example table is in ascending order.

`tarray<+tint>`:

| value      | padding   | cont | elt       | padding   | cont | elt       | padding   | cont |
| ---------- | --------- | ---- | --------- | --------- | ---- | --------- | --------- | ---- |
| `[]`       | `0000000` | `0`  |           |           |      |           |           |      |
| `[i1]`     | `0000000` | `1`  | `enc(i1)` | `0000000` | `0`  |           |           |      |
| `[i1, i2]` | `0000000` | `1`  | `enc(i1)` | `0000000` | `1`  | `enc(i2)` | `0000000` | `0`  |

`tarray<tint>`:

| value      | padding  | cont | missing | elt       | padding  | cont | missing | elt       | padding  | cont |
| ---------- | -------- | ---- | ------- | --------- | -------- | ---- | ------- | --------- | -------- | ---- |
| `[]`       | `000000` | `0`  |         |           |          |      |         |           |          |      |
| `[i1]`     | `000000` | `1`  | `0`     | `enc(i1)` | `000000` | `0`  |         |           |          |      |
| `[na]`     | `000000` | `1`  | `1`     |           | `000000` | `0`  |         |           |          |      |
| `[i1, i2]` | `000000` | `1`  | `0`     | `enc(i1)` | `000000` | `1`  | `0`     | `enc(i2)` | `000000` | `0`  |
| `[i1, na]` | `000000` | `1`  | `0`     | `enc(i1)` | `000000` | `1`  | `1`     |           | `000000` | `0`  |
| `[na, na]` | `000000` | `1`  | `1`     |           | `000000` | `1`  | `1`     |           | `000000` | `0`  |

`ttuple<tint, tarray<tint>>`

| value      | padding  | missing | missing | field1    | padding   | missing | padding  | cont |
| ---------- | -------- | ------- | ------- | --------- | --------- | ------- | -------- | ---- |
| `{i1, []}` | `000000` | `0`     | `0`     | `enc(i1)` | `0000000` | `0`     | `000000` | `0`  |
| `{i1, na}` | `000000` | `0`     | `0`     | `enc(i1)` | `0000000` | `1`     |          |      |
| `{na, []}` | `000000` | `0`     | `1`     |           | `0000000` | `0`     | `000000` | `0`  |
| `na`       | `000000` | `1`     |         |           |           |         |          |      |
