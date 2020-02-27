cc @tpoterba . Here because as we discovered, dev forum has a relatively short editing window. Will post to dev forum when "complete" and ready for broader discussion

# What are Physical Types?

Physical types are the classes that manage in-memory representations of Hail Types (Virtual Types), for both staged and unstaged code.

# Motivation:

- Improve performance by building specialized memory representations for data
- Make it easier for developers to work with in memory representations of Hail types

# Project technical goals:

- Remove requiredness from virtual types
- Implement at least one non-canonical physical type

# Relation to regions

The methods that take regions are those that construct a new in-memory representation (are either `def allocate` or convenience methods that wrap `allocate` and may perform some complex operations before calling `allocate`, e.g `copyFromType`). Allocated addresses may be read using static Region methods (e.g `Region.loadAddress`), because they are absolute memory addresses rather than relative to some region offset.

Long-term, methods besides `allocate` and wrapping methods, which need to allocate (for instance lazy-loading BGEN data) will be given the ability to do so without taking region as an argument (values will be associated with the regions that allocated them). Namely, regions may be placed on the values that own them.

# Physical Type organization

## Constructible types

Every PType has a "fundamentalType", which is the is the constructible representation for that type. It is, by default equal to the PType itself, but this may not always be the case (e.g [ComplexPType](#complex-ptypes))

## Collection PTypes

[PArray](#parray)

- Concrete implementations (canonical/non)

[PSet](#pset)

- Concrete implementations (canonical/non)

[PDict](#pdict)

- Concrete implementations (canonical/non)

[PNDArray](#pndict)

- Concrete implementations (canonical/non)

[PStruct](#pstruct)

- Concrete implementations (canonical/non)

[PTuple](#ptuple)

- Concrete implementations (canonical/non)

PString

- Concrete implementations (canonical/non)

PBinary

- Concrete implementations (canonical/non)

## <a name="complex-ptypes"></a> Complex PTypes

A nested physical type, that may be represented by physical types that the class does not explicitly take as an argument. In other words, the class' underlying physical type representation is different than any physical type arguments passed to it

PLocus

- Concrete implementations (canonical/non)

PCall

- Concrete implementations (canonical/non)

PInterval

- Concrete implementations (canonical/non)

## Primitive Types

While long-term these may have canonical and non-canonical types, that is outside the scope of this design document. When non-canonical primitives are introduced they will follow the strucutre outlined for non-primitive types.

PFloat32

- Represents a 4 byte float

PFloat64

- Represents an 8 byte float

PInt32

- Represents a 4 byte integer

PInt64

- Represents an 8 byte integer

PVoid

<br/>

# Common methods

```scala
  def constructAtAddress(mb: MethodBuilder, addr: Code[Long], region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Unit]
  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Unit
```

- Constructs a new value at `addr`, from `srcAddrss`
- Performs a deep copy when `srcPType != this`, or when `forceDeep == true`

```scala
def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] = ...
def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long = ...
```

- Allocates a new address and calls constructAtAddress
- For operations that can be shallow, returns srcAddress, skipping construction

# <a name="parray"></a> PArray

An abstract class for immutable ordered collections where all elements are of a single type. Does not contain the value constructor (e.g allocate)

## Core Methods

```scala
def allocate(region: Region, length: Int): Long = ...
def allocate(region: Code[Region], length: Code[Int]): Code[Long] = ...
```

- Allocate the memory needed for an array of `length` length. Cannot exceed 2^31 entries

```scala
def initialize(aoff: Long, length: Int, setMissing: Boolean = false) = ...
def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Code[Unit] = ...
```

- Initialize an allocated array by setting its elements to present or missing.

```scala
def isElementMissing(arrayAddress: Long, elementIndex: Int): Boolean= ...
def isElementMissing(arrayAddress: Long, elementIndex: Code[Int]): Code[Boolean] = ...
```

- Does the element at the given index exist

```scala
def loadLength(arrayAddress: Long): Int = ...
def loadLength(arrayAddress: Code[Long]): Code[Int] = ...
```

- Gets the array length, will not exceed 2^31

```scala
def loadElement(arrayAddress: Long, elementIndex: Int): Long = ...
def loadElement(arrayAddress: Code[Long], elementIndex: Code[Int]): Code[Long] = ...
```

- Gets the address of the element at the given index.
  - For pointer types loads the address at the offset into arrayAddress, otherwise returns that address

## <a name="parray"></a> PCanonicalArray

A growable array that is accessed by a pointer.

### Structure

Starting at `arrayAddress`:

[`4-byte length`, `n/8 byte missigness data`, `n * elementByteSize byte element data`]

# <a name="parray"></a> PSet

An abstract class for immutable ordered collections where all elements are unique

## Core Methods

```scala
def arrayFundamentalType: PArray
```

- The underlying array representation

## <a name="parray"></a> PCanonicalSet

A PCanonicalArray-backed implementation of PSet

# <a name="parray"></a> PDict

An abstract class for immutable unordered collections of key-value pairs. All keys must have one PType, and all values must have one (possibly different from keys) PType.

## Core Methods

```scala
def elementType: PStruct
```

- The PStruct representation of the key/value pair

```scala
def arrayFundamentalType: PArray
```

- The underlying array representation

## <a name="parray"></a> PCanonicalDict

A PCanonicalArray-backed implementation of PDict

# <a name="parray"></a> PNDArray

An abstract class for multidimensional arrays (tensors) that have a row-major or column-major layout.

## Core Methods

```scala
val shape: StaticallyKnownField[PTuple, Long]
val strides: StaticallyKnownField[PTuple, Long]
```

- Defines the tensor shape

```scala
def loadElementToIRIntermediate(indices: Array[Code[Long]], ndAddress: Code[Long], mb: MethodBuilder): Code[_]
```

- Load the element's primitive representation, as indexed by `indices`, which specifies the element index at every dimension in the PNDArray's shape

```scala
  def linearizeIndicesRowMajor(indices: Array[Code[Long]], shapeArray: Array[Code[Long]], mb: MethodBuilder): Code[Long]
```

- Get the off-heap index of the element (since NDArray elements are stored as a 1D series of bytes off-heap)

```scala
def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: Array[Code[Long]], mb: MethodBuilder): (Code[Unit], Array[Code[Long]])
```

- Generate the index path that represents the virtual, shape-dependent index into an arbitrary tensor

```scala
  def copyRowMajorToColumnMajor(rowMajorAddress: Code[Long], targetAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], mb: MethodBuilder): Code[Unit]

  def copyColumnMajorToRowMajor(colMajorAddress: Code[Long], targetAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], mb: MethodBuilder): Code[Unit]
```

- Interconvert between column and row major

```scala
  def construct(flags: Code[Int], offset: Code[Int], shapeBuilder: (StagedRegionValueBuilder => Code[Unit]),
    stridesBuilder: (StagedRegionValueBuilder => Code[Unit]), data: Code[Long], mb: MethodBuilder): Code[Long]
}
```

- Construct the NDArray off-heap

```scala
def arrayFundamentalType: PArray
```

- The underlying array representation

## <a name="parray"></a> PCanonicalNDArray

A PCanonicalArray-backed NDArray

# <a name="pstruct">PStruct</a>

An immutable, collection of unordered values, whose elements may be of different types.

## Core methods

```scala
def allocate(region: Region): Long
def allocate(region: Code[Region]): Code[Long]
```

- Allocate enough memory off-heap to store the requested elements

```scala
  def initialize(address: Long, setMissing: Boolean = false): Unit
  def stagedInitialize(address: Code[Long], setMissing: Boolean = false): Code[Unit]
```

- Set element missingness and store element length

```scala
def isFieldMissing(address: Long, fieldIdx: Int): Boolean
def isFieldMissing(address: Code[Long], fieldIdx: Code[Int]): Boolean
```

- Verify whether field is missing

```scala
def setFieldMissing(address: Long, fieldIdx: Int): Unit
def setFieldMissing(address: Code[Long], fieldIdx: Int): Code[Unit]

def setFieldPresent(address: Long, fieldIdx: Int): Unit
def setFieldPresent(address: Code[Long], fieldIdx: Int): Code[Unit]
```

- Set field present of missing at a given memory address

```scala
def loadField(address: Long, fieldIdx: Int): Long
def loadField(address: Code[Long], fieldIdx: Int): Code[Long]
```

- Load field at a given memory address

```scala
def storeField(address: Long, fieldIdx: Int): Long
def storeField(address: Code[Long], fieldIdx: Int): Code[Long]
```

- Store field at a given memory address
- (This does not exist yet, but should I believe)

## <a name="ptuple">PCanonicalStruct</a>

An immutable, fixed-length collection of unordered values (of possibly different types). Number of elements known statically.

# <a name="ptuple">PTuple</a>

An immutable, collection of ordered values, whose elements may be of different types.

## Core methods

```scala
val _types: IndexedSeq[PTupleField]
```

- The ordered representation of physical types that represent this collection

```scala
def allocate(region: Region): Long
def allocate(region: Code[Region]): Code[Long]
```

- Allocate enough memory off-heap to store the requested elements

```scala
  def initialize(address: Long, setMissing: Boolean = false): Unit
  def stagedInitialize(address: Code[Long], setMissing: Boolean = false): Code[Unit]
```

- Set element missingness and store element length

```scala
def isFieldDefined(address: Long, fieldIdx: Int): Boolean
def isFieldDefined(address: Code[Long], fieldIdx: Code[Int]): Boolean
```

```scala
def setFieldMissing(address: Long, fieldIdx: Int): Unit
def setFieldMissing(address: Code[Long], fieldIdx: Int): Code[Unit]

def setFieldPresent(address: Long, fieldIdx: Int): Unit
def setFieldPresent(address: Code[Long], fieldIdx: Int): Code[Unit]
```

- Set field present of missing at a given memory address

```scala
def loadField(address: Long, fieldIdx: Int): Long
def loadField(address: Code[Long], fieldIdx: Int): Code[Long]
```

- Load field at a given memory address

```scala
def storeField(address: Long, fieldIdx: Int): Long
def storeField(address: Code[Long], fieldIdx: Int): Code[Long]
```

- Store field at a given memory address
- (This does not exist yet, but should I believe)

## <a name="ptuple">PCanonicalTuple</a>

An immutable, fixed-length collection of ordered values (of possibly different types).

Number of elements known statically, and elements stored inline, just like PCanonicalStruct.
