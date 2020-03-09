package is.hail.expr.types.physical

import is.hail.annotations.{Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{Code, MethodBuilder, _}
import is.hail.expr.types.virtual.{TNDArray, Type}
import is.hail.utils.FastIndexedSeq

final case class PCanonicalNDArray(elementType: PType, nDims: Int, required: Boolean = false) extends PNDArray  {
  assert(elementType.required, "elementType must be required")

  def _asIdent: String = s"ndarray_of_${elementType.asIdent}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCNDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append(s",$nDims]")
  }

  @transient lazy val flags = new StaticallyKnownField(PInt32Required, off => Region.loadInt(representation.loadField(off, "flags")))
  @transient lazy val offset = new StaticallyKnownField(
    PInt32Required,
    off => Region.loadInt(representation.loadField(off, "offset"))
  )
  @transient lazy val shape = new StaticallyKnownField(
    PCanonicalTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*): PTuple,
    off => representation.loadField(off, "shape")
  )
  @transient lazy val strides = new StaticallyKnownField(
    PCanonicalTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*): PTuple,
    (off) => representation.loadField(off, "strides")
  )

  @transient lazy val data: StaticallyKnownField[PArray, Long] = new StaticallyKnownField(
    PArray(elementType, required = true),
    off => representation.loadField(off, "data")
  )

  lazy val representation: PStruct = {
    PStruct(required,
      ("flags", flags.pType),
      ("offset", offset.pType),
      ("shape", shape.pType),
      ("strides", strides.pType),
      ("data", data.pType))
  }

  override lazy val byteSize: Long = representation.byteSize

  override lazy val alignment: Long = representation.alignment

  override def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override lazy val fundamentalType: PType = representation.fundamentalType

  def numElements(shape: IndexedSeq[Code[Long]], mb: MethodBuilder): Code[Long] = {
    shape.foldLeft(1L: Code[Long])(_ * _)
  }

  def makeShapeBuilder(shapeArray: IndexedSeq[Code[Long]]): StagedRegionValueBuilder => Code[Unit] = { srvb =>
    coerce[Unit](Code(
      srvb.start(),
      Code(shapeArray.map(shapeElement => Code(
        srvb.addLong(shapeElement),
        srvb.advance()
      )))
    ))
  }

  def makeDefaultStridesBuilder(sourceShapeArray: IndexedSeq[Code[Long]], mb: MethodBuilder): StagedRegionValueBuilder => Code[Unit] = { srvb =>
    val runningProduct = mb.newLocal[Long]
    val tempShapeStorage = mb.newLocal[Long]
    val computedStrides = (0 until nDims).map(_ => mb.newField[Long])
    Code(
      srvb.start(),
      runningProduct := elementType.byteSize,
      Code.foreach((nDims - 1) to 0 by -1){ index =>
        Code(
          computedStrides(index) := runningProduct,
          tempShapeStorage := sourceShapeArray(index),
          runningProduct := runningProduct * (tempShapeStorage > 0L).mux(tempShapeStorage, 1L)
        )
      },
      Code.foreach(0 until nDims)(index =>
        Code(
          srvb.addLong(computedStrides(index)),
          srvb.advance()
        )
      )
    )
  }

  private def getElementAddress(indices: IndexedSeq[Code[Long]], nd: Code[Long], mb: MethodBuilder): Code[Long] = {
    val stridesTuple  = new CodePTuple(strides.pType, strides.load(nd))
    val bytesAway = mb.newLocal[Long]
    val dataStore = mb.newLocal[Long]

    coerce[Long](Code(
      dataStore := data.load(nd),
      bytesAway := 0L,
      indices.zipWithIndex.foldLeft(Code._empty){case (codeSoFar: Code[_], (requestedIndex: Code[Long], strideIndex: Int)) =>
        Code(
          codeSoFar,
          bytesAway := bytesAway + requestedIndex * stridesTuple(strideIndex))
      },
      bytesAway + data.pType.elementOffset(dataStore, data.pType.loadLength(dataStore), 0)
    ))
  }

  def loadElementToIRIntermediate(indices: IndexedSeq[Code[Long]], ndAddress: Code[Long], mb: MethodBuilder): Code[_] = {
    Region.loadIRIntermediate(data.pType.elementType)(this.getElementAddress(indices, ndAddress, mb))
  }

  def outOfBounds(indices: IndexedSeq[Code[Long]], nd: Code[Long], mb: MethodBuilder): Code[Boolean] = {
    val shapeTuple = new CodePTuple(shape.pType, shape.load(nd))
    val outOfBounds = mb.newField[Boolean]
    Code(
      outOfBounds := false,
      Code.foreach(0 until nDims) { dimIndex =>
        outOfBounds := outOfBounds || (indices(dimIndex) >= shapeTuple(dimIndex))
      },
      outOfBounds
    )
  }

  def linearizeIndicesRowMajor(indices: IndexedSeq[Code[Long]], shapeArray: IndexedSeq[Code[Long]], mb: MethodBuilder): Code[Long] = {
    val index = mb.newField[Long]
    val elementsInProcessedDimensions = mb.newField[Long]
    Code(
      index := 0L,
      elementsInProcessedDimensions := 1L,
      Code.foreach(shapeArray.zip(indices).reverse) { case (shapeElement, currentIndex) =>
        Code(
          index := index + currentIndex * elementsInProcessedDimensions,
          elementsInProcessedDimensions := elementsInProcessedDimensions * shapeElement
        )
      },
      index
    )
  }

  def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: IndexedSeq[Code[Long]], mb: MethodBuilder): (Code[Unit], IndexedSeq[Code[Long]]) = {
    val nDim = shapeArray.length
    val newIndices = (0 until nDim).map(_ => mb.newField[Long])
    val elementsInProcessedDimensions = mb.newField[Long]
    val workRemaining = mb.newField[Long]

    val createShape = Code(
      workRemaining := index,
      elementsInProcessedDimensions := shapeArray.fold(1L: Code[Long])(_ * _),
      Code.foreach(shapeArray.zip(newIndices)) { case (shapeElement, newIndex) =>
        Code(
          elementsInProcessedDimensions := elementsInProcessedDimensions / shapeElement,
          newIndex := workRemaining / elementsInProcessedDimensions,
          workRemaining := workRemaining % elementsInProcessedDimensions
        )
      }
    )
    (createShape, newIndices.map(_.load()))
  }

  def copyRowMajorToColumnMajor(rowMajorAddress: Code[Long], targetAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], mb: MethodBuilder): Code[Unit] = {
    val rowIndex = mb.newField[Long]
    val colIndex = mb.newField[Long]
    val rowMajorCoord = nCols * rowIndex + colIndex
    val colMajorCoord = nRows * colIndex + rowIndex
    val rowMajorFirstElementAddress = this.data.pType.firstElementOffset(rowMajorAddress, (nRows * nCols).toI)
    val targetFirstElementAddress = this.data.pType.firstElementOffset(targetAddress, (nRows * nCols).toI)
    val currentElement = Region.loadDouble(rowMajorFirstElementAddress + rowMajorCoord * 8L)

    Code.forLoop(rowIndex := 0L, rowIndex < nRows, rowIndex := rowIndex + 1L,
      Code.forLoop(colIndex := 0L, colIndex < nCols, colIndex := colIndex + 1L,
        Region.storeDouble(targetFirstElementAddress + colMajorCoord * 8L, currentElement)
      )
    )
  }

  def copyColumnMajorToRowMajor(colMajorAddress: Code[Long], targetAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], mb: MethodBuilder): Code[Unit] = {
    val rowIndex = mb.newField[Long]
    val colIndex = mb.newField[Long]
    val rowMajorCoord = nCols * rowIndex + colIndex
    val colMajorCoord = nRows * colIndex + rowIndex
    val colMajorFirstElementAddress = this.data.pType.firstElementOffset(colMajorAddress, (nRows * nCols).toI)
    val targetFirstElementAddress = this.data.pType.firstElementOffset(targetAddress, (nRows * nCols).toI)
    val currentElement = Region.loadDouble(colMajorFirstElementAddress + colMajorCoord * 8L)

    Code.forLoop(rowIndex := 0L, rowIndex < nRows, rowIndex := rowIndex + 1L,
      Code.forLoop(colIndex := 0L, colIndex < nCols, colIndex := colIndex + 1L,
        Region.storeDouble(targetFirstElementAddress + rowMajorCoord * 8L, currentElement)
      )
    )
  }

  def construct(flags: Code[Int], offset: Code[Int], shapeBuilder: (StagedRegionValueBuilder => Code[Unit]),
    stridesBuilder: (StagedRegionValueBuilder => Code[Unit]), data: Code[Long], mb: MethodBuilder): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(mb, this.representation)

    coerce[Long](Code(FastIndexedSeq(
      srvb.start(),
      srvb.addInt(flags),
      srvb.advance(),
      srvb.addInt(offset),
      srvb.advance(),
      srvb.addBaseStruct(this.shape.pType, shapeBuilder),
      srvb.advance(),
      srvb.addBaseStruct(this.strides.pType, stridesBuilder),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("data"))(data)),
      srvb.end()
    ))
  }

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] = {
    val sourceNDPType = srcPType.asInstanceOf[PNDArray]

    assert(this.elementType == sourceNDPType.elementType && this.nDims == sourceNDPType.nDims)

    this.representation.copyFromType(mb, region, sourceNDPType.representation, srcAddress, forceDeep)
  }

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], forceDeep)

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long  = {
    val sourceNDPType = srcPType.asInstanceOf[PNDArray]

    assert(this.elementType == sourceNDPType.elementType && this.nDims == sourceNDPType.nDims)

    this.representation.copyFromType(region, sourceNDPType.representation, srcAddress, forceDeep)
  }

  override def deepRename(t: Type) = deepRenameNDArray(t.asInstanceOf[TNDArray])

  private def deepRenameNDArray(t: TNDArray) =
    PCanonicalNDArray(this.elementType.deepRename(t.elementType), this.nDims, this.required)

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalNDArray(elementType, nDims, required)

  def constructAtAddress(mb: MethodBuilder, addr: Code[Long], region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Unit] =
    throw new NotImplementedError("constructAtAddress should only be called on fundamental types; PCanonicalNDarray is not fundamental")

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Unit =
    throw new NotImplementedError("constructAtAddress should only be called on fundamental types; PCanonicalNDarray is not fundamental")
}
