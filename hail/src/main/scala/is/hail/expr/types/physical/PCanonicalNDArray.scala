package is.hail.expr.types.physical

import is.hail.annotations.{ Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{Code, MethodBuilder, _}

final case class PCanonicalNDArray(elementType: PType, nDims: Int, required: Boolean = false) extends PNDArray  {
  assert(elementType.required, "elementType must be required")

  def _asIdent: String = s"ndarray_of_${elementType.asIdent}"

  override def _toPretty = throw new NotImplementedError("Only _pretty should be called.")

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("NDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append(s",$nDims]")
  }

  @transient lazy val flags = new StaticallyKnownField(PInt32Required, (r, off) => Region.loadInt(representation.loadField(r, off, "flags")))
  @transient lazy val offset = new StaticallyKnownField(
    PInt32Required,
    (r, off) => Region.loadInt(representation.loadField(r, off, "offset"))
  )
  @transient lazy val shape = new StaticallyKnownField(
    PTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*),
    (r, off) => representation.loadField(r, off, "shape")
  )
  @transient lazy val strides = new StaticallyKnownField(
    PTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*),
    (r, off) => representation.loadField(r, off, "strides")
  )

  @transient lazy val data: StaticallyKnownField[PArray, Long] = new StaticallyKnownField(
    PArray(elementType, required = true),
    (r, off) => representation.loadField(r, off, "data")
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

  def numElements(shape: Array[Code[Long]], mb: MethodBuilder): Code[Long] = {
    shape.foldLeft(const(1L))(_ * _)
  }

  def makeShapeBuilder(shapeArray: Array[Code[Long]]): StagedRegionValueBuilder => Code[Unit] = { srvb =>
    coerce[Unit](Code(
      srvb.start(),
      Code(shapeArray.map(shapeElement => Code(
        srvb.addLong(shapeElement),
        srvb.advance()
      )):_*)
    ))
  }

  def makeDefaultStridesBuilder(sourceShapeArray: Array[Code[Long]], mb: MethodBuilder): StagedRegionValueBuilder => Code[Unit] = { srvb =>
    val runningProduct = mb.newLocal[Long]
    val tempShapeStorage = mb.newLocal[Long]
    val computedStrides = (0 until nDims).map(_ => mb.newField[Long]).toArray
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

  def getElementAddress(indices: Array[Code[Long]], nd: Code[Long], region: Code[Region], mb: MethodBuilder): Code[Long] = {
    val stridesTuple  = new CodePTuple(strides.pType, region, strides.load(region, nd))
    val bytesAway = mb.newLocal[Long]
    val dataStore = mb.newLocal[Long]

    coerce[Long](Code(
      dataStore := data.load(region, nd),
      bytesAway := 0L,
      indices.zipWithIndex.foldLeft(Code._empty[Unit]){case (codeSoFar: Code[_], (requestedIndex: Code[Long], strideIndex: Int)) =>
        Code(
          codeSoFar,
          bytesAway := bytesAway + requestedIndex * stridesTuple(strideIndex))
      },
      bytesAway + data.pType.elementOffset(dataStore, data.pType.loadLength(dataStore), 0)
    ))
  }

  def loadElementToIRIntermediate(indices: Array[Code[Long]], ndAddress: Code[Long], region: Code[Region], mb: MethodBuilder): Code[_] = {
    Region.loadIRIntermediate(this.elementType)(this.getElementAddress(indices, ndAddress, region, mb))
  }

  def outOfBounds(indices: Array[Code[Long]], nd: Code[Long], region: Code[Region], mb: MethodBuilder): Code[Boolean] = {
    val shapeTuple = new CodePTuple(shape.pType, region, shape.load(region, nd))
    val outOfBounds = mb.newField[Boolean]
    Code(
      outOfBounds := false,
      Code.foreach(0 until nDims) { dimIndex =>
        outOfBounds := outOfBounds || (indices(dimIndex) >= shapeTuple(dimIndex))
      },
      outOfBounds
    )
  }

  def linearizeIndicesRowMajor(indices: Array[Code[Long]], shapeArray: Array[Code[Long]], region: Code[Region], mb: MethodBuilder): Code[Long] = {
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

  def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: Array[Code[Long]], region: Code[Region], mb: MethodBuilder): (Code[Unit], Array[Code[Long]]) = {
    val nDim = shapeArray.length
    val newIndices = (0 until nDim).map(_ => mb.newField[Long]).toArray
    val elementsInProcessedDimensions = mb.newField[Long]
    val workRemaining = mb.newField[Long]

    val createShape = Code(
      workRemaining := index,
      elementsInProcessedDimensions := shapeArray.fold(const(1L))(_ * _),
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

    coerce[Long](Code(
      srvb.start(),
      srvb.addInt(flags),
      srvb.advance(),
      srvb.addInt(offset),
      srvb.advance(),
      srvb.addBaseStruct(this.shape.pType, shapeBuilder),
      srvb.advance(),
      srvb.addBaseStruct(this.strides.pType, stridesBuilder),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("data"))(data),
      srvb.end()
    ))
  }
}
