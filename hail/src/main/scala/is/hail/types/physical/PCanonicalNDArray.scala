package is.hail.types.physical

import is.hail.annotations.{Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SNDArrayPointer, SNDArrayPointerCode}
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.utils.FastIndexedSeq

final case class PCanonicalNDArray(elementType: PType, nDims: Int, required: Boolean = false) extends PNDArray  {
  assert(elementType.required, "elementType must be required")

  def _asIdent: String = s"ndarray_of_${elementType.asIdent}"

  override def containsPointers: Boolean = true

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCNDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append(s",$nDims]")
  }

  @transient lazy val shape = new StaticallyKnownField(
    PCanonicalTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*): PTuple,
    off => representation.loadField(off, "shape")
  )

  def loadShape(cb: EmitCodeBuilder, off: Code[Long], idx: Int): Code[Long] =
    shape.pType.types(idx).loadCheapPCode(cb, shape.pType.fieldOffset(shape.load(off), idx)).asInt64.longCode(cb)

  def loadStride(cb: EmitCodeBuilder, off: Code[Long], idx: Int): Code[Long] =
    strides.pType.types(idx).loadCheapPCode(cb, strides.pType.fieldOffset(strides.load(off), idx)).asInt64.longCode(cb)

  @transient lazy val strides = new StaticallyKnownField(
    PCanonicalTuple(true, Array.tabulate(nDims)(_ => PInt64Required): _*): PTuple,
    (off) => representation.loadField(off, "strides")
  )

  @transient lazy val data: StaticallyKnownField[PArray, Long] = new StaticallyKnownField(
    PCanonicalArray(elementType, required = true),
    off => representation.loadField(off, "data")
  )

  lazy val representation: PStruct = {
    PCanonicalStruct(required,
      ("shape", shape.pType),
      ("strides", strides.pType),
      ("data", data.pType))
  }

  override lazy val byteSize: Long = representation.byteSize

  override lazy val alignment: Long = representation.alignment

  override def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override lazy val fundamentalType: PType = representation.fundamentalType

  override lazy val encodableType: PType = PCanonicalNDArray(elementType.encodableType, nDims, required)

  def numElements(shape: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long] = {
    shape.foldLeft(1L: Code[Long])(_ * _)
  }

  def makeShapeBuilder(shapeArray: IndexedSeq[Value[Long]]): StagedRegionValueBuilder => Code[Unit] = { srvb =>
    coerce[Unit](Code(
      srvb.start(),
      Code(shapeArray.map(shapeElement => Code(
        srvb.addLong(shapeElement),
        srvb.advance()
      )))
    ))
  }

  def makeColumnMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): StagedRegionValueBuilder => Code[Unit] = { srvb =>
    val runningProduct = mb.newLocal[Long]()
    val tempShapeStorage = mb.newLocal[Long]()
    Code(
      srvb.start(),
      runningProduct := elementType.byteSize,
      Code.foreach(0 until nDims){ index =>
        Code(
          srvb.addLong(runningProduct),
          srvb.advance(),
          tempShapeStorage := sourceShapeArray(index),
          runningProduct := runningProduct * (tempShapeStorage > 0L).mux(tempShapeStorage, 1L)
        )
      }
    )
  }

  def makeRowMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): StagedRegionValueBuilder => Code[Unit] = { srvb =>
    val runningProduct = mb.newLocal[Long]()
    val tempShapeStorage = mb.newLocal[Long]()
    val computedStrides = (0 until nDims).map(_ => mb.genFieldThisRef[Long]())
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

  private def getElementAddress(indices: IndexedSeq[Value[Long]], nd: Value[Long], mb: EmitMethodBuilder[_]): Code[Long] = {
    val stridesTuple  = new CodePTuple(strides.pType, new Value[Long] {
      def get: Code[Long] = strides.load(nd)
    })
    val bytesAway = mb.newLocal[Long]()
    val dataStore = mb.newLocal[Long]()

    coerce[Long](Code(
      dataStore := data.load(nd),
      bytesAway := 0L,
      indices.zipWithIndex.foldLeft(Code._empty) { case (codeSoFar: Code[_], (requestedIndex: Value[Long], strideIndex: Int)) =>
        Code(
          codeSoFar,
          bytesAway := bytesAway + requestedIndex * stridesTuple(strideIndex))
      },
      bytesAway + data.pType.elementOffset(dataStore, data.pType.loadLength(dataStore), 0)
    ))
  }

  def setElement(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: Code[_], mb: EmitMethodBuilder[_]): Code[Unit] = {
    Region.storeIRIntermediate(this.elementType)(getElementAddress(indices, ndAddress, mb), newElement)
  }

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long]): Code[Long] = {
    val off = getElementAddress(indices, ndAddress, cb.emb)
    data.pType.elementType.fundamentalType match {
      case _: PArray | _: PBinary =>
        Region.loadAddress(off)
      case _ =>
        off
    }
  }

  def loadElementToIRIntermediate(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], mb: EmitMethodBuilder[_]): Code[_] = {

    Region.loadIRIntermediate(data.pType.elementType)(getElementAddress(indices, ndAddress, mb))
  }

  def linearizeIndicesRowMajor(indices: IndexedSeq[Code[Long]], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long] = {
    val index = mb.genFieldThisRef[Long]()
    val elementsInProcessedDimensions = mb.genFieldThisRef[Long]()
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

  def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): (Code[Unit], IndexedSeq[Value[Long]]) = {
    val nDim = shapeArray.length
    val newIndices = (0 until nDim).map(_ => mb.genFieldThisRef[Long]())
    val elementsInProcessedDimensions = mb.genFieldThisRef[Long]()
    val workRemaining = mb.genFieldThisRef[Long]()

    val createShape = Code(
      workRemaining := index,
      elementsInProcessedDimensions := shapeArray.foldLeft(1L: Code[Long])(_ * _),
      Code.foreach(shapeArray.zip(newIndices)) { case (shapeElement, newIndex) =>
        Code(
          elementsInProcessedDimensions := elementsInProcessedDimensions / shapeElement,
          newIndex := workRemaining / elementsInProcessedDimensions,
          workRemaining := workRemaining % elementsInProcessedDimensions
        )
      }
    )
    (createShape, newIndices)
  }

  override def construct(
    shapeBuilder: StagedRegionValueBuilder => Code[Unit],
    stridesBuilder: StagedRegionValueBuilder => Code[Unit],
    data: Code[Long],
    mb: EmitMethodBuilder[_],
    region: Value[Region]
  ): SNDArrayPointerCode = {
    val srvb = new StagedRegionValueBuilder(mb, this.representation, region)

    new SNDArrayPointerCode(SNDArrayPointer(this), Code(Code(FastIndexedSeq(
      srvb.start(),
      srvb.addBaseStruct(this.shape.pType, shapeBuilder),
      srvb.advance(),
      srvb.addBaseStruct(this.strides.pType, stridesBuilder),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("data"))(data))),
      srvb.end()
    ))
  }

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long  = {
    val sourceNDPType = srcPType.asInstanceOf[PNDArray]
    assert(elementType == sourceNDPType.elementType && nDims == sourceNDPType.nDims)
    representation.copyFromAddress(region, sourceNDPType.representation, srcAddress, deepCopy)
  }

  override def deepRename(t: Type) = deepRenameNDArray(t.asInstanceOf[TNDArray])

  private def deepRenameNDArray(t: TNDArray) =
    PCanonicalNDArray(this.elementType.deepRename(t.elementType), this.nDims, this.required)

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalNDArray(elementType, nDims, required)

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    this.fundamentalType.unstagedStoreAtAddress(addr, region, srcPType.fundamentalType, srcAddress, deepCopy)

  def sType: SNDArrayPointer = SNDArrayPointer(this)

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SNDArrayPointerCode(sType, addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SNDArrayPointer(t) if t.equalModuloRequired(this) =>
          representation.store(cb, region, representation.loadCheapPCode(cb, value.asInstanceOf[SNDArrayPointerCode].a), deepCopy)
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    value.st match {
      case SNDArrayPointer(t) if t.equalModuloRequired(this) =>
        representation.storeAtAddress(cb, addr, region, representation.loadCheapPCode(cb, value.asInstanceOf[SNDArrayPointerCode].a), deepCopy)
    }
  }
}

