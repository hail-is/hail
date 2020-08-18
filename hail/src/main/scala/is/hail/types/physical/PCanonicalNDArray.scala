package is.hail.types.physical

import is.hail.annotations.{Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{Code, MethodBuilder, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
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

  def loadShape(off: Code[Long], idx: Int): Code[Long] =
    shape.pType.types(idx).load(shape.pType.fieldOffset(shape.load(off), idx)).tcode[Long]

  def loadStride(off: Code[Long], idx: Int): Code[Long] =
    strides.pType.types(idx).load(strides.pType.fieldOffset(strides.load(off), idx)).tcode[Long]

  @transient lazy val strides = new StaticallyKnownField(
    PCanonicalTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*): PTuple,
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

  def numElements(shape: IndexedSeq[Code[Long]], mb: EmitMethodBuilder[_]): Code[Long] = {
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

  def loadElementToIRIntermediate(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], mb: EmitMethodBuilder[_]): Code[_] = {
    Region.loadIRIntermediate(data.pType.elementType)(getElementAddress(indices, ndAddress, mb))
  }

  def outOfBounds(indices: IndexedSeq[Value[Long]], nd: Value[Long], mb: EmitMethodBuilder[_]): Code[Boolean] = {
    val shapeTuple = new CodePTuple(shape.pType, new Value[Long] {
      def get: Code[Long] = shape.load(nd)
    })
    val outOfBounds = mb.genFieldThisRef[Boolean]()
    Code(
      outOfBounds := false,
      Code.foreach(0 until nDims) { dimIndex =>
        outOfBounds := outOfBounds || (indices(dimIndex) >= shapeTuple(dimIndex))
      },
      outOfBounds
    )
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
  ): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(mb, this.representation, region)

    Code(Code(FastIndexedSeq(
      srvb.start(),
      srvb.addBaseStruct(this.shape.pType, shapeBuilder),
      srvb.advance(),
      srvb.addBaseStruct(this.strides.pType, stridesBuilder),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("data"))(data))),
      srvb.end()
    )
  }

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    val sourceNDPType = srcPType.asInstanceOf[PNDArray]

    assert(this.elementType == sourceNDPType.elementType && this.nDims == sourceNDPType.nDims)

    this.representation.copyFromType(mb, region, sourceNDPType.representation, srcAddress, deepCopy)
  }

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], deepCopy)

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long  = {
    val sourceNDPType = srcPType.asInstanceOf[PNDArray]
    assert(elementType == sourceNDPType.elementType && nDims == sourceNDPType.nDims)
    representation.copyFromAddress(region, sourceNDPType.representation, srcAddress, deepCopy)
  }

  override def deepRename(t: Type) = deepRenameNDArray(t.asInstanceOf[TNDArray])

  private def deepRenameNDArray(t: TNDArray) =
    PCanonicalNDArray(this.elementType.deepRename(t.elementType), this.nDims, this.required)

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalNDArray(elementType, nDims, required)

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] =
    this.fundamentalType.constructAtAddress(mb, addr, region, srcPType.fundamentalType, srcAddress, deepCopy)

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    this.fundamentalType.constructAtAddress(addr, region, srcPType.fundamentalType, srcAddress, deepCopy)
}

object PCanonicalNDArraySettable {
  def apply(cb: EmitCodeBuilder, pt: PCanonicalNDArray, name: String, sb: SettableBuilder): PCanonicalNDArraySettable = {
    new PCanonicalNDArraySettable(pt, sb.newSettable(name))
  }
}

class PCanonicalNDArraySettable(override val pt: PCanonicalNDArray, val a: Settable[Long]) extends PNDArrayValue with PSettable {
  //FIXME: Rewrite apply to not require a methodBuilder, meaning also rewrite loadElementToIRIntermediate
  def apply(indices: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Value[_] = {
    assert(indices.size == pt.nDims)
    new Value[Any] {
      override def get: Code[Any] = pt.loadElementToIRIntermediate(indices, a, mb)
    }
  }

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  override def get: PCode = new PCanonicalNDArrayCode(pt, a)

  override def store(pv: PCode): Code[Unit] = a := pv.asInstanceOf[PCanonicalNDArrayCode].a

  override def outOfBounds(indices: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Boolean] = {
    pt.outOfBounds(indices, a, mb)
  }

  override def shapes(): IndexedSeq[Value[Long]] = Array.tabulate(pt.nDims) { i =>
    new Value[Long] {
      def get: Code[Long] = pt.loadShape(a, i)
    }
  }

  override def sameShape(other: PNDArrayValue, mb: EmitMethodBuilder[_]): Code[Boolean] = {
    val comparator = this.pt.shape.pType.codeOrdering(mb, other.pt.shape.pType)
    val thisShape = this.pt.shape.load(this.a).asInstanceOf[Code[comparator.T]]
    val otherShape = other.pt.shape.load(other.value.asInstanceOf[Value[Long]]).asInstanceOf[Code[comparator.T]]
    comparator.equivNonnull(thisShape, otherShape)
  }
}

class PCanonicalNDArrayCode(val pt: PCanonicalNDArray, val a: Code[Long]) extends PNDArrayCode {

  override def code: Code[_] = a

  override def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  override def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = ???

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PNDArrayValue = {
    val s = PCanonicalNDArraySettable(cb, pt, name, sb)
    cb.assign(s, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): PNDArrayValue = memoize(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): PValue = memoize(cb, name, cb.fieldBuilder)
}
