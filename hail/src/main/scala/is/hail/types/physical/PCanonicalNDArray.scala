package is.hail.types.physical

import is.hail.annotations.{Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerSettable, SNDArrayPointer, SNDArrayPointerCode}
import is.hail.types.physical.stypes.interfaces.SBaseStructCode
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

  def loadShape(off: Long, idx: Int): Long = {
    val shapeTupleAddr = representation.loadField(off, 0)
    Region.loadLong(shape.pType.loadField(shapeTupleAddr, idx))
  }

  def loadStride(off: Long, idx: Int): Long = {
    val shapeTupleAddr = representation.loadField(off, 1)
    Region.loadLong(strides.pType.loadField(shapeTupleAddr, idx))
  }


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

  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long] = {
    shape.foldLeft(1L: Code[Long])(_ * _)
  }

  def makeShapeStruct(shapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): SBaseStructCode = {
    val structAddress = cb.newLocal[Long]("make_shape_addr")
    cb.assign(structAddress,  this.shape.pType.allocate(region))


    (0 until nDims).foreach{ index =>
      cb.append(Region.storeLong(this.shape.pType.fieldOffset(structAddress, index), shapeArray(index)))
    }

    PCode.apply(this.shape.pType, structAddress).asBaseStruct
  }

  def makeColumnMajorStridesStruct(sourceShapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): SBaseStructCode = {
    val runningProduct = cb.newLocal[Long]("make_column_major_strides_prod")
    val structAddress = cb.newLocal[Long]("make_column_major_strides_addr")

    cb.assign(structAddress,  this.strides.pType.allocate(region))
    cb.assign(runningProduct, elementType.byteSize)
    (0 until nDims).foreach{ index =>
      cb.append(Region.storeLong(this.strides.pType.fieldOffset(structAddress, index), runningProduct))
      cb.assign(runningProduct, runningProduct * (sourceShapeArray(index) > 0L).mux(sourceShapeArray(index), 1L))
    }

    PCode.apply(this.strides.pType, structAddress).asBaseStruct
  }

  def makeRowMajorStridesStruct(sourceShapeArray: IndexedSeq[Value[Long]],region: Value[Region], cb: EmitCodeBuilder): SBaseStructCode = {
    val runningProduct = cb.newLocal[Long]("make_row_major_strides_prod")
    val computedStrides = (0 until nDims).map(idx => cb.newField[Long](s"make_row_major_computed_stride_${idx}"))
    val structAddress = cb.newLocal[Long]("make_row_major_strides_addr")

    cb.assign(structAddress,  this.strides.pType.allocate(region))
    cb.assign(runningProduct, elementType.byteSize)
    ((nDims - 1) to 0 by -1).foreach{ index =>
      cb.assign(computedStrides(index), runningProduct)
      cb.assign(runningProduct, runningProduct * (sourceShapeArray(index) > 0L).mux(sourceShapeArray(index), 1L))
    }
    (0 until nDims).foreach{ index =>
      cb.append(Region.storeLong(this.strides.pType.fieldOffset(structAddress, index), computedStrides(index)))
    }

    PCode.apply(this.strides.pType, structAddress).asBaseStruct
  }

  def getElementAddress(indices: IndexedSeq[Long], nd: Long): Long = {
    val dataLength = (0 until nDims).map(loadShape(nd, _)).foldLeft(1L)(_ * _)
    val dataAddress = this.representation.loadField(nd, 2)

    var bytesAway = 0L
    indices.zipWithIndex.foreach{case (requestedIndex: Long, strideIndex: Int) =>
      bytesAway += requestedIndex * loadStride(nd, strideIndex)
    }

    bytesAway + data.pType.firstElementOffset(dataAddress, dataLength.toInt)
  }

  private def getElementAddress(indices: IndexedSeq[Value[Long]], nd: Value[Long], cb: EmitCodeBuilder): Code[Long] = {
    val ndarrayValue = PCode(this, nd).asNDArray.memoize(cb, "getElementAddressNDValue")
    val stridesTuple = ndarrayValue.strides(cb)
    val bytesAway = cb.newLocal[Long]("nd_get_element_address_bytes_away")
    val dataStore = cb.newLocal[Long]("nd_get_element_address_data_store")

    coerce[Long](Code(
      dataStore := data.load(nd),
      bytesAway := 0L,
      indices.zipWithIndex.foldLeft(Code._empty) { case (codeSoFar: Code[_], (requestedIndex: Value[Long], strideIndex: Int)) =>
        Code(
          codeSoFar,
          bytesAway := bytesAway + requestedIndex * stridesTuple(strideIndex))
      },
      bytesAway + data.pType.firstElementOffset(dataStore, data.pType.loadLength(dataStore))
    ))
  }

  def setElement(cb: EmitCodeBuilder, region: Value[Region],
    indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: SCode, deepCopy: Boolean): Unit = {
    elementType.storeAtAddress(cb, getElementAddress(indices, ndAddress, cb), region, newElement, deepCopy)
  }

  private def getElementAddressFromDataPointerAndStrides(indices: IndexedSeq[Value[Long]], dataFirstElementPointer: Value[Long], strides: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Long] = {
    val address = cb.newLocal[Long]("nd_get_element_address_bytes_away")
    cb.assign(address, dataFirstElementPointer)

    indices.zipWithIndex.foreach { case (requestedIndex, strideIndex) =>
      cb.assign(address, address + requestedIndex * strides(strideIndex))
    }
    address
  }

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long]): Code[Long] = {
    val off = getElementAddress(indices, ndAddress, cb)
    data.pType.elementType.fundamentalType match {
      case _: PArray | _: PBinary =>
        Region.loadAddress(off)
      case _ =>
        off
    }
  }

  def loadElementFromDataAndStrides(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndDataAddress: Value[Long], strides: IndexedSeq[Value[Long]]): Code[Long] = {
    val off = getElementAddressFromDataPointerAndStrides(indices, ndDataAddress, strides, cb)
    data.pType.elementType.fundamentalType match {
      case _: PArray | _: PBinary =>
        Region.loadAddress(off)
      case _ =>
        off
    }
  }

  def loadElementToIRIntermediate(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], cb: EmitCodeBuilder): Code[_] = {
    Region.loadIRIntermediate(data.pType.elementType)(getElementAddress(indices, ndAddress, cb))
  }

  override def construct(
    shapeCode: SBaseStructCode,
    stridesCode: SBaseStructCode,
    dataCode: Code[Long],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayPointerCode = {

    val dataVal = cb.newLocal[Long]("data_value_store")
    cb.assign(dataVal, dataCode)

    val shapeVal = shapeCode.memoize(cb, "shape_value")

    val stridesVal = stridesCode.memoize(cb, "shape_value")

    val ndAddr = cb.newLocal[Long]("ndarray_construct_addr")
    cb.assign(ndAddr, this.representation.allocate(region))
    shape.pType.storeAtAddress(cb, this.representation.fieldOffset(ndAddr, "shape"), region, shapeVal.get, false)
    strides.pType.storeAtAddress(cb, this.representation.fieldOffset(ndAddr, "strides"), region, stridesVal.get, false)
    cb.append(Region.storeLong(this.representation.fieldOffset(ndAddr, "data"), dataVal))

    new SNDArrayPointerCode(SNDArrayPointer(this), ndAddr)
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

  override def dataFirstElementPointer(ndAddr: Code[Long]): Code[Long] = data.pType.firstElementOffset(this.dataPArrayPointer(ndAddr))

  override def dataPArrayPointer(ndAddr: Code[Long]): Code[Long] = data.load(ndAddr)

  def loadFromNested(cb: EmitCodeBuilder, addr: Code[Long]): Code[Long] = addr
}

