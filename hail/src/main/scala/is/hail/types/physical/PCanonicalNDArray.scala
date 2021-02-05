package is.hail.types.physical

import is.hail.annotations.{Annotation, NDArray, Region, UnsafeOrdering}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.types.physical.stypes.concrete.{SNDArrayPointer, SNDArrayPointerCode}
import org.apache.spark.sql.Row

import is.hail.utils._

final case class PCanonicalNDArray(elementType: PType, nDims: Int, required: Boolean = false) extends PNDArray  {
  assert(elementType.required, "elementType must be required")

  def _asIdent: String = s"ndarray_of_${elementType.asIdent}"

  override def containsPointers: Boolean = true

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCNDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append(s",$nDims]")
  }

  lazy val shapeType: PCanonicalTuple = PCanonicalTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*)
  lazy val strideType: PCanonicalTuple = shapeType

  def loadShape(off: Long, idx: Int): Long = {
    val shapeTupleAddr = representation.loadField(off, 0)
    Region.loadLong(shapeType.loadField(shapeTupleAddr, idx))
  }

  def loadStride(off: Long, idx: Int): Long = {
    val shapeTupleAddr = representation.loadField(off, 1)
    Region.loadLong(strideType.loadField(shapeTupleAddr, idx))
  }


  def loadShapes(cb: EmitCodeBuilder, addr: Value[Long], settables: IndexedSeq[Settable[Long]]): Unit = {
    assert(settables.length == nDims)
    val shapeTuple = shapeType.loadCheapPCode(cb, representation.loadField(addr, "shape"))
      .memoize(cb, "pcndarray_shapetuple")
    (0 until nDims).foreach { dimIdx =>
      cb.assign(settables(dimIdx), shapeTuple.loadField(cb, dimIdx).get(cb).asLong.longCode(cb))
    }
  }
  
  def loadStrides(cb: EmitCodeBuilder, addr: Value[Long], settables: IndexedSeq[Settable[Long]]): Unit = {
    assert(settables.length == nDims)
    val strideTuple = strideType.loadCheapPCode(cb, representation.loadField(addr, "strides"))
      .memoize(cb, "pcndarray_stridetuple")
    (0 until nDims).foreach { dimIdx =>
      cb.assign(settables(dimIdx), strideTuple.loadField(cb, dimIdx).get(cb).asLong.longCode(cb))
    }
  }
  
  val dataType: PCanonicalArray = PCanonicalArray(elementType, required = true)  

  lazy val representation: PCanonicalStruct = {
    PCanonicalStruct(required,
      ("shape", shapeType),
      ("strides", strideType),
      ("data", dataType))
  }

  override lazy val byteSize: Long = representation.byteSize

  override lazy val alignment: Long = representation.alignment

  override def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override lazy val fundamentalType: PType = this

  override lazy val encodableType: PType = PCanonicalNDArray(elementType.encodableType, nDims, required)

  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long] = {
    shape.foldLeft(1L: Code[Long])(_ * _)
  }

  def makeColumnMajorStrides(sourceShapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = {
    val runningProduct = cb.newLocal[Long]("make_column_major_strides_prod")
    val computedStrides = (0 until nDims).map(idx => cb.newField[Long](s"make_column_major_computed_stride_${idx}"))

    cb.assign(runningProduct, elementType.byteSize)
    (0 until nDims).foreach{ index =>
      cb.assign(computedStrides(index), runningProduct)
      cb.assign(runningProduct, runningProduct * (sourceShapeArray(index) > 0L).mux(sourceShapeArray(index), 1L))
    }

    computedStrides
  }

  def makeRowMajorStrides(sourceShapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = {
    val runningProduct = cb.newLocal[Long]("make_row_major_strides_prod")
    val computedStrides = (0 until nDims).map(idx => cb.newField[Long](s"make_row_major_computed_stride_${idx}"))

    cb.assign(runningProduct, elementType.byteSize)
    ((nDims - 1) to 0 by -1).foreach{ index =>
      cb.assign(computedStrides(index), runningProduct)
      cb.assign(runningProduct, runningProduct * (sourceShapeArray(index) > 0L).mux(sourceShapeArray(index), 1L))
    }

    computedStrides
  }

  def getElementAddress(indices: IndexedSeq[Long], nd: Long): Long = {
    val dataLength = (0 until nDims).map(loadShape(nd, _)).foldLeft(1L)(_ * _)
    val dataAddress = this.representation.loadField(nd, 2)

    var bytesAway = 0L
    indices.zipWithIndex.foreach{case (requestedIndex: Long, strideIndex: Int) =>
      bytesAway += requestedIndex * loadStride(nd, strideIndex)
    }

    bytesAway + dataType.firstElementOffset(dataAddress, dataLength.toInt)
  }

  private def getElementAddress(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], nd: Value[Long]): Value[Long] = {
    val ndarrayValue = PCode(this, nd).asNDArray.memoize(cb, "getElementAddressNDValue")
    val stridesTuple = ndarrayValue.strides(cb)

    val dataStore = cb.newLocal[Long]("nd_get_element_address_data_store",
      representation.loadField(nd, "data"))

    cb.newLocal[Long]("pcndarray_get_element_addr", indices.zipWithIndex.map { case (requestedElementIndex, strideIndex) =>
      requestedElementIndex * stridesTuple(strideIndex)
    }.foldLeft(const(0L).get)(_ + _) + dataType.firstElementOffset(dataStore, dataType.loadLength(dataStore)))
  }

  def setElement(cb: EmitCodeBuilder, region: Value[Region],
    indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: SCode, deepCopy: Boolean): Unit = {
    elementType.storeAtAddress(cb, getElementAddress(cb, indices, ndAddress), region, newElement, deepCopy)
  }

  private def getElementAddressFromDataPointerAndStrides(indices: IndexedSeq[Value[Long]], dataFirstElementPointer: Value[Long], strides: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Long] = {
    val address = cb.newLocal[Long]("nd_get_element_address_bytes_away")
    cb.assign(address, dataFirstElementPointer)

    indices.zipWithIndex.foreach { case (requestedIndex, strideIndex) =>
      cb.assign(address, address + requestedIndex * strides(strideIndex))
    }
    address
  }

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long]): SCode = {
    val off = getElementAddress(cb, indices, ndAddress)
    elementType.loadCheapPCode(cb, elementType.loadFromNested(off))
  }

  def loadElementFromDataAndStrides(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndDataAddress: Value[Long], strides: IndexedSeq[Value[Long]]): Code[Long] = {
    val off = getElementAddressFromDataPointerAndStrides(indices, ndDataAddress, strides, cb)
    elementType.loadFromNested(off)
  }

  def allocate(shape: IndexedSeq[Value[Long]], region: Value[Region]): Code[Long] = {
    //Need to allocate enough space to construct my tuple, then to construct the array right next to it.
    val sizeOfArray = this.dataType.contentsByteSize(this.numElements(shape).toI)
    val sizeOfStruct = this.representation.byteSize
    // TODO: There might be alignment issues here
    region.allocateNDArray(sizeOfArray + sizeOfStruct)
  }

  def allocate(shape: IndexedSeq[Long], region: Region): Long = {
    //Need to allocate enough space to construct my tuple, then to construct the array right next to it.
    val sizeOfArray: Long = this.dataType.contentsByteSize(shape.product.toInt)
    val sizeOfStruct = this.representation.byteSize
    // TODO: There might be alignment issues here
    region.allocateNDArray(sizeOfArray + sizeOfStruct)
  }

  def construct(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    dataCode: Code[Long],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayPointerCode = {

    val inputDataPointer = cb.newLocal[Long]("data_value_store")
    cb.assign(inputDataPointer, dataCode)

    val ndAddr = cb.newLocal[Long]("ndarray_construct_addr")
    cb.assign(ndAddr, this.allocate(shape, region))
    shapeType.storeAtAddressFromFields(cb, cb.newLocal[Long]("construct_shape", this.representation.fieldOffset(ndAddr, "shape")),
      region,
      shape.map(s => EmitCode.present(cb.emb, primitive(s))),
      false)
    strideType.storeAtAddressFromFields(cb, cb.newLocal[Long]("construct_strides", this.representation.fieldOffset(ndAddr, "strides")),
      region,
      strides.map(s => EmitCode.present(cb.emb, primitive(s))),
      false)

    val newDataPointer = cb.newLocal("ndarray_construct_new_data_pointer", ndAddr + this.representation.byteSize)

    cb.append(Region.storeLong(this.representation.fieldOffset(ndAddr, "data"), newDataPointer))
    // TODO Don't keep recomputing num elements
    cb.append(Region.copyFrom(inputDataPointer, newDataPointer, this.dataType.contentsByteSize(this.numElements(shape).toI)))

    new SNDArrayPointerCode(SNDArrayPointer(this), ndAddr)
  }

  def constructDataFunction(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    cb: EmitCodeBuilder,
    region: Value[Region]
  )(writeDataToAddress: (Value[Long], EmitCodeBuilder) => Unit):  SNDArrayPointerCode = {

    val ndAddr = cb.newLocal[Long]("ndarray_construct_addr")
    cb.assign(ndAddr, this.allocate(shape, region))
    cb.println("Allocated in constructDataFunction, address was: ", ndAddr.toS)
    shapeType.storeAtAddressFromFields(cb, cb.newLocal[Long]("construct_shape", this.representation.fieldOffset(ndAddr, "shape")),
      region,
      shape.map(s => EmitCode.present(primitive(s))),
      false)
    strideType.storeAtAddressFromFields(cb, cb.newLocal[Long]("construct_strides", this.representation.fieldOffset(ndAddr, "strides")),
      region,
      strides.map(s => EmitCode.present(primitive(s))),
      false)

    val newDataPointer = cb.newLocal("ndarray_construct_new_data_pointer", ndAddr + this.representation.byteSize)
    //TODO Use the known length here
    val newFirstElementDataPointer = cb.newLocal[Long]("ndarray_construct_first_element_pointer", dataType.firstElementOffset(newDataPointer))

    cb.append(Region.storeLong(this.representation.fieldOffset(ndAddr, "data"), newDataPointer))
    cb.append(dataType.stagedInitialize(newDataPointer, this.numElements(shape).toI))
    writeDataToAddress(newFirstElementDataPointer, cb)

    new SNDArrayPointerCode(SNDArrayPointer(this), ndAddr)
  }

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long  = {
    val sourceNDPType = srcPType.asInstanceOf[PCanonicalNDArray]
    assert(elementType == sourceNDPType.elementType && nDims == sourceNDPType.nDims)
    representation.copyFromAddress(region, sourceNDPType.representation, srcAddress, deepCopy)
  }

  override def deepRename(t: Type) = deepRenameNDArray(t.asInstanceOf[TNDArray])

  private def deepRenameNDArray(t: TNDArray) =
    PCanonicalNDArray(this.elementType.deepRename(t.elementType), this.nDims, this.required)

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalNDArray(elementType, nDims, required)

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcND = srcPType.asInstanceOf[PCanonicalNDArray]

    if (deepCopy) {
      region.trackNDArray(srcAddress)
    }
    Region.storeAddress(addr, copyFromAddress(region, srcND, srcAddress, deepCopy))
  }

  def sType: SNDArrayPointer = SNDArrayPointer(this)

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SNDArrayPointerCode(sType, addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SNDArrayPointer(t) if t.equalModuloRequired(this)  =>
        // Does requiredness matter?
        val storedAddress = cb.newLocal[Long]("pcanonical_ndarray_store", value.asInstanceOf[SNDArrayPointerCode].a)
        if (deepCopy) {
          cb.println(s"Asked to deepCopy, going to track ", storedAddress.toS)
          region.trackNDArray(storedAddress)
        }
        storedAddress
      case SNDArrayPointer(t) =>
        ???
      case _ =>
        throw new UnsupportedOperationException("Only know how to store SNDArrayPointCodes")
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    cb += Region.storeAddress(addr, store(cb, region, value, deepCopy))
  }

  override def dataFirstElementPointer(ndAddr: Code[Long]): Code[Long] = dataType.firstElementOffset(this.dataPArrayPointer(ndAddr))

  override def dataPArrayPointer(ndAddr: Code[Long]): Code[Long] = representation.loadField(ndAddr, "data")

  def loadFromNested(addr: Code[Long]): Code[Long] = Region.loadAddress(addr)

  override def unstagedLoadFromNested(addr: Long): Long = Region.loadAddress(addr)

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    val aNDArray = annotation.asInstanceOf[NDArray]
    val addr = this.allocate(aNDArray.shape, region)

    val shapeRow = Annotation.fromSeq(aNDArray.shape)
    var runningProduct = this.representation.fieldType("data").asInstanceOf[PArray].elementType.byteSize
    val stridesArray = new Array[Long](aNDArray.shape.size)
    ((aNDArray.shape.size - 1) to 0 by -1).foreach { i =>
      stridesArray(i) = runningProduct
      runningProduct = runningProduct * (if (aNDArray.shape(i) > 0L) aNDArray.shape(i) else 1L)
    }
    var curAddr = addr
    val stridesRow = Row(stridesArray:_*)
    shapeType.unstagedStoreJavaObjectAtAddress(curAddr, shapeRow, region)
    curAddr += shapeType.byteSize
    strideType.unstagedStoreJavaObjectAtAddress(curAddr, stridesRow, region)
    curAddr += shapeType.byteSize
    dataType.unstagedStoreJavaObjectAtAddress(curAddr, aNDArray.getRowMajorElements(), region)

    addr
  }

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeAddress(addr, unstagedStoreJavaObject(annotation, region))
  }
}

