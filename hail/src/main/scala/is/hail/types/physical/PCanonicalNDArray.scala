package is.hail.types.physical

import is.hail.annotations.{Annotation, NDArray, Region, UnsafeOrdering}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{CodeParam, CodeParamType, EmitCode, EmitCodeBuilder, PCodeParam, Param, ParamType}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SNDArrayPointer, SNDArrayPointerCode}
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

  def loadShape(ndAddr: Long, idx: Int): Long = {
    val shapeTupleAddr = representation.loadField(ndAddr, 0)
    Region.loadLong(shapeType.loadField(shapeTupleAddr, idx))
  }

  def loadStride(ndAddr: Long, idx: Int): Long = {
    val shapeTupleAddr = representation.loadField(ndAddr, 1)
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

  override def unstagedLoadStrides(addr: Long): IndexedSeq[Long] = {
    (0 until nDims).map { dimIdx =>
      this.loadStride(addr, dimIdx)
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

  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long] = {
    shape.foldLeft(1L: Code[Long])(_ * _)
  }

  def numElements(shape: IndexedSeq[Long]): Long = {
    shape.foldLeft(1L)(_ * _)
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
    region.allocateNDArray(sizeOfArray + sizeOfStruct)
  }

  def allocate(shape: IndexedSeq[Long], region: Region): Long = {
    //Need to allocate enough space to construct my tuple, then to construct the array right next to it.
    val sizeOfArray: Long = this.dataType.contentsByteSize(shape.product.toInt)
    val sizeOfStruct = this.representation.byteSize
    region.allocateNDArray(sizeOfArray + sizeOfStruct)
  }

  def constructByCopyingArray(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    dataCode: SIndexableCode,
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): PNDArrayCode = {

    val cacheKey = ("constructByCopyingArray", this, dataCode.st)
    val mb = cb.emb.ecb.getOrGenEmitMethod("pcndarray_construct_by_copying_array", cacheKey,
      FastIndexedSeq[ParamType](classInfo[Region], dataCode.st.paramType) ++ (0 until 2 * nDims).map(_ => CodeParamType(LongInfo)),
      sType.paramType) { mb =>
      mb.emitPCode { cb =>

        val region = mb.getCodeParam[Region](1)
        val dataValue = mb.getPCodeParam(2).asIndexable.memoize(cb, "pcndarray_construct_by_copying_array_datavalue")
        val shape = (0 until nDims).map(i => mb.getCodeParam[Long](3 + i))
        val strides = (0 until nDims).map(i => mb.getCodeParam[Long](3 + nDims + i))

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
        dataType.storeContentsAtAddress(cb, newDataPointer, region, dataValue, true)

        new SNDArrayPointerCode(SNDArrayPointer(this), ndAddr)
      }
    }

    cb.invokePCode(mb, FastIndexedSeq[Param](region, PCodeParam(dataCode.asPCode)) ++ (shape.map(CodeParam(_)) ++ strides.map(CodeParam(_))): _*)
      .asNDArray
  }

  def constructDataFunction(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): (Value[Long], EmitCodeBuilder =>  SNDArrayPointerCode) = {

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
    //TODO Use the known length here
    val newFirstElementDataPointer = cb.newLocal[Long]("ndarray_construct_first_element_pointer", this.dataFirstElementPointer(ndAddr))

    cb.append(dataType.stagedInitialize(newDataPointer, this.numElements(shape).toI))

    (newFirstElementDataPointer, (cb: EmitCodeBuilder) => new SNDArrayPointerCode(SNDArrayPointer(this), ndAddr))
  }

  def unstagedConstructDataFunction(
     shape: IndexedSeq[Long],
     strides: IndexedSeq[Long],
     region: Region
   )(writeDataToAddress: Long => Unit): Long = {

    val ndAddr = this.allocate(shape, region)
    shapeType.unstagedStoreJavaObjectAtAddress(ndAddr, Row(shape:_*), region)
    strideType.unstagedStoreJavaObjectAtAddress(ndAddr + shapeType.byteSize, Row(strides:_*), region)

    val newDataPointer = ndAddr + this.representation.byteSize
    Region.storeLong(this.representation.fieldOffset(ndAddr, 2), newDataPointer)

    val newFirstElementDataPointer = this.unstagedDataFirstElementPointer(ndAddr)
    dataType.initialize(newDataPointer, numElements(shape).toInt)
    writeDataToAddress(newFirstElementDataPointer)

    ndAddr
  }

  private def deepPointerCopy(region: Region, ndAddress: Long): Unit = {
    // Tricky, need to rewrite the address of the data pointer to point to directly after the struct.
    val shape = this.unstagedLoadShapes(ndAddress)
    val firstElementAddressOld = this.unstagedDataFirstElementPointer(ndAddress)
    assert(this.elementType.containsPointers)
    val arrayAddressNew = ndAddress + this.representation.byteSize
    val numElements = this.numElements(shape)
    this.dataType.initialize(arrayAddressNew, numElements.toInt)
    Region.storeLong(this.representation.fieldOffset(ndAddress, 2), arrayAddressNew)
    val firstElementAddressNew = this.dataType.firstElementOffset(arrayAddressNew)


    var currentIdx = 0
    while(currentIdx < numElements) {
      val currentElementAddressOld = firstElementAddressOld + currentIdx * elementType.byteSize
      val currentElementAddressNew = firstElementAddressNew + currentIdx * elementType.byteSize
      this.elementType.unstagedStoreAtAddress(currentElementAddressNew, region, this.elementType, elementType.unstagedLoadFromNested(currentElementAddressOld), true)
      currentIdx += 1
    }
  }

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long  = {
    val srcNDPType = srcPType.asInstanceOf[PCanonicalNDArray]
    assert(nDims == srcNDPType.nDims)


    if (equalModuloRequired(srcPType)) { // The situation where you can just memcpy, but then still have to update pointers.
      if (!deepCopy) {
        return srcAddress
      }

      // Deep copy, two scenarios.
      if (elementType.containsPointers) {
        // Can't just reference count change, since the elements have to be copied and updated.
        val numBytes = PNDArray.getByteSize(srcAddress)
        val newAddress =  region.allocateNDArray(numBytes)
        Region.copyFrom(srcAddress, newAddress, numBytes)
        deepPointerCopy(region, newAddress)
        newAddress
      }
      else {
        region.trackNDArray(srcAddress)
        srcAddress
      }
    }
    else {  // The situation where maybe the structs inside the ndarray have different requiredness
      // Deep copy doesn't matter, we have to make a new one no matter what.
      val srcShape = srcPType.asInstanceOf[PNDArray].unstagedLoadShapes(srcAddress)
      val srcStrides = srcPType.asInstanceOf[PNDArray].unstagedLoadStrides(srcAddress)
      val newAddress = this.unstagedConstructDataFunction(srcShape, srcStrides, region){ firstElementAddress =>
        var currentAddressToWrite = firstElementAddress

        SNDArray.unstagedForEachIndex(srcShape) { indices =>
          val srcElementAddress = srcNDPType.getElementAddress(indices, srcAddress)
          this.elementType.unstagedStoreAtAddress(currentAddressToWrite, region, srcNDPType.elementType, srcElementAddress, true)
          currentAddressToWrite += elementType.byteSize
        }
      }

      newAddress
    }

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
        val storedAddress = cb.newLocal[Long]("pcanonical_ndarray_store", value.asInstanceOf[SNDArrayPointerCode].a)
        if (deepCopy) {
          cb.append(region.trackNDArray(storedAddress))
        }
        storedAddress
      case SNDArrayPointer(t) =>
        val oldND = value.asNDArray.memoize(cb, "pcanonical_ndarray_store_old")
        val shape = oldND.shapes(cb)
        val newStrides = makeColumnMajorStrides(shape, region, cb)
        val (targetDataFirstElementAddr, finish) = this.constructDataFunction(shape, newStrides, cb, region)

        val currentOffset = cb.newLocal[Long]("pcanonical_ndarray_store_offset", targetDataFirstElementAddr)
        SNDArray.forEachIndex(cb, shape, "PCanonicalNDArray_store") { (cb, currentIndices) =>
          val oldElement = oldND.loadElement(currentIndices, cb)
          elementType.storeAtAddress(cb, currentOffset, region, oldElement, true)
          cb.assign(currentOffset, currentOffset + elementType.byteSize)
        }

        finish(cb).a
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    cb += Region.storeAddress(addr, store(cb, region, value, deepCopy))
  }

  def unstagedDataFirstElementPointer(ndAddr: Long): Long = dataType.firstElementOffset(unstagedDataPArrayPointer(ndAddr))

  def unstagedDataPArrayPointer(ndAddr: Long): Long = representation.loadField(ndAddr, 2)

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
