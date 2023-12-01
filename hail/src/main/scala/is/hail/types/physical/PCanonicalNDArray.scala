package is.hail.types.physical

import is.hail.annotations.{Annotation, NDArray, Region, UnsafeOrdering}
import is.hail.asm4s.{Code, _}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{CodeParam, CodeParamType, EmitCode, EmitCodeBuilder, Param, ParamType, SCodeParam}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.utils._
import org.apache.spark.sql.Row

final case class PCanonicalNDArray(elementType: PType, nDims: Int, required: Boolean = false) extends PNDArray  {
  assert(elementType.required, "elementType must be required")
  assert(!elementType.containsPointers, "ndarrays do not currently support elements which contain arrays, ndarrays, or strings")

  override def _asIdent: String =
    s"${nDims}darray_of_${elementType.asIdent}"

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
    assert(settables.length == nDims, s"got ${ settables.length } settables, expect ${ nDims } dims")
    val shapeTuple = shapeType.loadCheapSCode(cb, representation.loadField(addr, "shape"))
    (0 until nDims).foreach { dimIdx =>
      cb.assign(settables(dimIdx), shapeTuple.loadField(cb, dimIdx).get(cb).asLong.value)
    }
  }

  def loadStrides(cb: EmitCodeBuilder, addr: Value[Long], settables: IndexedSeq[Settable[Long]]): Unit = {
    assert(settables.length == nDims)
    val strideTuple = strideType.loadCheapSCode(cb, representation.loadField(addr, "strides"))
    (0 until nDims).foreach { dimIdx =>
      cb.assign(settables(dimIdx), strideTuple.loadField(cb, dimIdx).get(cb).asLong.value)
    }
  }

  override def unstagedLoadStrides(addr: Long): IndexedSeq[Long] = {
    (0 until nDims).map { dimIdx =>
      this.loadStride(addr, dimIdx)
    }
  }

  lazy val representation: PCanonicalStruct = {
    PCanonicalStruct(required,
      ("shape", shapeType),
      ("strides", strideType),
      ("data", PInt64Required))
  }

  override lazy val byteSize: Long = representation.byteSize

  override lazy val alignment: Long = representation.alignment

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering = representation.unsafeOrdering(sm)

  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long] = {
    shape.foldLeft(1L: Code[Long])(_ * _)
  }

  def numElements(shape: IndexedSeq[Long]): Long = {
    shape.foldLeft(1L)(_ * _)
  }

  def makeColumnMajorStrides(sourceShapeArray: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = {
    val strides = new Array[Value[Long]](nDims)
    for (i <- 0 until nDims) {
      if (i == 0) strides(i) = const(elementType.byteSize)
      else strides(i) = cb.memoize(strides(i-1) * (sourceShapeArray(i-1) > 0L).mux(sourceShapeArray(i-1), 1L))
    }

    strides
  }

  def makeRowMajorStrides(sourceShapeArray: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = {
    val strides = new Array[Value[Long]](nDims)
    for (i <- (nDims - 1) to 0 by -1) {
      if (i == nDims - 1) strides(i) = const(elementType.byteSize)
      else strides(i) = cb.memoize(strides(i+1) * (sourceShapeArray(i+1) > 0L).mux(sourceShapeArray(i+1), 1L))
    }

    strides
  }

  def getElementAddress(indices: IndexedSeq[Long], nd: Long): Long = {
    var bytesAway = 0L
    indices.zipWithIndex.foreach{case (requestedIndex: Long, strideIndex: Int) =>
      bytesAway += requestedIndex * loadStride(nd, strideIndex)
    }
    bytesAway + this.unstagedDataFirstElementPointer(nd)
  }

  private def getElementAddress(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], nd: Value[Long]): Value[Long] = {
    val ndarrayValue = loadCheapSCode(cb, nd).asNDArray
    val stridesTuple = ndarrayValue.strides

    cb.newLocal[Long]("pcndarray_get_element_addr", indices.zipWithIndex.map { case (requestedElementIndex, strideIndex) =>
      requestedElementIndex * stridesTuple(strideIndex)
    }.foldLeft(const(0L).get)(_ + _) + ndarrayValue.firstDataAddress)
  }

  def setElement(cb: EmitCodeBuilder, region: Value[Region],
    indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: SValue, deepCopy: Boolean): Unit = {
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

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long]): SValue = {
    val off = getElementAddress(cb, indices, ndAddress)
    elementType.loadCheapSCode(cb, elementType.loadFromNested(off))
  }

  def loadElementFromDataAndStrides(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndDataAddress: Value[Long], strides: IndexedSeq[Value[Long]]): Code[Long] = {
    val off = getElementAddressFromDataPointerAndStrides(indices, ndDataAddress, strides, cb)
    elementType.loadFromNested(off)
  }

  def contentsByteSize(numElements: Long): Long = this.elementType.byteSize * numElements

  def contentsByteSize(numElements: Code[Long]): Code[Long] = {
    numElements * elementType.byteSize
  }

  def allocateData(shape: IndexedSeq[Value[Long]], region: Value[Region]): Code[Long] = {
    val sizeOfArray = this.contentsByteSize(this.numElements(shape).toL)
    region.allocateSharedChunk(sizeOfArray)
  }

  def allocateData(shape: IndexedSeq[Long], region: Region): Long = {
    val sizeOfArray: Long = this.contentsByteSize(shape.product)
    region.allocateSharedChunk(sizeOfArray)
  }

  def constructUninitialized(
    shape: IndexedSeq[SizeValue],
    strides: IndexedSeq[Value[Long]],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayPointerValue = {
    constructByCopyingDataPointer(shape, strides, this.allocateData(shape, region), cb, region)
  }

  def constructUninitialized(
    shape: IndexedSeq[SizeValue],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayPointerValue = {
    constructByCopyingDataPointer(shape, makeColumnMajorStrides(shape, cb), this.allocateData(shape, region), cb, region)
  }

  def constructByCopyingArray(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    dataCode: SIndexableValue,
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayValue = {
    assert(shape.length == nDims, s"nDims = ${ nDims }, nShapeElts=${ shape.length }")
    assert(strides.length == nDims, s"nDims = ${ nDims }, nShapeElts=${ strides.length }")

    val mb = cb.emb.ecb.getOrDefineEmitMethod("pcndarray_construct_by_copying_array",
      FastSeq[ParamType](classInfo[Region], dataCode.st.paramType) ++ (0 until 2 * nDims).map(_ => CodeParamType(LongInfo)),
      sType.paramType
    ) { mb =>
      mb.emitSCode { cb =>

        val region = mb.getCodeParam[Region](1)
        val dataValue = mb.getSCodeParam(2).asIndexable
        val shape = (0 until nDims).map(i => SizeValueDyn(mb.getCodeParam[Long](3 + i)))
        val strides = (0 until nDims).map(i => mb.getCodeParam[Long](3 + nDims + i))

        val result = constructUninitialized(shape, strides, cb, region)

        dataValue.st match {
          case SIndexablePointer(PCanonicalArray(otherElementType, _)) if otherElementType == elementType =>
            cb += Region.copyFrom(dataValue.asInstanceOf[SIndexablePointerValue].elementsAddress, result.firstDataAddress, dataValue.loadLength().toL * elementType.byteSize)
          case _ =>
            val loopCtr = cb.newLocal[Long]("pcanonical_ndarray_construct_by_copying_loop_idx")
            cb.for_(cb.assign(loopCtr, 0L), loopCtr < dataValue.loadLength().toL, cb.assign(loopCtr, loopCtr + 1L), {
              elementType.storeAtAddress(cb, result.firstDataAddress + (loopCtr * elementType.byteSize), region, dataValue.loadElement(cb, loopCtr.toI).get(cb, "NDArray elements cannot be missing"), true)
            })
        }

        result
      }
    }

    val newShape = shape.map {
      case s: SizeValue => s
      case s => SizeValueDyn(s)
    }

    cb.invokeSCode(mb, FastSeq[Param](cb.this_, region, SCodeParam(dataCode)) ++ (newShape.map(CodeParam(_)) ++ strides.map(CodeParam(_))): _*)
      .asNDArray
      .coerceToShape(cb, newShape)
  }

  def constructDataFunction(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): (Value[Long], EmitCodeBuilder => SNDArrayPointerValue) = {
    val newShape = shape.map {
      case s: SizeValue => s
      case s => SizeValueDyn(s)
    }
    val result = constructUninitialized(newShape, strides, cb, region)

    (result.firstDataAddress, (cb: EmitCodeBuilder) => result)
  }

  def constructByCopyingDataPointer(
    shape: IndexedSeq[SizeValue],
    strides: IndexedSeq[Value[Long]],
    dataPtr: Code[Long],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayPointerValue = {
    val ndAddr = cb.newLocal[Long]("ndarray_construct_addr")
    cb.assign(ndAddr, this.representation.allocate(region))
    shapeType.storeAtAddress(cb, cb.newLocal[Long]("construct_shape", this.representation.fieldOffset(ndAddr, "shape")),
      region,
      SStackStruct.constructFromArgs(cb, region, shapeType.virtualType, shape.map(s => EmitCode.present(cb.emb, primitive(s))): _*),
      false)
    strideType.storeAtAddress(cb, cb.newLocal[Long]("construct_strides", this.representation.fieldOffset(ndAddr, "strides")),
      region,
      SStackStruct.constructFromArgs(cb, region, strideType.virtualType, strides.map(s => EmitCode.present(cb.emb, primitive(s))): _*),
      false)
    val newDataPointer = cb.newLocal("ndarray_construct_new_data_pointer", dataPtr)
    cb += Region.storeAddress(this.representation.fieldOffset(ndAddr, 2), newDataPointer)
    new SNDArrayPointerValue(sType, ndAddr, shape, strides, newDataPointer)
  }

  def constructByActuallyCopyingData(
    toBeCopied: SNDArrayValue,
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayValue = {
    val oldDataAddr = toBeCopied.firstDataAddress
    val numDataBytes = cb.newLocal("constructByActuallyCopyingData_numDataBytes", Region.getSharedChunkByteSize(oldDataAddr))
    cb.if_(numDataBytes < 0L, cb._fatal("numDataBytes was ", numDataBytes.toS))
    val newDataAddr = cb.newLocal("constructByActuallyCopyingData_newDataAddr", region.allocateSharedChunk(numDataBytes))
    cb += Region.copyFrom(oldDataAddr, newDataAddr, numDataBytes)
    constructByCopyingDataPointer(
      toBeCopied.shapes,
      toBeCopied.strides,
      newDataAddr,
      cb,
      region
    )
  }

  def _copyFromAddress(sm: HailStateManager, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long  = {
    val srcNDPType = srcPType.asInstanceOf[PCanonicalNDArray]
    assert(nDims == srcNDPType.nDims)

    if (equalModuloRequired(srcPType) && !deepCopy) {
      return srcAddress
    }

    val newAddress = this.representation.allocate(region)
    unstagedStoreAtAddress(sm, newAddress, region, srcPType, srcAddress, deepCopy)
    newAddress
  }

  override def deepRename(t: Type) = deepRenameNDArray(t.asInstanceOf[TNDArray])

  private def deepRenameNDArray(t: TNDArray) =
    PCanonicalNDArray(this.elementType.deepRename(t.elementType), this.nDims, this.required)

  def setRequired(required: Boolean): PCanonicalNDArray =
    if(required == this.required) this else PCanonicalNDArray(elementType, nDims, required)

  def unstagedStoreAtAddress(sm: HailStateManager, destAddress: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcNDPType = srcPType.asInstanceOf[PCanonicalNDArray]
    assert(nDims == srcNDPType.nDims)

    if (equalModuloRequired(srcPType)) { // The situation where you can just memcpy
      Region.copyFrom(srcAddress, destAddress, this.representation.field("shape").typ.byteSize + this.representation.field("strides").typ.byteSize)

      val srcDataAddress = srcNDPType.unstagedDataFirstElementPointer(srcAddress)

      assert(!elementType.containsPointers)

      val newDataAddress = {
        if (deepCopy) {
          region.trackSharedChunk(srcDataAddress)
        }
        srcDataAddress
      }
      Region.storeAddress(this.representation.fieldOffset(destAddress, 2), newDataAddress)
    }
    else {  // The situation where maybe the structs inside the ndarray have different requiredness
      val srcShape = srcPType.asInstanceOf[PNDArray].unstagedLoadShapes(srcAddress)
      val srcStrides = srcPType.asInstanceOf[PNDArray].unstagedLoadStrides(srcAddress)

      shapeType.unstagedStoreJavaObjectAtAddress(sm, destAddress, Row(srcShape:_*), region)
      strideType.unstagedStoreJavaObjectAtAddress(sm, destAddress + shapeType.byteSize, Row(srcStrides:_*), region)

      val newDataPointer = this.allocateData(srcShape, region)
      Region.storeLong(this.representation.fieldOffset(destAddress, 2), newDataPointer)

      val newFirstElementDataPointer = this.unstagedDataFirstElementPointer(destAddress)

      var currentAddressToWrite = newFirstElementDataPointer

      SNDArray.unstagedForEachIndex(srcShape) { indices =>
        val srcElementAddress = srcNDPType.getElementAddress(indices, srcAddress)
        this.elementType.unstagedStoreAtAddress(sm, currentAddressToWrite, region, srcNDPType.elementType, srcElementAddress, true)
        currentAddressToWrite += elementType.byteSize
      }
    }
  }

  def sType: SNDArrayPointer = SNDArrayPointer(setRequired(false))

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SNDArrayPointerValue = {
    val a = cb.memoize(addr)
    val shapeTuple = shapeType.loadCheapSCode(cb, representation.loadField(a, "shape"))
    val shape = Array.tabulate(nDims)(i => SizeValueDyn(shapeTuple.loadField(cb, i).get(cb).asLong.value))
    val strideTuple = strideType.loadCheapSCode(cb, representation.loadField(a, "strides"))
    val strides = Array.tabulate(nDims)(strideTuple.loadField(cb, _).get(cb).asLong.value)
    val firstDataAddress = cb.memoize(dataFirstElementPointer(a))
    new SNDArrayPointerValue(sType, a, shape, strides, firstDataAddress)
  }

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): Value[Long] = {
    val addr = cb.memoize(this.representation.allocate(region))
    storeAtAddress(cb, addr, region, value, deepCopy)
    addr
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SValue, deepCopy: Boolean): Unit = {
    val targetAddr = cb.newLocal[Long]("pcanonical_ndarray_store_at_addr_target", addr)
    val inputSNDValue = value.asNDArray
    val shape = inputSNDValue.shapes
    val strides = inputSNDValue.strides
    val dataAddr = inputSNDValue.firstDataAddress
    shapeType.storeAtAddress(cb, cb.newLocal[Long]("construct_shape", this.representation.fieldOffset(targetAddr, "shape")),
      region,
      SStackStruct.constructFromArgs(cb, region, shapeType.virtualType, shape.map(s => EmitCode.present(cb.emb, primitive(s))): _*),
      false)
    strideType.storeAtAddress(cb, cb.newLocal[Long]("construct_strides", this.representation.fieldOffset(targetAddr, "strides")),
      region,
      SStackStruct.constructFromArgs(cb, region, strideType.virtualType, strides.map(s => EmitCode.present(cb.emb, primitive(s))): _*),
      false)

    value.st match {
      case SNDArrayPointer(t) if t.equalModuloRequired(this) =>
        if (deepCopy) {
          region.trackSharedChunk(cb, dataAddr)
        }
        cb += Region.storeAddress(this.representation.fieldOffset(targetAddr, "data"), dataAddr)
      case _ =>
        val newDataAddr = this.allocateData(shape, region)
        cb += Region.storeAddress(this.representation.fieldOffset(targetAddr, "data"), newDataAddr)
        val outputSNDValue = loadCheapSCode(cb, targetAddr)
        outputSNDValue.coiterateMutate(cb, region, true, (inputSNDValue, "input")){
          case Seq(dest, elt) =>
            elt
        }
    }
  }

  def unstagedDataFirstElementPointer(ndAddr: Long): Long =
    Region.loadAddress(representation.loadField(ndAddr, 2))

  override def dataFirstElementPointer(ndAddr: Code[Long]): Code[Long] = Region.loadAddress(representation.loadField(ndAddr, "data"))

  def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def unstagedLoadFromNested(addr: Long): Long = addr

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region): Long = {
    val addr = this.representation.allocate(region)
    this.unstagedStoreJavaObjectAtAddress(sm, addr, annotation, region)
    addr
  }

  override def unstagedStoreJavaObjectAtAddress(sm: HailStateManager, addr: Long, annotation: Annotation, region: Region): Unit = {
    val aNDArray = annotation.asInstanceOf[NDArray]

    var runningProduct = this.elementType.byteSize
    val stridesArray = new Array[Long](aNDArray.shape.size)
    ((aNDArray.shape.size - 1) to 0 by -1).foreach { i =>
      stridesArray(i) = runningProduct
      runningProduct = runningProduct * (if (aNDArray.shape(i) > 0L) aNDArray.shape(i) else 1L)
    }
    val dataFirstElementAddress = this.allocateData(aNDArray.shape, region)
    var curElementAddress = dataFirstElementAddress
    aNDArray.getRowMajorElements().foreach{ element =>
      elementType.unstagedStoreJavaObjectAtAddress(sm, curElementAddress, element, region)
      curElementAddress += elementType.byteSize
    }
    val shapeRow = Row(aNDArray.shape: _*)
    val stridesRow = Row(stridesArray: _*)
    this.representation.unstagedStoreJavaObjectAtAddress(sm, addr, Row(shapeRow, stridesRow, dataFirstElementAddress), region)
  }


  override def copiedType: PType = {
    val copiedElement = elementType.copiedType
    if (copiedElement.eq(elementType))
      this
    else
      PCanonicalNDArray(copiedElement, nDims, required)
  }
}
