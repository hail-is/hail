package is.hail.types.physical

import is.hail.annotations.{Annotation, NDArray, Region, UnsafeOrdering}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{CodeParam, CodeParamType, EmitCode, EmitCodeBuilder, Param, ParamType, SCodeParam}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerSettable, SNDArrayPointer, SNDArrayPointerCode, SNDArrayPointerSettable, SStackStruct}
import org.apache.spark.sql.Row
import is.hail.utils._

final case class PCanonicalNDArray(elementType: PType, nDims: Int, required: Boolean = false) extends PNDArray  {
  assert(elementType.required, "elementType must be required")
  assert(!elementType.containsPointers, "ndarrays do not currently support elements which contain arrays, ndarrays, or strings")

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
    assert(settables.length == nDims, s"got ${ settables.length } settables, expect ${ nDims } dims")
    val shapeTuple = shapeType.loadCheapSCode(cb, representation.loadField(addr, "shape"))
      .memoize(cb, "pcndarray_shapetuple")
    (0 until nDims).foreach { dimIdx =>
      cb.assign(settables(dimIdx), shapeTuple.loadField(cb, dimIdx).get(cb).asLong.longCode(cb))
    }
  }

  def loadStrides(cb: EmitCodeBuilder, addr: Value[Long], settables: IndexedSeq[Settable[Long]]): Unit = {
    assert(settables.length == nDims)
    val strideTuple = strideType.loadCheapSCode(cb, representation.loadField(addr, "strides"))
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

  lazy val representation: PCanonicalStruct = {
    PCanonicalStruct(required,
      ("shape", shapeType),
      ("strides", strideType),
      ("data", PInt64Required))
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
    var bytesAway = 0L
    indices.zipWithIndex.foreach{case (requestedIndex: Long, strideIndex: Int) =>
      bytesAway += requestedIndex * loadStride(nd, strideIndex)
    }
    bytesAway + this.unstagedDataFirstElementPointer(nd)
  }

  private def getElementAddress(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], nd: Value[Long]): Value[Long] = {
    val ndarrayValue = loadCheapSCode(cb, nd).asNDArray.memoize(cb, "getElementAddressNDValue")
    val stridesTuple = ndarrayValue.strides(cb)

    cb.newLocal[Long]("pcndarray_get_element_addr", indices.zipWithIndex.map { case (requestedElementIndex, strideIndex) =>
      requestedElementIndex * stridesTuple(strideIndex)
    }.foldLeft(const(0L).get)(_ + _) + ndarrayValue.firstDataAddress(cb))
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

  def constructByCopyingArray(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    dataCode: SIndexableCode,
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayCode = {
    assert(shape.length == nDims, s"nDims = ${ nDims }, nShapeElts=${ shape.length }")
    assert(strides.length == nDims, s"nDims = ${ nDims }, nShapeElts=${ strides.length }")

    val cacheKey = ("constructByCopyingArray", this, dataCode.st)
    val mb = cb.emb.ecb.getOrGenEmitMethod("pcndarray_construct_by_copying_array", cacheKey,
      FastIndexedSeq[ParamType](classInfo[Region], dataCode.st.paramType) ++ (0 until 2 * nDims).map(_ => CodeParamType(LongInfo)),
      sType.paramType) { mb =>
      mb.emitSCode { cb =>

        val region = mb.getCodeParam[Region](1)
        val dataValue = mb.getSCodeParam(2).asIndexable.memoize(cb, "pcndarray_construct_by_copying_array_datavalue")
        val shape = (0 until nDims).map(i => mb.getCodeParam[Long](3 + i))
        val strides = (0 until nDims).map(i => mb.getCodeParam[Long](3 + nDims + i))

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

        val newDataPointer = cb.newLocal("ndarray_construct_new_data_pointer", this.allocateData(shape, region))
        cb.append(Region.storeAddress(this.representation.fieldOffset(ndAddr, "data"), newDataPointer))
        val result = new SNDArrayPointerCode(sType, ndAddr).memoize(cb, "construct_by_copying_array_result")

        dataValue.st match {
          case SIndexablePointer(PCanonicalArray(otherElementType, _)) if otherElementType == elementType =>
            cb += Region.copyFrom(dataValue.asInstanceOf[SIndexablePointerSettable].elementsAddress, newDataPointer, dataValue.loadLength().toL * elementType.byteSize)
          case _ =>
            val loopCtr = cb.newLocal[Long]("pcanonical_ndarray_construct_by_copying_loop_idx")
            cb.forLoop(cb.assign(loopCtr, 0L), loopCtr < dataValue.loadLength().toL, cb.assign(loopCtr, loopCtr + 1L), {
              elementType.storeAtAddress(cb, newDataPointer + (loopCtr * elementType.byteSize), region, dataValue.loadElement(cb, loopCtr.toI).get(cb, "NDArray elements cannot be missing"), true)
            })
        }

        result
      }
    }

    cb.invokeSCode(mb, FastIndexedSeq[Param](region, SCodeParam(dataCode)) ++ (shape.map(CodeParam(_)) ++ strides.map(CodeParam(_))): _*)
      .asNDArray
  }

  def constructDataFunction(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): (Value[Long], EmitCodeBuilder =>  SNDArrayPointerCode) = {

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

    val newDataPointer = cb.newLocal("ndarray_construct_new_data_pointer", this.allocateData(shape, region))
    cb.append(Region.storeLong(this.representation.fieldOffset(ndAddr, "data"), newDataPointer))
    val newFirstElementDataPointer = cb.newLocal[Long]("ndarray_construct_first_element_pointer", this.dataFirstElementPointer(ndAddr))

    (newFirstElementDataPointer, (cb: EmitCodeBuilder) => new SNDArrayPointerCode(sType, ndAddr))
  }

  def constructByCopyingDataPointer(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    dataPtr: Code[Long],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayPointerCode = {
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
    cb += Region.storeAddress(this.representation.fieldOffset(ndAddr, 2), dataPtr)
    new SNDArrayPointerCode(sType, ndAddr)
  }

  def constructByActuallyCopyingData(
    toBeCopied: SNDArrayValue,
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): SNDArrayCode = {
    val oldDataAddr = toBeCopied.firstDataAddress(cb)
    val numDataBytes = cb.newLocal("constructByActuallyCopyingData_numDataBytes", Region.getSharedChunkByteSize(oldDataAddr))
    cb.ifx(numDataBytes < 0L, cb._fatal("numDataBytes was ", numDataBytes.toS))
    val newDataAddr = cb.newLocal("constructByActuallyCopyingData_newDataAddr", region.allocateSharedChunk(numDataBytes))
    cb += Region.copyFrom(oldDataAddr, newDataAddr, numDataBytes)
    constructByCopyingDataPointer(
      toBeCopied.shapes(cb),
      toBeCopied.strides(cb),
      newDataAddr,
      cb,
      region
    )
  }

  override def deepRename(t: Type) = deepRenameNDArray(t.asInstanceOf[TNDArray])

  private def deepRenameNDArray(t: TNDArray) =
    PCanonicalNDArray(this.elementType.deepRename(t.elementType), this.nDims, this.required)

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalNDArray(elementType, nDims, required)

  def sType: SNDArrayPointer = SNDArrayPointer(setRequired(false).asInstanceOf[PCanonicalNDArray])

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SCode = new SNDArrayPointerCode(sType, addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    val addr = cb.newField[Long]("pcanonical_ndarray_store", this.representation.allocate(region))
    storeAtAddress(cb, addr, region, value, deepCopy)
    addr
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    val targetAddr = cb.newLocal[Long]("pcanonical_ndarray_store_at_addr_target", addr)
    val inputSNDValue = value.asNDArray.memoize(cb, "pcanonical_ndarray_store_at_addr_input")
    val shape = inputSNDValue.shapes(cb)
    val strides = inputSNDValue.strides(cb)
    val dataAddr = inputSNDValue.firstDataAddress(cb)
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
        val outputSNDValue = new SNDArrayPointerCode(sType, targetAddr).memoize(cb, "pcanonical_ndarray_store_at_addr_output")
        outputSNDValue.coiterateMutate(cb, region, true, (inputSNDValue.get, "input")){
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

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    val addr = this.representation.allocate(region)
    this.unstagedStoreJavaObjectAtAddress(addr, annotation, region)
    addr
  }

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
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
      elementType.unstagedStoreJavaObjectAtAddress(curElementAddress, element, region)
      curElementAddress += elementType.byteSize
    }
    val shapeRow = Row(aNDArray.shape: _*)
    val stridesRow = Row(stridesArray: _*)
    this.representation.unstagedStoreJavaObjectAtAddress(addr, Row(shapeRow, stridesRow, dataFirstElementAddress), region)
  }
}
