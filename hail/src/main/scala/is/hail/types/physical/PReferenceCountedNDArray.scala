package is.hail.types.physical
import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.utils._
import is.hail.asm4s.{Code, _}
import is.hail.types.virtual.Type

class PReferenceCountedNDArray(val elementType: PType, val nDims: Int, val required: Boolean = false) extends PNDArray {
  @transient lazy val shape = new StaticallyKnownField(
    PCanonicalTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*): PTuple,
    off => 0L
  )

  @transient lazy val strides = new StaticallyKnownField(
    PCanonicalTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*): PTuple,
    (off) => 0L
  )

  override lazy val data: StaticallyKnownField[PArray, Long] = ???
  override lazy val representation: PStruct = ???

  override def numElements(shape: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long] = ???

  override def setElement(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: Code[_], mb: EmitMethodBuilder[_]): Code[Unit] = ???

  override def loadElementToIRIntermediate(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], mb: EmitMethodBuilder[_]): Code[_] = ???

  override def linearizeIndicesRowMajor(indices: IndexedSeq[Code[Long]], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long] = ???

  override def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): (Code[Unit], IndexedSeq[Value[Long]]) = ???

  override def construct(shapeBuilder: StagedRegionValueBuilder => Code[Unit], stridesBuilder: StagedRegionValueBuilder => Code[Unit], data: Code[Long], mb: EmitMethodBuilder[_], region: Value[Region]): Code[Long] = {
    ???
  }

  def allocateAndInitialize(cb: EmitCodeBuilder, region: Value[Region], shapeValue: PBaseStructValue, stridesValue: PBaseStructValue): Value[Long] = {
    val sizeInBytes = cb.newField[Long]("ndarray_initialize_size_in_bytes")
    val numElements = (0 until nDims).map(i => shapeValue.loadField(cb, i).get(cb).tcode[Long]).foldLeft(const(1L).get)((a, b) => a * b)

    cb.assign(sizeInBytes, numElements * elementType.byteSize + shape.pType.byteSize + strides.pType.byteSize)
    val ndAddr = cb.newField[Long]("ndarray_addr_alloc_and_init")
    cb.assign(ndAddr, region.allocateNDArray(sizeInBytes))

    (0 until nDims).map(i => {
      cb.append(Region.storeLong(ndAddr + (8 * i), shapeValue.loadField(cb, i).get(cb).tcode[Long]))
      cb.append(Region.storeLong(ndAddr + shape.pType.byteSize + (8 * i), shapeValue.loadField(cb, i).get(cb).tcode[Long]))
    })

    ndAddr
  }

  def elementsAddress(ndAddr: Code[Long]): Code[Long] = {
    ndAddr + shape.pType.byteSize + strides.pType.byteSize
  }

  override def _asIdent: String = ???

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = ???

  override def encodableType: PType = new PReferenceCountedNDArray(elementType.encodableType, nDims, required)

  override def setRequired(required: Boolean): PType = new PReferenceCountedNDArray(this.elementType, this.nDims, required)

  override def containsPointers: Boolean = true

  override def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    EmitCodeBuilder.scopedCode[Long](mb) { cb =>
      val dest = cb.newField[Long]("ref_counted_ndarray_cFT_dest")
      cb.assign(dest, region.allocate(8L, 8L))
      cb.append(Region.storeAddress(dest, srcAddress))
      if (deepCopy) {
        cb.append(region.trackNDArray(srcAddress))
      }
      dest
    }
  }

  override def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] = {
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], deepCopy)
  }

  override protected def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    val destAddr = region.allocate(8L)
    Region.storeAddress(destAddr, srcAddress)
    if (deepCopy) {
      region.trackNDArray(srcAddress)
    }
    destAddr
  }

  override def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] = {
    EmitCodeBuilder.scopedVoid(mb) {cb =>
      val storedSrcAddr = cb.newLocal[Long]("ref_count_ndarray_caa_src")
      cb.assign(storedSrcAddr, srcAddress)
      cb.append(Region.storeAddress(addr, storedSrcAddr))
      if (deepCopy) {
        cb.append(region.trackNDArray(storedSrcAddr))
      }
    }
  }

  override def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    Region.storeAddress(addr, srcAddress)
    if (deepCopy) {
      region.trackNDArray(addr)
    }
  }

  override def unsafeOrdering(): UnsafeOrdering = throw new NotImplementedError("Not implemented.")
}
