package is.hail.types.physical

import is.hail.annotations.{Region, _}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, EmitValue, IEmitCode}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerCode, SIndexablePointerSettable}
import is.hail.types.physical.stypes.interfaces.SContainer
import is.hail.types.virtual.{TArray, Type}
import is.hail.utils._

// This is a pointer array, whose byteSize is the size of its pointer
final case class PCanonicalArray(elementType: PType, required: Boolean = false) extends PArray {
  assert(elementType.isRealizable)

  def _asIdent = s"array_of_${elementType.asIdent}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCArray[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }


  val elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  val contentsAlignment: Long = elementType.alignment.max(4)

  val lengthHeaderBytes: Long = 4

  override val byteSize: Long = 8

  override val fundamentalType: PCanonicalArray = {
    if (elementType == elementType.fundamentalType) {
      this
    } else {
      this.copy(elementType = elementType.fundamentalType)
    }
  }

  override val encodableType: PCanonicalArray = {
    if (elementType == elementType.encodableType) {
      this
    } else {
      this.copy(elementType = elementType.encodableType)
    }
  }

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalArray(elementType, required)

  def loadLength(aoff: Long): Int =
    Region.loadInt(aoff)

  def loadLength(aoff: Code[Long]): Code[Int] =
    Region.loadInt(aoff)

  def storeLength(aoff: Long, length: Int): Unit =
    Region.storeInt(aoff, length)

  def storeLength(aoff: Code[Long], length: Code[Int]): Code[Unit] =
    Region.storeInt(aoff, length)

  def nMissingBytes(len: Code[Int]): Code[Int] = UnsafeUtils.packBitsToBytes(len)

  def nMissingBytes(len: Int): Int = UnsafeUtils.packBitsToBytes(len)

  private def contentsByteSize(length: Int): Long =
    elementsOffset(length) + length * elementByteSize

  private def contentsByteSize(length: Code[Int]): Code[Long] = {
    Code.memoize(length, "contentsByteSize_arr_len") { length =>
      elementsOffset(length) + length.toL * elementByteSize
    }
  }

  private def _elementsOffset(length: Int): Long =
    if (elementRequired)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes + nMissingBytes(length), elementType.alignment)

  private def _elementsOffset(length: Code[Int]): Code[Long] =
    if (elementRequired)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(nMissingBytes(length).toL + lengthHeaderBytes, elementType.alignment)

  private lazy val lengthOffsetTable = 10
  private lazy val elementsOffsetTable: Array[Long] = Array.tabulate[Long](lengthOffsetTable)(i => _elementsOffset(i))

  def elementsOffset(length: Int): Long = {
    if (length < lengthOffsetTable)
      elementsOffsetTable(length)
    else
      _elementsOffset(length)
  }

  def elementsOffset(length: Code[Int]): Code[Long] = {
    _elementsOffset(length)
  }

  def isElementDefined(aoff: Long, i: Int): Boolean =
    elementRequired || !Region.loadBit(aoff + lengthHeaderBytes, i)

  def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    if (elementRequired)
      true
    else
      !Region.loadBit(aoff + lengthHeaderBytes, i.toL)

  def isElementMissing(aoff: Long, i: Int): Boolean =
    !isElementDefined(aoff, i)

  def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    !isElementDefined(aoff, i)

  def setElementMissing(aoff: Long, i: Int) {
    if (!elementRequired)
      Region.setBit(aoff + lengthHeaderBytes, i)
  }

  def setElementMissing(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    if (!elementRequired)
      Region.setBit(aoff + lengthHeaderBytes, i.toL)
    else
      Code._fatal[Unit](s"Required element cannot be missing")

  def setElementPresent(aoff: Long, i: Int) {
    if (!elementRequired)
      Region.clearBit(aoff + lengthHeaderBytes, i.toLong)
  }

  def setElementPresent(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    if (!elementRequired)
      Region.clearBit(aoff + lengthHeaderBytes, i.toL)
    else
      Code._empty

  def firstElementOffset(aoff: Long, length: Int): Long =
    aoff + elementsOffset(length)

  def firstElementOffset(aoff: Code[Long], length: Code[Int]): Code[Long] =
    aoff + elementsOffset(length)

  def firstElementOffset(aoff: Code[Long]): Code[Long] =
    Code.memoize(aoff, "pcarr_first_elem_off_aoff") { aoff =>
      firstElementOffset(aoff, loadLength(aoff))
    }

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    firstElementOffset(aoff, length) + i * elementByteSize

  def elementOffset(aoff: Long, i: Int): Long =
    firstElementOffset(aoff, loadLength(aoff)) + i * elementByteSize

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    firstElementOffset(aoff, length) + i.toL * const(elementByteSize)

  def elementOffset(aoff: Code[Long], i: Code[Int]): Code[Long] =
    Code.memoize(aoff, "pcarr_elem_off_aoff") { aoff =>
      firstElementOffset(aoff, loadLength(aoff)) + i.toL * const(elementByteSize)
    }

  private def elementOffsetFromFirst(firstElementAddr: Code[Long], i: Code[Int]): Code[Long] = firstElementAddr + i.toL * const(elementByteSize)

  def nextElementAddress(currentOffset: Long) =
    currentOffset + elementByteSize

  def nextElementAddress(currentOffset: Code[Long]) =
    currentOffset + elementByteSize

  def loadElement(aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(aoff: Long, i: Int): Long = loadElement(aoff, loadLength(aoff), i)

  def loadElement(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(aoff: Code[Long], i: Code[Int]): Code[Long] =
    Code.memoize(aoff, "pcarr_load_elem_aoff") { aoff =>
      loadElement(aoff, loadLength(aoff), i)
    }

  class Iterator (
    private[this] val aoff: Long,
    private[this] val length: Int,
    private[this] var i: Int = 0
  ) extends PArrayIterator {
    private[this] val firstElementOffset = PCanonicalArray.this.firstElementOffset(
      aoff, length)
    def hasNext: Boolean = i != length
    def isDefined: Boolean = isElementDefined(aoff, i)
    def value: Long =
      firstElementOffset + i * elementByteSize
    def iterate: Unit = i += 1
  }

  def elementIterator(aoff: Long, length: Int): Iterator = new Iterator(aoff, length)

  def allocate(region: Region, length: Int): Long = {
    region.allocate(contentsAlignment, contentsByteSize(length))
  }

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    region.allocate(contentsAlignment, contentsByteSize(length))

  private def writeMissingness(aoff: Long, length: Int, value: Byte) {
    Region.setMemory(aoff + lengthHeaderBytes, nMissingBytes(length), value)
  }

  def setAllMissingBits(aoff: Long, length: Int) {
    if (!elementRequired)
      writeMissingness(aoff, length, -1)
  }

  def clearMissingBits(aoff: Long, length: Int) {
    if (!elementRequired)
      writeMissingness(aoff, length, 0)
  }

  def initialize(aoff: Long, length: Int, setMissing: Boolean = false) {
    Region.storeInt(aoff, length)
    if (setMissing)
      setAllMissingBits(aoff, length)
    else
      clearMissingBits(aoff, length)
  }

  def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Code[Unit] = {
    if (elementRequired)
      Region.storeInt(aoff, length)
    else
      Code.memoize(aoff, "staged_init_aoff",
        length, "staged_init_length") { (aoff, length) =>
        Code(
          Region.storeInt(aoff, length),
          Region.setMemory(aoff + const(lengthHeaderBytes), nMissingBytes(length).toL, const(if (setMissing) (-1).toByte else 0.toByte)))
      }
  }

  def zeroes(region: Region, length: Int): Long = {
    require(elementType.isNumeric)
    val aoff = allocate(region, length)
    initialize(aoff, length)
    Region.setMemory(aoff + elementsOffset(length), length * elementByteSize, 0.toByte)
    aoff
  }

  def zeroes(mb: EmitMethodBuilder[_], region: Value[Region], length: Code[Int]): Code[Long] = {
    require(elementType.isNumeric)
    Code.memoize(length, "pcarr_zeros_len") { length =>
      val aoff = mb.newLocal[Long]()
      Code(
        aoff := allocate(region, length),
        stagedInitialize(aoff, length),
        Region.setMemory(aoff + elementsOffset(length), length.toL * elementByteSize, 0.toByte),
        aoff)
    }
  }

  def forEach(mb: EmitMethodBuilder[_], aoff: Code[Long], body: Code[Long] => Code[Unit]): Code[Unit] = {
    val i = mb.newLocal[Int]()
    val n = mb.newLocal[Int]()
    Code(
      n := loadLength(aoff),
      i := 0,
      Code.whileLoop(i < n,
        isElementDefined(aoff, i).mux(
          body(loadElement(aoff, n, i)),
          Code._empty
        )))
  }

  override def unsafeOrdering(): UnsafeOrdering =
    unsafeOrdering(this)

  override def unsafeOrdering(rightType: PType): UnsafeOrdering = {
    val right = rightType.asInstanceOf[PContainer]
    val eltOrd = elementType.unsafeOrdering(
      right.elementType)

    new UnsafeOrdering {
      override def compare(o1: Long, o2: Long): Int = {
        val length1 = loadLength(o1)
        val length2 = right.loadLength(o2)

        var i = 0
        while (i < math.min(length1, length2)) {
          val leftDefined = isElementDefined(o1, i)
          val rightDefined = right.isElementDefined(o2, i)

          if (leftDefined && rightDefined) {
            val eOff1 = loadElement(o1, length1, i)
            val eOff2 = right.loadElement(o2, length2, i)
            val c = eltOrd.compare(eOff1, eOff2)
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            return c
          }
          i += 1
        }
        Integer.compare(length1, length2)
      }
    }
  }

  def hasMissingValues(srcAddress: Code[Long]): Code[Boolean] = {
    if (elementRequired)
      return const(false)

    Code.memoize(srcAddress, "pcarr_has_missing_vals_src") { srcAddress =>
      Region.containsNonZeroBits(srcAddress + lengthHeaderBytes, loadLength(srcAddress).toL)
    }
  }

  def deepPointerCopy(cb: EmitCodeBuilder, region: Value[Region], dstAddressCode: Code[Long], len: Value[Int]): Unit = {
    if (!this.elementType.fundamentalType.containsPointers) {
      return
    }

    val dstAddress = cb.newLocal[Long]("pcarr_deep_ptr_copy_dst")
    cb.assign(dstAddress, dstAddressCode)
    val currentIdx = cb.newLocal[Int]("pcarray_deep_pointer_copy_current_idx")
    val currentElementAddress = cb.newLocal[Long]("pcarray_deep_pointer_copy_current_element_addr")
    cb.forLoop(
      cb.assign(currentIdx, 0),
      currentIdx < len,
      cb.assign(currentIdx, currentIdx + 1),
      cb.ifx(isElementDefined(dstAddress, currentIdx),
        {
          cb.assign(currentElementAddress, elementOffset(dstAddress, len, currentIdx))
          this.elementType.fundamentalType match {
            // FIXME this logic needs to go before we can create new ptypes
            case t@(_: PBinary | _: PArray) =>
              t.storeAtAddress(cb, currentElementAddress, region, t.loadCheapPCode(cb, Region.loadAddress(currentElementAddress)), deepCopy = true)
            case t: PCanonicalBaseStruct =>
              t.deepPointerCopy(cb, region, currentElementAddress)
            case t: PType => throw new RuntimeException(s"Type isn't supported ${ t }")
          }
        }))
  }

  def deepPointerCopy(region: Region, dstAddress: Long) {
    if(!this.elementType.fundamentalType.containsPointers) {
      return
    }

    val numberOfElements = this.loadLength(dstAddress)
    var currentIdx = 0
    while(currentIdx < numberOfElements) {
      if(this.isElementDefined(dstAddress, currentIdx)) {
        val currentElementAddress = this.elementOffset(dstAddress, numberOfElements, currentIdx)
        this.elementType.fundamentalType match {
          case t@(_: PBinary | _: PArray) =>
            Region.storeAddress(currentElementAddress, t.copyFromAddress(region, t, Region.loadAddress(currentElementAddress), deepCopy = true))
          case t: PCanonicalBaseStruct =>
            t.deepPointerCopy(region, currentElementAddress)
          case t: PType => fatal(s"Type isn't supported ${t}")
        }
      }

      currentIdx += 1
    }
  }

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    val srcArrayT = srcPType.asInstanceOf[PArray]

    if (equalModuloRequired(srcArrayT)) {
      if (!deepCopy)
        return srcAddress

      val len = srcArrayT.loadLength(srcAddress)
      val newAddr = allocate(region, len)
      Region.copyFrom(srcAddress, newAddr, contentsByteSize(len))
      deepPointerCopy(region, newAddr)
      newAddr
    } else {
      val len = srcArrayT.loadLength(srcAddress)
      val newAddr = allocate(region, len)

      initialize(newAddr, len, setMissing = true)
      var i = 0
      val srcElementT = srcArrayT.elementType
      while (i < len) {
        if (srcArrayT.isElementDefined(srcAddress, i)) {
          setElementPresent(newAddr, i)
          elementType.unstagedStoreAtAddress(elementOffset(newAddr, len, i), region, srcElementT, srcArrayT.loadElement(srcAddress, len, i), deepCopy)
        } else
          assert(!elementType.required)

        i += 1
      }
      newAddr
    }
  }

  def sType: SContainer = SIndexablePointer(this)

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SIndexablePointerCode(SIndexablePointer(this), addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SIndexablePointer(PCanonicalArray(otherElementType, _)) if otherElementType == elementType =>
        if (deepCopy) {
          val newAddr = cb.newLocal[Long]("pcarray_store_newaddr")
          val pcInd = value.asIndexable.memoize(cb, "pcarray_store_src_sametype").asInstanceOf[SIndexablePointerSettable]
          cb.assign(newAddr, allocate(region, pcInd.length))
          cb += Region.copyFrom(pcInd.a, newAddr, contentsByteSize(pcInd.length))
          deepPointerCopy(cb, region, newAddr, pcInd.length)
          newAddr
        } else {
          value.asInstanceOf[SIndexablePointerCode].a
        }
      case _ =>
        val newAddr = cb.newLocal[Long]("pcarray_store_newaddr")
        val indexable = value.asIndexable.memoize(cb, "pcarray_store_src_difftype")
        val length = indexable.loadLength()
        cb.assign(newAddr, allocate(region, length))
        cb += stagedInitialize(newAddr, length, setMissing = false)

        val idx = cb.newLocal[Int]("pcarray_store_at_addr_idx", 0)
        cb.whileLoop(idx < length, {
          indexable
            .loadElement(cb, idx)
            .consume(
              cb,
              { cb.append(setElementMissing(newAddr, idx)) },
              { pc => elementType.storeAtAddress(cb, elementOffset(newAddr, length, idx), region, pc, deepCopy) }
            )
          cb.assign(idx, idx + 1)
        })
        newAddr
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    cb += Region.storeAddress(addr, store(cb, region, value, deepCopy))
  }


  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcArray = srcPType.asInstanceOf[PArray]
    Region.storeAddress(addr, copyFromAddress(region, srcArray, srcAddress, deepCopy))
  }

  override def deepRename(t: Type): PType = deepRenameArray(t.asInstanceOf[TArray])

  private def deepRenameArray(t: TArray): PArray =
    PCanonicalArray(this.elementType.deepRename(t.elementType), this.required)

  def constructFromElements(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean)
    (f: (EmitCodeBuilder, Value[Int]) => IEmitCode): SIndexablePointerCode = {

    val addr = cb.newLocal[Long]("pcarray_construct1_addr", allocate(region, length))
    cb += stagedInitialize(addr, length, setMissing = false)
    val i = cb.newLocal[Int]("pcarray_construct1_i", 0)

    val firstElementAddr = firstElementOffset(addr, length)
    cb.whileLoop(i < length, {
      f(cb, i).consume(cb,
        cb += setElementMissing(addr, i),
        { sc =>
          elementType.storeAtAddress(cb, elementOffsetFromFirst(firstElementAddr, i), region, sc, deepCopy = deepCopy)
        })

      cb.assign(i, i + 1)
    })

    new SIndexablePointerCode(SIndexablePointer(this), addr)
  }

  // unsafe StagedArrayBuilder-like interface that gives caller control over adding elements and finishing
  def constructFromFunctions(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean):
  (((EmitCodeBuilder, Value[Int], IEmitCode) => Unit, (EmitCodeBuilder => SIndexablePointerCode))) = {

    val addr = cb.newLocal[Long]("pcarray_construct2_addr", allocate(region, length))
    cb += stagedInitialize(addr, length, setMissing = false)
    val i = cb.newLocal[Int]("pcarray_construct2_i", 0)

    val firstElementAddr = firstElementOffset(addr, length)

    val addElement: (EmitCodeBuilder, Value[Int], IEmitCode) => Unit = { case (cb, i, iec) =>
      iec.consume(cb,
        cb += setElementMissing(addr, i),
        { sc =>
          elementType.storeAtAddress(cb, elementOffsetFromFirst(firstElementAddr, i), region, sc, deepCopy = deepCopy)
        })
    }
    val finish: EmitCodeBuilder => SIndexablePointerCode = _ => new SIndexablePointerCode(SIndexablePointer(this), addr)
    (addElement, finish)
  }

  def loadFromNested(cb: EmitCodeBuilder, addr: Code[Long]): Code[Long] = Region.loadAddress(addr)

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    val is = annotation.asInstanceOf[IndexedSeq[Annotation]]
    val valueAddress = allocate(region, is.length)
    assert(is.length >= 0)

    is match {
      case uis: UnsafeIndexedSeq => {
        this.unstagedStoreAtAddress(valueAddress, region, uis.t, uis.aoff, region.ne(uis.region))
      }
      case is: IndexedSeq[Annotation] => {
        initialize(valueAddress, is.length)
        var i = 0
        var curElementAddress = firstElementOffset(valueAddress, is.length)
        while (i < is.length) {
          if (is(i) == null) {
            setElementMissing(valueAddress, i)
          }
          else {
            elementType.unstagedStoreJavaObjectAtAddress(curElementAddress, is(i), region)
          }
          curElementAddress = nextElementAddress(curElementAddress)
          i += 1
        }
      }
    }


    valueAddress
  }

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeAddress(addr, unstagedStoreJavaObject(annotation, region))
  }

}
