package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
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

  def printDebug(cb: EmitCodeBuilder, addr: Value[Long]): Unit = {
    val a = cb.newLocal[Long]("a", addr)
    val l = cb.memoize(loadLength(addr))
    cb.println("array header:")
    cb.whileLoop(a < firstElementOffset(addr, l), {
      cb.println("  ", Code.invokeStatic1[java.lang.Long, Long, String]("toHexString", Region.loadLong(a)),
                 " (", Code.invokeStatic1[java.lang.Integer, Int, String]("toHexString", Region.loadInt(a)),
                 " ", Code.invokeStatic1[java.lang.Integer, Int, String]("toHexString", Region.loadInt(a+4)),
                 ")")
      cb.assign(a, a + 8)
    })
  }

  private val elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  private val contentsAlignment: Long = elementType.alignment.max(4)

  private val lengthHeaderBytes: Long = 4

  override val byteSize: Long = 8

  def setRequired(required: Boolean): PCanonicalArray =
    if (required == this.required) this else PCanonicalArray(elementType, required)

  def loadLength(aoff: Long): Int =
    Region.loadInt(aoff)

  def loadLength(aoff: Code[Long]): Code[Int] =
    Region.loadInt(aoff)

  def storeLength(aoff: Long, length: Int): Unit =
    Region.storeInt(aoff, length)

  def storeLength(cb: EmitCodeBuilder, aoff: Code[Long], length: Code[Int]): Unit =
    cb += Region.storeInt(aoff, length)

  def nMissingBytes(len: Code[Int]): Code[Int] = UnsafeUtils.packBitsToBytes(len)

  def nMissingBytes(len: Int): Int = UnsafeUtils.packBitsToBytes(len)

  def contentsByteSize(length: Int): Long =
    elementsOffset(length) + length * elementByteSize

  def contentsByteSize(length: Code[Int]): Code[Long] = {
    Code.memoize(length, "contentsByteSize_arr_len") { length =>
      elementsOffset(length) + length.toL * elementByteSize
    }
  }

  private def _elementsOffset(length: Int): Long =
    if (elementRequired)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, contentsAlignment)
    else
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes + nMissingBytes(length), contentsAlignment)

  private def _elementsOffset(length: Code[Int]): Code[Long] =
    if (elementRequired)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, contentsAlignment)
    else
      UnsafeUtils.roundUpAlignment(nMissingBytes(length).toL + lengthHeaderBytes, contentsAlignment)

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

  def missingBytesOffset: Long = lengthHeaderBytes

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

  def setElementMissing(cb: EmitCodeBuilder, aoff: Code[Long], i: Code[Int]): Unit =
    if (!elementRequired)
      cb += Region.setBit(aoff + lengthHeaderBytes, i.toL)
    else
      cb._fatal(s"Required element cannot be missing")

  def setElementPresent(aoff: Long, i: Int) {
    if (!elementRequired)
      Region.clearBit(aoff + lengthHeaderBytes, i.toLong)
  }

  def setElementPresent(cb: EmitCodeBuilder, aoff: Code[Long], i: Code[Int]): Unit =
    if (!elementRequired)
      cb += Region.clearBit(aoff + lengthHeaderBytes, i.toL)

  def firstElementOffset(aoff: Long, length: Int): Long =
    aoff + elementsOffset(length)

  def firstElementOffset(aoff: Long): Long =
    aoff + elementsOffset(loadLength(aoff))

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

  override def incrementElementOffset(currentOffset: Long, increment: Int): Long =
    currentOffset + increment * elementByteSize

  override def incrementElementOffset(currentOffset: Code[Long], increment: Code[Int]): Code[Long] =
    currentOffset + increment.toL * elementByteSize

  override def pastLastElementOffset(aoff: Long, length: Int): Long =
    firstElementOffset(aoff, length) + length * elementByteSize

  override def pastLastElementOffset(aoff: Code[Long], length: Value[Int]): Code[Long] =
    firstElementOffset(aoff, length) + length.toL * elementByteSize

  def nextElementAddress(currentOffset: Long) =
    currentOffset + elementByteSize

  def nextElementAddress(currentOffset: Code[Long]) =
    currentOffset + elementByteSize

  def loadElement(aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.unstagedLoadFromNested(off)
  }

  def loadElement(aoff: Long, i: Int): Long = loadElement(aoff, loadLength(aoff), i)

  def loadElement(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, length, i)
    elementType.loadFromNested(off)
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

  override def stagedInitialize(cb: EmitCodeBuilder, aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Unit = {
    if (elementRequired)
      cb += Region.storeInt(aoff, length)
    else {
      val aoffMem = cb.memoize[Long](aoff)
      val lengthMem = cb.memoize[Int](length)
      cb += Region.storeInt(aoffMem, lengthMem)
      cb += Region.setMemory(aoffMem + const(lengthHeaderBytes), nMissingBytes(lengthMem).toL, const(if (setMissing) (-1).toByte else 0.toByte))
    }
  }

  def zeroes(region: Region, length: Int): Long = {
    require(elementType.isNumeric)
    val aoff = allocate(region, length)
    initialize(aoff, length)
    Region.setMemory(aoff + elementsOffset(length), length * elementByteSize, 0.toByte)
    aoff
  }

  def zeroes(cb: EmitCodeBuilder, region: Value[Region], length: Code[Int]): Code[Long] = {
    require(elementType.isNumeric)
    val lengthMem = cb.memoize(length)
    val aoff = cb.memoize[Long](allocate(region, lengthMem))
    stagedInitialize(cb, aoff, lengthMem)
    cb += Region.setMemory(aoff + elementsOffset(lengthMem), lengthMem.toL * elementByteSize, 0.toByte)
    aoff
  }

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering =
    unsafeOrdering(sm, this)

  override def unsafeOrdering(sm: HailStateManager, rightType: PType): UnsafeOrdering = {
    val right = rightType.asInstanceOf[PContainer]
    val eltOrd = elementType.unsafeOrdering(sm, right.elementType)

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
    if (!this.elementType.containsPointers) {
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
          this.elementType.storeAtAddress(cb, currentElementAddress, region, this.elementType.loadCheapSCode(cb, this.elementType.loadFromNested(currentElementAddress)), true)
        }))
  }

  def deepPointerCopy(sm: HailStateManager, region: Region, dstAddress: Long) {
    if(!this.elementType.containsPointers) {
      return
    }

    val numberOfElements = this.loadLength(dstAddress)
    var currentIdx = 0
    while(currentIdx < numberOfElements) {
      if(this.isElementDefined(dstAddress, currentIdx)) {
        val currentElementAddress = this.elementOffset(dstAddress, numberOfElements, currentIdx)
        val currentElementAddressFromNested = this.elementType.unstagedLoadFromNested(currentElementAddress)
        this.elementType.unstagedStoreAtAddress(sm, currentElementAddress, region, this.elementType, currentElementAddressFromNested, true)
      }

      currentIdx += 1
    }
  }

  def _copyFromAddress(sm: HailStateManager, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    val srcArrayT = srcPType.asInstanceOf[PArray]

    if (equalModuloRequired(srcArrayT)) {
      if (!deepCopy)
        return srcAddress

      val len = srcArrayT.loadLength(srcAddress)
      val newAddr = allocate(region, len)
      Region.copyFrom(srcAddress, newAddr, contentsByteSize(len))
      deepPointerCopy(sm, region, newAddr)
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
          elementType.unstagedStoreAtAddress(sm, elementOffset(newAddr, len, i), region, srcElementT, srcArrayT.loadElement(srcAddress, len, i), deepCopy)
        } else
          assert(!elementType.required)

        i += 1
      }
      newAddr
    }
  }

  def sType: SIndexablePointer = SIndexablePointer(setRequired(false))

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SIndexablePointerValue = {
    val a = cb.memoize(addr)
    val length = cb.memoize(loadLength(a))
    val offset = cb.memoize(firstElementOffset(a, length))
    new SIndexablePointerValue(sType, a, length, offset)
  }

  def storeContentsAtAddress(cb: EmitCodeBuilder, addr: Value[Long], region: Value[Region], indexable: SIndexableValue, deepCopy: Boolean): Unit = {
    val length = indexable.loadLength()
    indexable.st match {
      case SIndexablePointer(PCanonicalArray(otherElementType, _)) if otherElementType == elementType =>
          cb += Region.copyFrom(indexable.asInstanceOf[SIndexablePointerValue].a, addr, contentsByteSize(length))
          deepPointerCopy(cb, region, addr, length)
      case SIndexablePointer(otherType@PCanonicalArray(otherElementType, _)) if otherElementType.equalModuloRequired(elementType) =>
        // other is optional, constructing required
        if (elementType.required) {
          cb.ifx(indexable.hasMissingValues(cb),
            cb._fatal("tried to copy array with missing values to array of required elements"))
        }
        stagedInitialize(cb, addr, indexable.loadLength(), setMissing = false)

        cb += Region.copyFrom(otherType.firstElementOffset(indexable.asInstanceOf[SIndexablePointerValue].a), this.firstElementOffset(addr), length.toL * otherType.elementByteSize)
        if (deepCopy)
          deepPointerCopy(cb, region, addr, length)
      case _ =>
        stagedInitialize(cb, addr, length, setMissing = false)

        val idx = cb.newLocal[Int]("pcarray_store_at_addr_idx", 0)
        cb.whileLoop(idx < length, {
          indexable
            .loadElement(cb, idx)
            .consume(
              cb,
              { setElementMissing(cb, addr, idx) },
              { pc => elementType.storeAtAddress(cb, elementOffset(addr, length, idx), region, pc, deepCopy) }
            )
          cb.assign(idx, idx + 1)
        })
    }
  }

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): Value[Long] = {
    assert(value.st.virtualType.isInstanceOf[TArray])
    value.st match {
      case SIndexablePointer(PCanonicalArray(otherElementType, _)) if otherElementType == elementType && !deepCopy =>
        value.asInstanceOf[SIndexablePointerValue].a
      case _ =>
        val idxValue = value.asIndexable
        val newAddr = cb.memoize(allocate(region, idxValue.loadLength()))
        storeContentsAtAddress(cb, newAddr, region, idxValue, deepCopy)
        newAddr
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SValue, deepCopy: Boolean): Unit = {
    cb += Region.storeAddress(addr, store(cb, region, value, deepCopy))
  }


  def unstagedStoreAtAddress(sm: HailStateManager, addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcArray = srcPType.asInstanceOf[PArray]
    Region.storeAddress(addr, copyFromAddress(sm, region, srcArray, srcAddress, deepCopy))
  }

  override def deepRename(t: Type): PType = deepRenameArray(t.asInstanceOf[TArray])

  private def deepRenameArray(t: TArray): PArray =
    PCanonicalArray(this.elementType.deepRename(t.elementType), this.required)

  def padWithMissing(cb: EmitCodeBuilder, region: Value[Region], oldLength: Value[Int], newLength: Value[Int], srcAddress: Value[Long]): Value[Long] = {
    val dstAddress = cb.memoize(allocate(region, newLength))
    stagedInitialize(cb, dstAddress, newLength, setMissing = true)
    cb += Region.copyFrom(srcAddress + lengthHeaderBytes, dstAddress + lengthHeaderBytes, nMissingBytes(oldLength).toL)
    cb += Region.copyFrom(
      srcAddress + elementsOffset(oldLength),
      dstAddress + elementsOffset(newLength),
      oldLength.toL * elementByteSize)
    dstAddress
  }

  def constructFromElements(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean)
    (f: (EmitCodeBuilder, Value[Int]) => IEmitCode): SIndexablePointerValue = {

    val addr = cb.newLocal[Long]("pcarray_construct1_addr", allocate(region, length))
    stagedInitialize(cb, addr, length, setMissing = false)
    val i = cb.newLocal[Int]("pcarray_construct1_i", 0)

    val firstElementAddr = cb.newLocal[Long]("pcarray_construct1_firstelementaddr", firstElementOffset(addr, length))
    cb.whileLoop(i < length, {
      f(cb, i).consume(cb,
        setElementMissing(cb, addr, i),
        { sc =>
          elementType.storeAtAddress(cb, elementOffsetFromFirst(firstElementAddr, i), region, sc, deepCopy = deepCopy)
        })

      cb.assign(i, i + 1)
    })

    new SIndexablePointerValue(sType, addr, length, firstElementAddr)
  }

  // unsafe StagedArrayBuilder-like interface that gives caller control over pushing elements and finishing
  def constructFromFunctions(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean):
  (((EmitCodeBuilder, IEmitCode) => Unit, (EmitCodeBuilder => SIndexablePointerValue))) = {

    val addr = cb.newLocal[Long]("pcarray_construct2_addr", allocate(region, length))
    stagedInitialize(cb, addr, length, setMissing = false)
    val currentElementIndex = cb.newLocal[Int]("pcarray_construct2_current_idx", 0)
    val firstElementAddress = cb.newLocal[Long]("pcarray_construct2_first_addr", firstElementOffset(addr, length))
    val currentElementAddress = cb.newLocal[Long]("pcarray_construct2_current_addr", firstElementAddress)

    val push: (EmitCodeBuilder, IEmitCode) => Unit = { case (cb, iec) =>
      iec.consume(cb,
        setElementMissing(cb, addr, currentElementIndex),
        { sc =>
          elementType.storeAtAddress(cb, currentElementAddress, region, sc, deepCopy = deepCopy)
        })
        cb.assign(currentElementIndex, currentElementIndex + 1)
        cb.assign(currentElementAddress, currentElementAddress + elementByteSize)
    }
    val finish: EmitCodeBuilder => SIndexablePointerValue = { (cb: EmitCodeBuilder) =>
      cb.ifx(currentElementIndex.cne(length), cb._fatal("PCanonicalArray.constructFromFunctions push was called the wrong number of times: len=",
        length.toS, ", calls=", currentElementIndex.toS))
      new SIndexablePointerValue(sType, addr, length, firstElementAddress)
    }
    (push, finish)
  }

  def constructFromIndicesUnsafe(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean):
  (((EmitCodeBuilder, Value[Int], IEmitCode) => Unit, (EmitCodeBuilder => SIndexablePointerValue))) = {

    val addr = cb.newLocal[Long]("pcarray_construct2_addr", allocate(region, length))
    stagedInitialize(cb, addr, length, setMissing = false)
    val firstElementAddress = cb.newLocal[Long]("pcarray_construct2_first_addr", firstElementOffset(addr, length))

    val push: (EmitCodeBuilder, Value[Int], IEmitCode) => Unit = {
      case (cb, idx, iec) =>
        iec.consume(cb,
          setElementMissing(cb, addr, idx),
          { sc =>
            setElementPresent(cb, addr, idx)
            elementType.storeAtAddress(cb, firstElementAddress + idx.toL * elementByteSize, region, sc, deepCopy = deepCopy)
          })
    }
    val finish: EmitCodeBuilder => SIndexablePointerValue = { (cb: EmitCodeBuilder) =>
      new SIndexablePointerValue(sType, addr, length, firstElementAddress)
    }
    (push, finish)
  }


  def loadFromNested(addr: Code[Long]): Code[Long] = Region.loadAddress(addr)

  override def unstagedLoadFromNested(addr: Long): Long = Region.loadAddress(addr)

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region): Long = {
    val is = annotation.asInstanceOf[IndexedSeq[Annotation]]
    val valueAddress = allocate(region, is.length)
    assert(is.length >= 0)

    initialize(valueAddress, is.length)
    var i = 0
    var curElementAddress = firstElementOffset(valueAddress, is.length)
    while (i < is.length) {
      if (is(i) == null) {
        setElementMissing(valueAddress, i)
      }
      else {
        elementType.unstagedStoreJavaObjectAtAddress(sm, curElementAddress, is(i), region)
      }
      curElementAddress = nextElementAddress(curElementAddress)
      i += 1
    }

    valueAddress
  }

  override def unstagedStoreJavaObjectAtAddress(sm: HailStateManager, addr: Long, annotation: Annotation, region: Region): Unit = {
     annotation match {
       case uis: UnsafeIndexedSeq => this.unstagedStoreAtAddress(sm, addr, region, uis.t, uis.aoff, region.ne(uis.region))
       case is: IndexedSeq[Annotation] => Region.storeAddress(addr, unstagedStoreJavaObject(sm, annotation, region))
     }
  }

  override def copiedType: PType = {
    val copiedElement = elementType.copiedType
    if (copiedElement.eq(elementType))
      this
    else
      PCanonicalArray(copiedElement, required)
  }



  def forEachDefined(cb: EmitCodeBuilder, aoff: Value[Long])(f: (EmitCodeBuilder, Value[Int], SValue) => Unit) {
    val length = cb.memoize(loadLength(aoff))
    val elementsAddress = cb.memoize(firstElementOffset(aoff))
    val idx = cb.newLocal[Int]("foreach_pca_idx", 0)
    val elementPtr = cb.newLocal[Long]("foreach_pca_elt_ptr", elementsAddress)
    val et = elementType
    cb.whileLoop(idx < length, {
      cb.ifx(isElementMissing(aoff, idx),
        {}, // do nothing,
        {
          val elt = et.loadCheapSCode(cb, et.loadFromNested(elementPtr))
          f(cb, idx, elt)
        })
      cb.assign(idx, idx + 1)
      cb.assign(elementPtr, elementPtr + elementByteSize)
    })
  }
}
