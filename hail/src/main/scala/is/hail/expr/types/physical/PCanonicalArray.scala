package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.asm4s.joinpoint._
import is.hail.expr.ir.{EmitMethodBuilder, PCanonicalIndexableCode, PCode}
import is.hail.expr.types.virtual.{TArray, Type}
import is.hail.utils._

// This is a pointer array, whose byteSize is the size of its pointer
final case class PCanonicalArray(elementType: PType, required: Boolean = false) extends PArray {
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
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes + nMissingBytes(length), elementType.alignment)

  private def _elementsOffset(length: Code[Int]): Code[Long] =
    if (elementType.required)
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
    elementType.required || !Region.loadBit(aoff + lengthHeaderBytes, i)

  def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    if (elementType.required)
      true
    else
      !Region.loadBit(aoff + lengthHeaderBytes, i.toL)

  def isElementMissing(aoff: Long, i: Int): Boolean =
    !isElementDefined(aoff, i)

  def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    !isElementDefined(aoff, i)

  def setElementMissing(aoff: Long, i: Int) {
    if (!elementType.required)
      Region.setBit(aoff + lengthHeaderBytes, i)
  }

  def setElementMissing(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    if (!elementType.required)
      Region.setBit(aoff + lengthHeaderBytes, i.toL)
    else
      Code._fatal[Unit](s"Required element cannot be missing")

  def setElementPresent(aoff: Long, i: Int) {
    if (!elementType.required)
      Region.clearBit(aoff + lengthHeaderBytes, i.toLong)
  }

  def setElementPresent(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    if (!elementType.required)
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
    if (!elementType.required)
      writeMissingness(aoff, length, -1)
  }

  def clearMissingBits(aoff: Long, length: Int) {
    if (!elementType.required)
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
    if (elementType.required)
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

  def anyMissing(mb: EmitMethodBuilder[_], aoff: Code[Long]): Code[Boolean] =
    if (elementType.required)
      false
    else {
      val n = mb.newLocal[Long]()
      val ret = mb.newLocal[Boolean]()
      val ptr = mb.newLocal[Long]()
      val L = CodeLabel()
      Code.memoize(aoff,"pcarr_any_missing_aoff") { aoff =>
        Code(
          n := aoff + ((loadLength(aoff) >>> 5) * 4 + 4).toL,
          ptr := aoff + 4L,
          L,
          (ptr < n).mux(
            Region.loadInt(ptr).cne(0).mux(
              ret := true,
              Code(
                ptr := ptr + 4L,
                L.goto)),
            (Region.loadByte(ptr) >>>
              (const(32) - (loadLength(aoff) | 31))).cne(0).mux(
              ret := true,
              ret := false)),
          ret.load())
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
    if (elementType.required)
      return const(false)

    Code.memoize(srcAddress, "pcarr_has_missing_vals_src") { srcAddress =>
      Region.containsNonZeroBits(srcAddress + lengthHeaderBytes, loadLength(srcAddress).toL)
    }
  }

  def checkedConvertFrom(mb: EmitMethodBuilder[_], r: Value[Region], srcAddress: Code[Long], sourceType: PContainer, msg: String): Code[Long] = {
    assert(sourceType.elementType.isPrimitive && this.isOfType(sourceType))

    if (sourceType.elementType.required == this.elementType.required)
      return srcAddress

    val a = mb.newLocal[Long]()

    Code(
      a := srcAddress,
      sourceType.hasMissingValues(a).orEmpty(Code._fatal[Unit](msg)), {
        val newOffset = mb.genFieldThisRef[Long]()

        Code.memoize(sourceType.loadLength(a), "pcarr_check_conv_from_len") { len =>
          Code(
            newOffset := allocate(r, len),
            stagedInitialize(newOffset, len),
            Region.copyFrom(sourceType.firstElementOffset(a, len), firstElementOffset(newOffset, len), len.toL * elementByteSize),
            newOffset)
        }
      }
    )
  }

  def copyFrom(region: Region, srcOff: Long): Long = {
    val destOff = allocate(region, loadLength(srcOff))
    Region.copyFrom(srcOff,  destOff, contentsByteSize(loadLength(srcOff)))
    destOff
  }

  def copyFrom(mb: EmitMethodBuilder[_], region: Code[Region], srcOff: Code[Long]): Code[Long] = {
    val destOff = mb.genFieldThisRef[Long]()
    Code(
      destOff := allocate(region, loadLength(srcOff)),
      Region.copyFrom(srcOff, destOff, contentsByteSize(loadLength(srcOff))),
      destOff
    )
  }

  def deepPointerCopy(mb: EmitMethodBuilder[_], region: Value[Region], dstAddress: Code[Long]): Code[Unit] = {
    if (!this.elementType.fundamentalType.containsPointers) {
      return Code._empty
    }

    Code.memoize(dstAddress, "pcarr_deep_ptr_copy_dst") { dstAddress =>
      val numberOfElements = mb.newLocal[Int]()
      val currentIdx = mb.newLocal[Int]()
      val currentElementAddress = mb.genFieldThisRef[Long]()
      Code(
        currentIdx := const(0),
        numberOfElements := this.loadLength(dstAddress),
        Code.whileLoop(currentIdx < numberOfElements,
          this.isElementDefined(dstAddress, currentIdx).orEmpty(
            Code(
              currentElementAddress := this.elementOffset(dstAddress, numberOfElements, currentIdx),
              this.elementType.fundamentalType match {
                case t@(_: PBinary | _: PArray) =>
                  Region.storeAddress(currentElementAddress, t.copyFromType(mb, region, t, Region.loadAddress(currentElementAddress), deepCopy = true))
                case t: PCanonicalBaseStruct =>
                  t.deepPointerCopy(mb, region, currentElementAddress)
                case t: PType => fatal(s"Type isn't supported ${t}")
              })),
          currentIdx := currentIdx + const(1)))
    }
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
            Region.storeAddress(currentElementAddress, t.copyFromType(region, t, Region.loadAddress(currentElementAddress), deepCopy = true))
          case t: PCanonicalBaseStruct =>
            t.deepPointerCopy(region, currentElementAddress)
          case t: PType => fatal(s"Type isn't supported ${t}")
        }
      }

      currentIdx += 1
    }
  }

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    val sourceType = srcPType.asInstanceOf[PArray]
    constructOrCopy(mb, region, sourceType, srcAddress, deepCopy)
  }

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    val sourceType = srcPType.asInstanceOf[PArray]
    constructOrCopy(region, sourceType, srcAddress, deepCopy)
  }

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], deepCopy)

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val srcArray = srcPType.asInstanceOf[PArray]
    Region.storeAddress(addr, constructOrCopy(mb, region, srcArray, srcAddress, deepCopy))
  }

  private def constructOrCopy(mb: EmitMethodBuilder[_], region: Value[Region], srcArray: PArray, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    if (srcArray.isInstanceOf[PCanonicalArray] && srcArray.elementType == elementType) {
      if (deepCopy) {
        val newAddr = mb.newLocal[Long]()
        val len = mb.newLocal[Int]()
        val srcAddrVar = mb.newLocal[Long]()
        Code(
          srcAddrVar := srcAddress,
          len := srcArray.loadLength(srcAddrVar),
          newAddr := allocate(region, len),
          Region.copyFrom(srcAddrVar, newAddr, contentsByteSize(len)),
          deepPointerCopy(mb, region, newAddr),
          newAddr)
      } else
        srcAddress
    } else {
      assert(elementType.required <= srcArray.elementType.required)

      val len = mb.newLocal[Int]()
      val srcAddrVar = mb.newLocal[Long]()
      val newAddr = mb.newLocal[Long]()
      val i = mb.newLocal[Int]()

      Code(
        srcAddrVar := srcAddress,
        len := srcArray.loadLength(srcAddrVar),
        newAddr := allocate(region, len),
        stagedInitialize(newAddr, len, setMissing = true),
        i := 0,
        Code.whileLoop(i < len,
          srcArray.isElementDefined(srcAddrVar, i).orEmpty(
            Code(
              setElementPresent(newAddr, i),
              elementType.constructAtAddress(mb, elementOffset(newAddr, len, i), region, srcArray.elementType, srcArray.loadElement(srcAddrVar, len, i), deepCopy))),
          i := i + 1
        ),
        newAddr)
    }
  }

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcArray = srcPType.asInstanceOf[PArray]
    Region.storeAddress(addr, constructOrCopy(region, srcArray, srcAddress, deepCopy))
  }

  private def constructOrCopy(region: Region, srcArray: PArray, srcAddress: Long, deepCopy: Boolean): Long = {
    if (srcArray.isInstanceOf[PCanonicalArray] && srcArray.elementType == elementType) {
      if (deepCopy) {
        val len = srcArray.loadLength(srcAddress)
        val newAddr = allocate(region, len)
        Region.copyFrom(srcAddress, newAddr, contentsByteSize(len))
        deepPointerCopy(region, newAddr)
        newAddr
      } else
        srcAddress
    } else {
      val len = srcArray.loadLength(srcAddress)
      val newAddr = allocate(region, len)

      assert(elementType.required <= srcArray.elementType.required)

      initialize(newAddr, len, setMissing = true)
      var i = 0
      val srcElement = srcArray.elementType
      while (i < len) {
        if (srcArray.isElementDefined(srcAddress, i)) {
          setElementPresent(newAddr, i)
          elementType.constructAtAddress(elementOffset(newAddr, len, i), region, srcElement, srcArray.loadElement(srcAddress, len, i), deepCopy)
        }
        i += 1
      }
      newAddr
    }
  }

  override def deepRename(t: Type): PType = deepRenameArray(t.asInstanceOf[TArray])

  private def deepRenameArray(t: TArray): PArray =
    PCanonicalArray(this.elementType.deepRename(t.elementType), this.required)

  override def load(src: Code[Long]): PCode =
    new PCanonicalIndexableCode(this, Region.loadAddress(src))
}
