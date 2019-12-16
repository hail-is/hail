package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.asm4s.joinpoint._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._

final case class PCanonicalArray(elementType: PType, required: Boolean = false) extends PArray {
  def _asIdent = s"array_of_${elementType.asIdent}"
  def _toPretty = s"Array[$elementType]"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("array<")
    elementType.pyString(sb)
    sb.append('>')
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Array[")
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

  def copy(elementType: PType = this.elementType, required: Boolean = this.required): PCanonicalArray = PCanonicalArray(elementType, required)

  def loadLength(region: Region, aoff: Long): Int =
    loadLength(aoff)

  def loadLength(aoff: Long): Int =
    Region.loadInt(aoff)

  def loadLength(aoff: Code[Long]): Code[Int] =
    Region.loadInt(aoff)

  def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    loadLength(aoff)

  def storeLength(region: Region, aoff: Long, length: Int): Unit =
    storeLength(aoff, length)

  def storeLength(aoff: Long, length: Int): Unit =
    Region.storeInt(aoff, length)

  def storeLength(region: Code[Region], aoff: Code[Long], length: Code[Int]): Code[Unit] =
    storeLength(aoff, length)

  def storeLength(aoff: Code[Long], length: Code[Int]): Code[Unit] =
    Region.storeInt(aoff, length)

  def nMissingBytes(len: Code[Int]): Code[Int] = UnsafeUtils.packBitsToBytes(len)

  def nMissingBytes(len: Int): Int = UnsafeUtils.packBitsToBytes(len)

  private def contentsByteSize(length: Int): Long =
    elementsOffset(length) + length * elementByteSize

  private def contentsByteSize(length: Code[Int]): Code[Long] = {
    elementsOffset(length) + length.toL * elementByteSize
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

  def isElementDefined(region: Region, aoff: Long, i: Int): Boolean = isElementDefined(aoff, i)

  def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    if (elementType.required)
      true
    else
      !Region.loadBit(aoff + lengthHeaderBytes, i.toL)

  def isElementDefined(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    isElementDefined(aoff, i)

  def isElementMissing(region: Region, aoff: Long, i: Int): Boolean =
    !isElementDefined(aoff, i)

  def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    !isElementDefined(aoff, i)

  def isElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    isElementMissing(aoff, i)

  def setElementMissing(region: Region, aoff: Long, i: Int) {
    assert(!elementType.required)
    Region.setBit(aoff + lengthHeaderBytes, i)
  }

  def setElementMissing(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    Region.setBit(aoff + lengthHeaderBytes, i.toL)

  def setElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] =
    setElementMissing(aoff, i)

  def setElementPresent(region: Region, aoff: Long, i: Int) {
    assert(!elementType.required)
    Region.clearBit(aoff + lengthHeaderBytes, i)
  }

  def setElementPresent(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    Region.clearBit(aoff + lengthHeaderBytes, i.toL)

  def setElementPresent(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] =
    setElementPresent(aoff, i)

  def firstElementOffset(aoff: Long, length: Int): Long =
    aoff + elementsOffset(length)

  def firstElementOffset(aoff: Code[Long], length: Code[Int]): Code[Long] =
    aoff + elementsOffset(length)

  def firstElementOffset(aoff: Code[Long]): Code[Long] =
    firstElementOffset(aoff, loadLength(aoff))

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    firstElementOffset(aoff, length) + i * elementByteSize

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    firstElementOffset(aoff, length) + i.toL * const(elementByteSize)

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long =
    elementOffset(aoff, loadLength(region, aoff), i)

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    elementOffset(aoff, loadLength(region, aoff), i)

  def loadElement(aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Region, aoff: Long, length: Int, i: Int): Long = loadElement(aoff, length, i)

  def loadElement(region: Region, aoff: Long, i: Int): Long = loadElement(aoff, loadLength(aoff), i)

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] = {
    val length = loadLength(region, aoff)
    loadElement(region, aoff, length, i)
  }

  def allocate(region: Region, length: Int): Long = {
    region.allocate(contentsAlignment, contentsByteSize(length))
  }

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    region.allocate(contentsAlignment, contentsByteSize(length))

  private def writeMissingness(region: Region, aoff: Long, length: Int, value: Byte) {
    Region.setMemory(aoff + lengthHeaderBytes, nMissingBytes(length), value)
  }

  def setAllMissingBits(region: Region, aoff: Long, length: Int) {
    if (elementType.required)
      return
    writeMissingness(region, aoff, length, -1)
  }

  def clearMissingBits(region: Region, aoff: Long, length: Int) {
    if (elementType.required)
      return
    writeMissingness(region, aoff, length, 0)
  }

  def initialize(region: Region, aoff: Long, length: Int, setMissing: Boolean = false) {
    Region.storeInt(aoff, length)
    if (setMissing)
      setAllMissingBits(region, aoff, length)
    else
      clearMissingBits(region, aoff, length)
  }

  def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Code[Unit] = {
    if (elementType.required)
      Region.storeInt(aoff, length)
    else
      Code(
        Region.storeInt(aoff, length),
        Region.setMemory(aoff + const(lengthHeaderBytes), nMissingBytes(length).toL, const(if (setMissing) (-1).toByte else 0.toByte)))
  }

  def zeroes(region: Region, length: Int): Long = {
    require(elementType.isNumeric)
    val aoff = allocate(region, length)
    initialize(region, aoff, length)
    Region.setMemory(aoff + elementsOffset(length), length * elementByteSize, 0.toByte)
    aoff
  }

  def zeroes(mb: MethodBuilder, region: Code[Region], length: Code[Int]): Code[Long] = {
    require(elementType.isNumeric)
    val aoff = mb.newLocal[Long]
    Code(
      aoff := allocate(region, length),
      stagedInitialize(aoff, length),
      Region.setMemory(aoff + elementsOffset(length), length.toL * elementByteSize, 0.toByte),
      aoff)
  }

  def anyMissing(mb: MethodBuilder, aoff: Code[Long]): Code[Boolean] =
    if (elementType.required)
      false
    else {
      val n = mb.newLocal[Long]
      JoinPoint.CallCC[Code[Boolean]] { (jb, ret) =>
        val loop = jb.joinPoint[Code[Long]](mb)
        loop.define { ptr =>
          (ptr < n).mux(
            Region.loadInt(ptr).cne(0).mux(
              ret(true),
              loop(ptr + 4L)),
            (Region.loadByte(ptr) >>>
              (const(32) - (loadLength(aoff) | 31))).cne(0).mux(
              ret(true),
              ret(false)))
        }
        Code(
          n := aoff + ((loadLength(aoff) >>> 5) * 4 + 4).toL,
          loop(aoff + 4L))
      }
    }

  def forEach(mb: MethodBuilder, region: Code[Region], aoff: Code[Long], body: Code[Long] => Code[Unit]): Code[Unit] = {
    val i = mb.newLocal[Int]
    val n = mb.newLocal[Int]
    Code(
      n := loadLength(aoff),
      i := 0,
      Code.whileLoop(i < n,
        isElementDefined(aoff, i).mux(
          body(loadElement(region, aoff, n, i)),
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
      override def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val length1 = loadLength(r1, o1)
        val length2 = right.loadLength(r2, o2)

        var i = 0
        while (i < math.min(length1, length2)) {
          val leftDefined = isElementDefined(r1, o1, i)
          val rightDefined = right.isElementDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val eOff1 = loadElement(r1, o1, length1, i)
            val eOff2 = right.loadElement(r2, o2, length2, i)
            val c = eltOrd.compare(r1, eOff1, r2, eOff2)
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

  def hasMissingValues(sourceOffset: Code[Long]): Code[Boolean] = {
    if(elementType.required) {
      return const(false)
    }

    Region.containsNonZeroBits(sourceOffset + lengthHeaderBytes, loadLength(sourceOffset).toL)
  }

  def checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], sourceOffset: Code[Long], sourceType: PContainer, msg: String): Code[Long] = {
    assert(sourceType.elementType.isPrimitive && this.isOfType(sourceType))

    if (sourceType.elementType.required == this.elementType.required) {
      return sourceOffset
    }

    Code(
      sourceType.hasMissingValues(sourceOffset).orEmpty(Code._fatal(msg)), {
        val newOffset = mb.newField[Long]
        val len = sourceType.loadLength(sourceOffset)

        Code(
          newOffset := allocate(r, len),
          stagedInitialize(newOffset, len),
          Region.copyFrom(sourceType.firstElementOffset(sourceOffset, len), firstElementOffset(newOffset, len), len.toL * elementByteSize),
          newOffset
        )
      }
    )
  }

  def copyFrom(region: Region, srcOff: Long): Long = {
    val destOff = allocate(region, loadLength(srcOff))
    Region.copyFrom(srcOff,  destOff, contentsByteSize(loadLength(srcOff)))
    destOff
  }

  def copyFrom(mb: MethodBuilder, region: Code[Region], srcOff: Code[Long]): Code[Long] = {
    val destOff = mb.newField[Long]
    Code(
      destOff := allocate(region, loadLength(srcOff)),
      Region.copyFrom(srcOff, destOff, contentsByteSize(loadLength(srcOff))),
      destOff
    )
  }
}
