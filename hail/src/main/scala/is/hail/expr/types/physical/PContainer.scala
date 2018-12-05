package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.cxx
import is.hail.utils._

object PContainer {
  def loadLength(region: Region, aoff: Long): Int =
    region.loadInt(aoff)

  def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    region.loadInt(aoff)
}

abstract class PContainer extends PType {
  def elementType: PType

  def elementByteSize: Long

  override def byteSize: Long = 8

  def contentsAlignment: Long

  override def children = FastSeq(elementType)

  final def loadLength(region: Region, aoff: Long): Int =
    PContainer.loadLength(region, aoff)

  final def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    PContainer.loadLength(region, aoff)

  def _elementsOffset(length: Int): Long =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(4, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(4 + ((length + 7) >>> 3), elementType.alignment)

  def _elementsOffset(length: Code[Int]): Code[Long] =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(4, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(((length.toL + 7L) >>> 3) + 4L, elementType.alignment)

  var elementsOffsetTable: Array[Long] = _

  def elementsOffset(length: Int): Long = {
    if (elementsOffsetTable == null)
      elementsOffsetTable = Array.tabulate[Long](10)(i => _elementsOffset(i))

    if (length < 10)
      elementsOffsetTable(length)
    else
      _elementsOffset(length)
  }

  def elementsOffset(length: Code[Int]): Code[Long] = {
    // FIXME: incorporate table, maybe?
    _elementsOffset(length)
  }

  def contentsByteSize(length: Int): Long =
    elementsOffset(length) + length * elementByteSize

  def contentsByteSize(length: Code[Int]): Code[Long] = {
    elementsOffset(length) + length.toL * elementByteSize
  }

  def isElementMissing(region: Region, aoff: Long, i: Int): Boolean =
    !isElementDefined(region, aoff, i)

  def isElementDefined(region: Region, aoff: Long, i: Int): Boolean =
    elementType.required || !region.loadBit(aoff + 4, i)

  def isElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    !isElementDefined(region, aoff, i)

  def isElementDefined(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    if (elementType.required)
      true
    else
      !region.loadBit(aoff + 4, i.toL)

  def setElementMissing(region: Region, aoff: Long, i: Int) {
    assert(!elementType.required)
    region.setBit(aoff + 4, i)
  }

  def setElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] = {
    region.setBit(aoff + 4L, i.toL)
  }

  def setElementPresent(region: Region, aoff: Long, i: Int) {
    assert(!elementType.required)
    region.clearBit(aoff + 4, i)
  }

  def setElementPresent(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] = {
    region.clearBit(aoff + 4L, i.toL)
  }

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    aoff + elementsOffset(length) + i * elementByteSize

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long =
    elementOffset(aoff, loadLength(region, aoff), i)

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    aoff + elementsOffset(length) + i.toL * const(elementByteSize)

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    elementOffset(aoff, loadLength(region, aoff), i)

  def loadElement(region: Region, aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Region, aoff: Long, i: Int): Long =
    loadElement(region, aoff, region.loadInt(aoff), i)

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    loadElement(region, aoff, region.loadInt(aoff), i)

  def allocate(region: Region, length: Int): Long = {
    region.allocate(contentsAlignment, contentsByteSize(length))
  }

  // FIXME expose intrinsic to just memset this
  private def writeMissingness(region: Region, aoff: Long, length: Int, value: Byte) {
    val nMissingBytes = (length + 7) / 8
    var i = 0
    while (i < nMissingBytes) {
      region.storeByte(aoff + 4 + i, value)
      i += 1
    }
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
    region.storeInt(aoff, length)
    if (setMissing)
      setAllMissingBits(region, aoff, length)
    else
      clearMissingBits(region, aoff, length)
  }

  def initialize(region: Code[Region], aoff: Code[Long], length: Code[Int], a: Settable[Int]): Code[Unit] = {
    var c = region.storeInt(aoff, length)
    if (elementType.required)
      return c
    Code(
      c,
      a.store((length + 7) >>> 3),
      Code.whileLoop(a > 0,
        Code(
          a.store(a - 1),
          region.storeByte(aoff + 4L + a.toL, const(0))
        )
      )
    )
  }

  override def unsafeOrdering(): UnsafeOrdering =
    unsafeOrdering(this)

  override def unsafeOrdering(rightType: PType): UnsafeOrdering = {
    require(this.isOfType(rightType))

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

  def cxxImpl: String = {
    elementType match {
      case _: PStruct =>
        s"ArrayAddrImpl<${ elementType.required },${ elementType.byteSize },${ elementType.alignment }>"
      case _ =>
        s"ArrayLoadImpl<${ cxx.typeToCXXType(elementType) },${ elementType.required },${ elementType.byteSize },${ elementType.alignment }>"
    }
  }

  def cxxLoadLength(a: cxx.Code): cxx.Code = {
    s"$cxxImpl::load_length($a)"
  }

  def cxxIsElementMissing(a: cxx.Code, i: cxx.Code): cxx.Code = {
    s"$cxxImpl::is_element_missing($a, $i)"
  }

  def cxxNMissingBytes(len: cxx.Code): cxx.Code = {
    if (elementType.required)
      "0"
    else
      s"(($len + 7) >> 3)"
  }

  def cxxContentsByteSize(len: cxx.Code): cxx.Code = {
    s"${ cxxElementsOffset(len) } + $len * ${ UnsafeUtils.arrayElementSize(elementType) }"
  }

  def cxxElementsOffset(len: cxx.Code): cxx.Code = {
    s"$cxxImpl::elements_offset($len)"
  }

  def cxxElementAddress(a: cxx.Code, i: cxx.Code): cxx.Code = {
    s"$cxxImpl::element_address($a, $i)"
  }

  def cxxLoadElement(a: cxx.Code, i: cxx.Code): cxx.Code = {
    s"$cxxImpl::load_element($a, $i)"
  }
}
