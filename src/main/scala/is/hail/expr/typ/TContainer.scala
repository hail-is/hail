package is.hail.expr.typ

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.utils._

object TContainer {
  def loadLength(region: Region, aoff: Long): Int =
    region.loadInt(aoff)

  def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    region.loadInt(aoff)
}

abstract class TContainer extends Type {
  def elementType: Type

  def elementByteSize: Long

  override def byteSize: Long = 8

  def contentsAlignment: Long

  override def children = Seq(elementType)

  final def loadLength(region: Region, aoff: Long): Int =
    TContainer.loadLength(region, aoff)

  final def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    TContainer.loadLength(region, aoff)

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
      case _: TArray | _: TBinary => region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: TArray | _: TBinary => region.loadAddress(off)
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

  def clearMissingBits(region: Region, aoff: Long, length: Int) {
    if (elementType.required)
      return
    val nMissingBytes = (length + 7) / 8
    var i = 0
    while (i < nMissingBytes) {
      region.storeByte(aoff + 4 + i, 0)
      i += 1
    }
  }

  def initialize(region: Region, aoff: Long, length: Int) {
    region.storeInt(aoff, length)
    clearMissingBits(region, aoff, length)
  }

  def initialize(region: Code[Region], aoff: Code[Long], length: Code[Int], a: LocalRef[Int]): Code[Unit] = {
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

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val eltOrd = elementType.unsafeOrdering(missingGreatest)

    new UnsafeOrdering {
      override def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val length1 = loadLength(r1, o1)
        val length2 = loadLength(r2, o2)

        var i = 0
        while (i < math.min(length1, length2)) {
          val leftDefined = isElementDefined(r1, o1, i)
          val rightDefined = isElementDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val eOff1 = loadElement(r1, o1, length1, i)
            val eOff2 = loadElement(r2, o2, length2, i)
            val c = eltOrd.compare(r1, eOff1, r2, eOff2)
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            if (missingGreatest)
              return c
            else
              return -c
          }
          i += 1
        }
        Integer.compare(length1, length2)
      }
    }
  }
}
