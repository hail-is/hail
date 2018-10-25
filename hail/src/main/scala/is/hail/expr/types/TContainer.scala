package is.hail.expr.types

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

  override def children = FastSeq(elementType)

  final def loadLength(region: Region, aoff: Long): Int =
    TContainer.loadLength(region, aoff)

  def _elementsOffset(length: Int): Long =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(4, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(4 + ((length + 7) >>> 3), elementType.alignment)

  var elementsOffsetTable: Array[Long] = _

  def elementsOffset(length: Int): Long = {
    if (elementsOffsetTable == null)
      elementsOffsetTable = Array.tabulate[Long](10)(i => _elementsOffset(i))

    if (length < 10)
      elementsOffsetTable(length)
    else
      _elementsOffset(length)
  }

  def isElementMissing(region: Region, aoff: Long, i: Int): Boolean =
    !isElementDefined(region, aoff, i)

  def isElementDefined(region: Region, aoff: Long, i: Int): Boolean =
    elementType.required || !region.loadBit(aoff + 4, i)

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    aoff + elementsOffset(length) + i * elementByteSize

  def loadElement(region: Region, aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: TArray | _: TBinary => region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Region, aoff: Long, i: Int): Long =
    loadElement(region, aoff, region.loadInt(aoff), i)
}
