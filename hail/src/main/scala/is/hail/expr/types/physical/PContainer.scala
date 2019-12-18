package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder

abstract class PContainer extends PIterable {
  override def containsPointers: Boolean = true

  def elementByteSize: Long

  def contentsAlignment: Long

  def loadLength(region: Region, aoff: Long): Int

  def loadLength(aoff: Long): Int

  def loadLength(aoff: Code[Long]): Code[Int]

  def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int]

  def storeLength(region: Region, aoff: Long, length: Int): Unit

  def storeLength(aoff: Code[Long], length: Code[Int]): Code[Unit]

  def storeLength(region: Code[Region], aoff: Code[Long], length: Code[Int]): Code[Unit]

  def nMissingBytes(len: Code[Int]): Code[Int]

  def lengthHeaderBytes: Long

  def elementsOffset(length: Int): Long

  def elementsOffset(length: Code[Int]): Code[Long]

  def isElementMissing(region: Region, aoff: Long, i: Int): Boolean

  def isElementDefined(aoff: Long, i: Int): Boolean

  def isElementDefined(region: Region, aoff: Long, i: Int): Boolean

  def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean]

  def isElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean]

  def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean]

  def isElementDefined(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean]

  def setElementMissing(region: Region, aoff: Long, i: Int)

  def setElementMissing(aoff: Code[Long], i: Code[Int]): Code[Unit]

  def setElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit]

  def setElementPresent(region: Region, aoff: Long, i: Int)

  def setElementPresent(aoff: Code[Long], i: Code[Int]): Code[Unit]

  def setElementPresent(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit]

  def firstElementOffset(aoff: Long, length: Int): Long

  def elementOffset(aoff: Long, length: Int, i: Int): Long

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long]

  def firstElementOffset(aoff: Code[Long], length: Code[Int]): Code[Long]

  def firstElementOffset(aoff: Code[Long]): Code[Long]

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long]

  def loadElement(aoff: Long, length: Int, i: Int): Long

  def copyFrom(region: Region, srcOff: Long): Long

  def copyFrom(mb: MethodBuilder, region: Code[Region], srcOff: Code[Long]): Code[Long]

  def loadElement(region: Region, aoff: Long, length: Int, i: Int): Long

  def loadElement(region: Region, aoff: Long, i: Int): Long

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long]

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long]

  def allocate(region: Region, length: Int): Long

  def allocate(region: Code[Region], length: Code[Int]): Code[Long]

  def setAllMissingBits(region: Region, aoff: Long, length: Int)

  def clearMissingBits(region: Region, aoff: Long, length: Int)

  def initialize(region: Region, aoff: Long, length: Int, setMissing: Boolean = false)

  def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Code[Unit]

  def zeroes(region: Region, length: Int): Long

  def zeroes(mb: MethodBuilder, region: Code[Region], length: Code[Int]): Code[Long]

  def anyMissing(mb: MethodBuilder, aoff: Code[Long]): Code[Boolean]

  def forEach(mb: MethodBuilder, region: Code[Region], aoff: Code[Long], body: Code[Long] => Code[Unit]): Code[Unit]

  def hasMissingValues(sourceOffset: Code[Long]): Code[Boolean]

  def checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], sourceOffset: Code[Long], sourceType: PContainer, msg: String): Code[Long]
}
