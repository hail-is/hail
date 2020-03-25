package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder

abstract class PContainer extends PIterable {
  override def containsPointers: Boolean = true

  def elementByteSize: Long

  def contentsAlignment: Long

  def loadLength(aoff: Long): Int

  def loadLength(aoff: Code[Long]): Code[Int]

  def storeLength(aoff: Code[Long], length: Code[Int]): Code[Unit]

  def nMissingBytes(len: Code[Int]): Code[Int]

  def lengthHeaderBytes: Long

  def elementsOffset(length: Int): Long

  def elementsOffset(length: Code[Int]): Code[Long]

  def isElementMissing(aoff: Long, i: Int): Boolean

  def isElementDefined(aoff: Long, i: Int): Boolean

  def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean]

  def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean]

  def setElementMissing(aoff: Long, i: Int)

  def setElementMissing(aoff: Code[Long], i: Code[Int]): Code[Unit]

  def setElementPresent(aoff: Long, i: Int)

  def setElementPresent(aoff: Code[Long], i: Code[Int]): Code[Unit]

  def firstElementOffset(aoff: Long, length: Int): Long

  def elementOffset(aoff: Long, length: Int, i: Int): Long

  def elementOffset(aoff: Long, i: Int): Long

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long]

  def elementOffset(aoff: Code[Long], i: Code[Int]): Code[Long]

  def firstElementOffset(aoff: Code[Long], length: Code[Int]): Code[Long]

  def firstElementOffset(aoff: Code[Long]): Code[Long]

  def copyFrom(region: Region, srcOff: Long): Long

  def copyFrom(mb: EmitMethodBuilder[_], region: Code[Region], srcOff: Code[Long]): Code[Long]

  def loadElement(aoff: Long, length: Int, i: Int): Long

  def loadElement(aoff: Long, i: Int): Long

  def loadElement(aoff: Code[Long], i: Code[Int]): Code[Long]

  def loadElement(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long]

  def allocate(region: Region, length: Int): Long

  def allocate(region: Code[Region], length: Code[Int]): Code[Long]

  def setAllMissingBits(aoff: Long, length: Int)

  def clearMissingBits(aoff: Long, length: Int)

  def initialize(aoff: Long, length: Int, setMissing: Boolean = false)

  def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Code[Unit]

  def zeroes(region: Region, length: Int): Long

  def zeroes(mb: EmitMethodBuilder[_], region: Value[Region], length: Code[Int]): Code[Long]

  def anyMissing(mb: EmitMethodBuilder[_], aoff: Code[Long]): Code[Boolean]

  def forEach(mb: EmitMethodBuilder[_], aoff: Code[Long], body: Code[Long] => Code[Unit]): Code[Unit]

  def hasMissingValues(sourceOffset: Code[Long]): Code[Boolean]

  def checkedConvertFrom(mb: EmitMethodBuilder[_], r: Value[Region], sourceOffset: Code[Long], sourceType: PContainer, msg: String): Code[Long]

  def nextElementAddress(currentOffset: Long): Long

  def nextElementAddress(currentOffset: Code[Long]): Code[Long]
}
