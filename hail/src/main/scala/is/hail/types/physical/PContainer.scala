package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode}

abstract class PContainer extends PIterable {
  override def containsPointers: Boolean = true

  def elementByteSize: Long

  def contentsAlignment: Long

  def loadLength(aoff: Long): Int

  def loadLength(aoff: Code[Long])(implicit line: LineNumber): Code[Int]

  def storeLength(aoff: Code[Long], length: Code[Int])(implicit line: LineNumber): Code[Unit]

  def nMissingBytes(len: Code[Int])(implicit line: LineNumber): Code[Int]

  def lengthHeaderBytes: Long

  def elementsOffset(length: Int): Long

  def elementsOffset(length: Code[Int])(implicit line: LineNumber): Code[Long]

  def isElementMissing(aoff: Long, i: Int): Boolean

  def isElementDefined(aoff: Long, i: Int): Boolean

  def isElementMissing(aoff: Code[Long], i: Code[Int])(implicit line: LineNumber): Code[Boolean]

  def isElementDefined(aoff: Code[Long], i: Code[Int])(implicit line: LineNumber): Code[Boolean]

  def setElementMissing(aoff: Long, i: Int)

  def setElementMissing(aoff: Code[Long], i: Code[Int])(implicit line: LineNumber): Code[Unit]

  def setElementPresent(aoff: Long, i: Int)

  def setElementPresent(aoff: Code[Long], i: Code[Int])(implicit line: LineNumber): Code[Unit]

  def firstElementOffset(aoff: Long, length: Int): Long

  def elementOffset(aoff: Long, length: Int, i: Int): Long

  def elementOffset(aoff: Long, i: Int): Long

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int])(implicit line: LineNumber): Code[Long]

  def elementOffset(aoff: Code[Long], i: Code[Int])(implicit line: LineNumber): Code[Long]

  def firstElementOffset(aoff: Code[Long], length: Code[Int])(implicit line: LineNumber): Code[Long]

  def firstElementOffset(aoff: Code[Long])(implicit line: LineNumber): Code[Long]

  def copyFrom(region: Region, srcOff: Long): Long

  def copyFrom(mb: EmitMethodBuilder[_], region: Code[Region], srcOff: Code[Long])(implicit line: LineNumber): Code[Long]

  def loadElement(aoff: Long, length: Int, i: Int): Long

  def loadElement(aoff: Long, i: Int): Long

  def loadElement(aoff: Code[Long], i: Code[Int])(implicit line: LineNumber): Code[Long]

  def loadElement(aoff: Code[Long], length: Code[Int], i: Code[Int])(implicit line: LineNumber): Code[Long]

  def allocate(region: Region, length: Int): Long

  def allocate(region: Code[Region], length: Code[Int])(implicit line: LineNumber): Code[Long]

  def setAllMissingBits(aoff: Long, length: Int)

  def clearMissingBits(aoff: Long, length: Int)

  def initialize(aoff: Long, length: Int, setMissing: Boolean = false)

  def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false)(implicit line: LineNumber): Code[Unit]

  def zeroes(region: Region, length: Int): Long

  def zeroes(mb: EmitMethodBuilder[_], region: Value[Region], length: Code[Int])(implicit line: LineNumber): Code[Long]

  def forEach(mb: EmitMethodBuilder[_], aoff: Code[Long], body: Code[Long] => Code[Unit])(implicit line: LineNumber): Code[Unit]

  def hasMissingValues(sourceOffset: Code[Long])(implicit line: LineNumber): Code[Boolean]

  def nextElementAddress(currentOffset: Long): Long

  def nextElementAddress(currentOffset: Code[Long])(implicit line: LineNumber): Code[Long]
}

abstract class PIndexableValue extends PValue {
  def loadLength(): Value[Int]

  def isElementMissing(i: Code[Int])(implicit line: LineNumber): Code[Boolean]

  def isElementDefined(i: Code[Int])(implicit line: LineNumber): Code[Boolean] =
    !isElementMissing(i)

  def loadElement(cb: EmitCodeBuilder, i: Code[Int])(implicit line: LineNumber): IEmitCode
}

abstract class PIndexableCode extends PCode {
  def pt: PContainer

  def loadLength()(implicit line: LineNumber): Code[Int]

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PIndexableValue

  def memoizeField(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PIndexableValue
}
