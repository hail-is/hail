package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeOrdering}
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.expr.ir.EmitMethodBuilder

trait PArrayBackedContainer extends PContainer {
  val arrayRep: PArray

  lazy val elementByteSize = arrayRep.elementByteSize

  lazy val contentsAlignment = arrayRep.contentsAlignment

  lazy val lengthHeaderBytes: Long = arrayRep.lengthHeaderBytes

  override lazy val byteSize: Long = arrayRep.byteSize

  override lazy val fundamentalType = PCanonicalArray(elementType.fundamentalType, required)

  def loadLength(region: Region, aoff: Long): Int =
    arrayRep.loadLength(region, aoff)

  def loadLength(aoff: Long): Int =
    arrayRep.loadLength(aoff)

  def loadLength(aoff: Code[Long]): Code[Int] =
    arrayRep.loadLength(aoff)

  def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    arrayRep.loadLength(region, aoff)

  def storeLength(region: Region, aoff: Long, length: Int): Unit =
    arrayRep.storeLength(region, aoff, length)

  def storeLength(aoff: Code[Long], length: Code[Int]): Code[Unit] =
    arrayRep.storeLength(aoff, length)

  def storeLength(region: Code[Region], aoff: Code[Long], length: Code[Int]): Code[Unit] =
    arrayRep.storeLength(region, aoff, length)

  def nMissingBytes(len: Code[Int]): Code[Int] =
    arrayRep.nMissingBytes(len)

  def elementsOffset(length: Int): Long =
    arrayRep.elementsOffset(length)

  def elementsOffset(length: Code[Int]): Code[Long] =
    arrayRep.elementsOffset(length)

  def isElementDefined(aoff: Long, i: Int): Boolean =
    arrayRep.isElementDefined(aoff, i)

  def isElementDefined(region: Region, aoff: Long, i: Int): Boolean =
    arrayRep.isElementDefined(region, aoff, i)

  def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    arrayRep.isElementDefined(aoff, i)

  def isElementDefined(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    arrayRep.isElementDefined(region, aoff, i)

  def isElementMissing(region: Region, aoff: Long, i: Int): Boolean =
    arrayRep.isElementMissing(region, aoff, i)

  def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    arrayRep.isElementMissing(aoff, i)

  def isElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    arrayRep.isElementMissing(region, aoff, i)

  def setElementMissing(region: Region, aoff: Long, i: Int) =
    arrayRep.setElementMissing(region, aoff, i)

  def setElementMissing(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    arrayRep.setElementMissing(aoff, i)

  def setElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] =
    arrayRep.setElementMissing(region, aoff, i)

  def setElementPresent(region: Region, aoff: Long, i: Int): Unit =
    arrayRep.setElementPresent(region, aoff, i)

  def setElementPresent(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    arrayRep.setElementPresent(aoff, i)

  def setElementPresent(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] =
    arrayRep.setElementPresent(region, aoff, i)

  def firstElementOffset(aoff: Long, length: Int): Long =
    arrayRep.firstElementOffset(aoff, length)

  def firstElementOffset(aoff: Code[Long], length: Code[Int]): Code[Long] =
    arrayRep.firstElementOffset(aoff, length)

  def firstElementOffset(aoff: Code[Long]): Code[Long] =
    arrayRep.firstElementOffset(aoff)

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    arrayRep.elementOffset(aoff, length, i)

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    arrayRep.elementOffset(aoff, length, i)

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long =
    arrayRep.elementOffsetInRegion(region, aoff, i)

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    arrayRep.elementOffsetInRegion(region, aoff, i)

  def loadElement(aoff: Long, length: Int, i: Int): Long =
    arrayRep.loadElement(aoff, length, i)

  def loadElement(region: Region, aoff: Long, length: Int, i: Int): Long =
    arrayRep.loadElement(region, aoff, length, i)

  def loadElement(region: Region, aoff: Long, i: Int): Long =
    arrayRep.loadElement(region, aoff, i)

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    arrayRep.loadElement(region, aoff, length, i)

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    arrayRep.loadElement(region, aoff, i)

  def allocate(region: Region, length: Int): Long =
    arrayRep.allocate(region, length)

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    arrayRep.allocate(region, length)

  def setAllMissingBits(region: Region, aoff: Long, length: Int) =
    arrayRep.setAllMissingBits(region, aoff, length)

  def clearMissingBits(region: Region, aoff: Long, length: Int) =
    arrayRep.clearMissingBits(region, aoff, length)

  def initialize(region: Region, aoff: Long, length: Int, setMissing: Boolean = false) =
    arrayRep.initialize(region, aoff, length, setMissing)

  def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Code[Unit] =
    arrayRep.stagedInitialize(aoff, length, setMissing)

  def zeroes(region: Region, length: Int): Long =
    arrayRep.zeroes(region, length)

  def zeroes(mb: MethodBuilder, region: Code[Region], length: Code[Int]): Code[Long] =
    arrayRep.zeroes(mb, region, length)

  def anyMissing(mb: MethodBuilder, aoff: Code[Long]): Code[Boolean] =
    arrayRep.anyMissing(mb, aoff)

  def forEach(mb: MethodBuilder, region: Code[Region], aoff: Code[Long], body: Code[Long] => Code[Unit]): Code[Unit] =
    arrayRep.forEach(mb, region, aoff, body)

  def hasMissingValues(sourceOffset: Code[Long]): Code[Boolean] =
    arrayRep.hasMissingValues(sourceOffset)

  def checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], sourceOffset: Code[Long], sourceType: PContainer, msg: String): Code[Long] =
    arrayRep.checkedConvertFrom(mb, r, sourceOffset, sourceType, msg)

  def copyFrom(region: Region, srcOff: Long): Long =
    arrayRep.copyFrom(region, srcOff)

  def copyFrom(mb: MethodBuilder, region: Code[Region], srcOff: Code[Long]): Code[Long] =
    arrayRep.copyFrom(mb, region, srcOff)

  override def unsafeOrdering: UnsafeOrdering = unsafeOrdering(this)

  override def unsafeOrdering(rightType: PType): UnsafeOrdering = arrayRep.unsafeOrdering(rightType)
}
