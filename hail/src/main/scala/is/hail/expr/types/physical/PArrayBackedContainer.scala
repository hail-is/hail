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

  def loadLength(aoff: Long): Int =
    arrayRep.loadLength(aoff)

  def loadLength(aoff: Code[Long]): Code[Int] =
    arrayRep.loadLength(aoff)

  def storeLength(aoff: Code[Long], length: Code[Int]): Code[Unit] =
    arrayRep.storeLength(aoff, length)

  def nMissingBytes(len: Code[Int]): Code[Int] =
    arrayRep.nMissingBytes(len)

  def elementsOffset(length: Int): Long =
    arrayRep.elementsOffset(length)

  def elementsOffset(length: Code[Int]): Code[Long] =
    arrayRep.elementsOffset(length)

  def isElementDefined(aoff: Long, i: Int): Boolean =
    arrayRep.isElementDefined(aoff, i)

  def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    arrayRep.isElementDefined(aoff, i)

  def isElementMissing(aoff: Long, i: Int): Boolean =
    arrayRep.isElementMissing(aoff, i)

  def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    arrayRep.isElementMissing(aoff, i)

  def setElementMissing(aoff: Long, i: Int) =
    arrayRep.setElementMissing(aoff, i)

  def setElementMissing(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    arrayRep.setElementMissing(aoff, i)

  def setElementPresent(aoff: Long, i: Int) {
      arrayRep.setElementPresent(aoff, i)
  }

  def setElementPresent(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    arrayRep.setElementPresent(aoff, i)

  def firstElementOffset(aoff: Long, length: Int): Long =
    arrayRep.firstElementOffset(aoff, length)

  def firstElementOffset(aoff: Code[Long], length: Code[Int]): Code[Long] =
    arrayRep.firstElementOffset(aoff, length)

  def firstElementOffset(aoff: Code[Long]): Code[Long] =
    arrayRep.firstElementOffset(aoff)

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    arrayRep.elementOffset(aoff, length, i)

  def elementOffset(aoff: Long, i: Int): Long =
    arrayRep.elementOffset(aoff, loadLength(aoff), i)

  def elementOffset(aoff: Code[Long], i: Code[Int]): Code[Long] =
    arrayRep.elementOffset(aoff, loadLength(aoff), i)

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    arrayRep.elementOffset(aoff, length, i)

  def loadElement(aoff: Long, length: Int, i: Int): Long =
    arrayRep.loadElement(aoff, length, i)

  def loadElement(aoff: Long, i: Int): Long =
    arrayRep.loadElement(aoff, loadLength(aoff), i)

  def loadElement(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    arrayRep.loadElement(aoff, length, i)

  def loadElement(aoff: Code[Long], i: Code[Int]): Code[Long] =
    arrayRep.loadElement(aoff, loadLength(aoff), i)

  def allocate(region: Region, length: Int): Long =
    arrayRep.allocate(region, length)

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    arrayRep.allocate(region, length)

  def setAllMissingBits(aoff: Long, length: Int) =
    arrayRep.setAllMissingBits(aoff, length)

  def clearMissingBits(aoff: Long, length: Int) =
    arrayRep.clearMissingBits(aoff, length)

  def initialize(aoff: Long, length: Int, setMissing: Boolean = false) =
    arrayRep.initialize(aoff, length, setMissing)

  def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Code[Unit] =
    arrayRep.stagedInitialize(aoff, length, setMissing)

  def zeroes(region: Region, length: Int): Long =
    arrayRep.zeroes(region, length)

  def zeroes(mb: MethodBuilder, region: Code[Region], length: Code[Int]): Code[Long] =
    arrayRep.zeroes(mb, region, length)

  def anyMissing(mb: MethodBuilder, aoff: Code[Long]): Code[Boolean] =
    arrayRep.anyMissing(mb, aoff)

  def forEach(mb: MethodBuilder, aoff: Code[Long], body: Code[Long] => Code[Unit]): Code[Unit] =
    arrayRep.forEach(mb, aoff, body)

  def hasMissingValues(sourceOffset: Code[Long]): Code[Boolean] =
    arrayRep.hasMissingValues(sourceOffset)

  def checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], sourceOffset: Code[Long], sourceType: PContainer, msg: String): Code[Long] =
    arrayRep.checkedConvertFrom(mb, r, sourceOffset, sourceType, msg)

  def copyFrom(region: Region, srcOff: Long): Long =
    arrayRep.copyFrom(region, srcOff)

  def copyFrom(mb: MethodBuilder, region: Code[Region], srcOff: Code[Long]): Code[Long] =
    arrayRep.copyFrom(mb, region, srcOff)

  override def unsafeOrdering: UnsafeOrdering =
    unsafeOrdering(this)

  override def unsafeOrdering(rightType: PType): UnsafeOrdering =
    arrayRep.unsafeOrdering(rightType)

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] =
    this.arrayRep.copyFromType(mb, region, srcPType.asInstanceOf[PArrayBackedContainer].arrayRep, srcAddress, forceDeep)

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], forceDeep)

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long =
    this.arrayRep.copyFromType(region, srcPType.asInstanceOf[PArrayBackedContainer].arrayRep, srcAddress, forceDeep)

  def nextElementAddress(currentOffset: Long) =
    arrayRep.nextElementAddress(currentOffset)

  def nextElementAddress(currentOffset: Code[Long]) =
    arrayRep.nextElementAddress(currentOffset)

  def constructAtAddress(mb: MethodBuilder, addr: Code[Long], region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Unit] =
    arrayRep.constructAtAddress(mb, addr, region, srcPType.asInstanceOf[PArrayBackedContainer].arrayRep, srcAddress, forceDeep)

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Unit =
    arrayRep.constructAtAddress(addr, region, srcPType.asInstanceOf[PArrayBackedContainer].arrayRep, srcAddress, forceDeep)
}
