package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeOrdering}
import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerCode}
import is.hail.types.physical.stypes.interfaces.SContainer

trait PArrayBackedContainer extends PContainer {
  val arrayRep: PArray

  lazy val elementByteSize = arrayRep.elementByteSize

  lazy val contentsAlignment = arrayRep.contentsAlignment

  lazy val lengthHeaderBytes: Long = arrayRep.lengthHeaderBytes

  override lazy val byteSize: Long = arrayRep.byteSize

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

  def zeroes(mb: EmitMethodBuilder[_], region: Value[Region], length: Code[Int]): Code[Long] =
    arrayRep.zeroes(mb, region, length)

  def forEach(mb: EmitMethodBuilder[_], aoff: Code[Long], body: Code[Long] => Code[Unit]): Code[Unit] =
    arrayRep.forEach(mb, aoff, body)

  def hasMissingValues(sourceOffset: Code[Long]): Code[Boolean] =
    arrayRep.hasMissingValues(sourceOffset)

  override def unsafeOrdering: UnsafeOrdering =
    unsafeOrdering(this)

  override def unsafeOrdering(rightType: PType): UnsafeOrdering =
    arrayRep.unsafeOrdering(rightType)

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    arrayRep.copyFromAddress(region, srcPType.asInstanceOf[PArrayBackedContainer].arrayRep, srcAddress, deepCopy)

  def nextElementAddress(currentOffset: Long): Long =
    arrayRep.nextElementAddress(currentOffset)

  def nextElementAddress(currentOffset: Code[Long]): Code[Long] =
    arrayRep.nextElementAddress(currentOffset)

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    arrayRep.unstagedStoreAtAddress(addr, region, srcPType.asInstanceOf[PArrayBackedContainer].arrayRep, srcAddress, deepCopy)

  def sType: SContainer = SIndexablePointer(this)

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SIndexablePointerCode(SIndexablePointer(this), addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = arrayRep.store(cb, region, value, deepCopy)

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit =
    arrayRep.storeAtAddress(cb, addr, region, value, deepCopy)

  def loadFromNested(addr: Code[Long]): Code[Long] = arrayRep.loadFromNested(addr)

  override def unstagedLoadFromNested(addr: Long): Long = arrayRep.unstagedLoadFromNested(addr)

  def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeAddress(addr, unstagedStoreJavaObject(annotation, region))
  }
}
