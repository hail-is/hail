package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeOrdering}
import is.hail.asm4s.{Code, Value}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}

trait PArrayBackedContainer extends PContainer {
  val arrayRep: PArray

  override lazy val byteSize: Long = arrayRep.byteSize

  override def loadLength(aoff: Long): Int =
    arrayRep.loadLength(aoff)

  override def loadLength(aoff: Code[Long]): Code[Int] =
    arrayRep.loadLength(aoff)

  override def storeLength(cb: EmitCodeBuilder, aoff: Code[Long], length: Code[Int]): Unit =
    arrayRep.storeLength(cb, aoff, length)

  override def elementsOffset(length: Int): Long =
    arrayRep.elementsOffset(length)

  override def elementsOffset(length: Code[Int]): Code[Long] =
    arrayRep.elementsOffset(length)

  override def isElementDefined(aoff: Long, i: Int): Boolean =
    arrayRep.isElementDefined(aoff, i)

  override def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    arrayRep.isElementDefined(aoff, i)

  override def isElementMissing(aoff: Long, i: Int): Boolean =
    arrayRep.isElementMissing(aoff, i)

  override def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    arrayRep.isElementMissing(aoff, i)

  override def setElementMissing(aoff: Long, i: Int) =
    arrayRep.setElementMissing(aoff, i)

  override def setElementMissing(cb: EmitCodeBuilder, aoff: Code[Long], i: Code[Int]): Unit =
    arrayRep.setElementMissing(cb, aoff, i)

  override def setElementPresent(aoff: Long, i: Int): Unit =
    arrayRep.setElementPresent(aoff, i)

  override def setElementPresent(cb: EmitCodeBuilder, aoff: Code[Long], i: Code[Int]): Unit =
    arrayRep.setElementPresent(cb, aoff, i)

  override def firstElementOffset(aoff: Long, length: Int): Long =
    arrayRep.firstElementOffset(aoff, length)

  override def firstElementOffset(aoff: Code[Long], length: Code[Int]): Code[Long] =
    arrayRep.firstElementOffset(aoff, length)

  override def firstElementOffset(aoff: Code[Long]): Code[Long] =
    arrayRep.firstElementOffset(aoff)

  override def elementOffset(aoff: Long, length: Int, i: Int): Long =
    arrayRep.elementOffset(aoff, length, i)

  override def elementOffset(aoff: Long, i: Int): Long =
    arrayRep.elementOffset(aoff, loadLength(aoff), i)

  override def elementOffset(aoff: Code[Long], i: Code[Int]): Code[Long] =
    arrayRep.elementOffset(aoff, loadLength(aoff), i)

  override def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    arrayRep.elementOffset(aoff, length, i)

  override def loadElement(aoff: Long, length: Int, i: Int): Long =
    arrayRep.loadElement(aoff, length, i)

  override def loadElement(aoff: Long, i: Int): Long =
    arrayRep.loadElement(aoff, loadLength(aoff), i)

  override def loadElement(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    arrayRep.loadElement(aoff, length, i)

  override def loadElement(aoff: Code[Long], i: Code[Int]): Code[Long] =
    arrayRep.loadElement(aoff, loadLength(aoff), i)

  override def allocate(region: Region, length: Int): Long =
    arrayRep.allocate(region, length)

  override def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    arrayRep.allocate(region, length)

  override def setAllMissingBits(aoff: Long, length: Int) =
    arrayRep.setAllMissingBits(aoff, length)

  override def clearMissingBits(aoff: Long, length: Int) =
    arrayRep.clearMissingBits(aoff, length)

  def contentsByteSize(length: Int): Long =
    arrayRep.contentsByteSize(length)

  def contentsByteSize(length: Code[Int]): Code[Long] =
    arrayRep.contentsByteSize(length)

  override def initialize(aoff: Long, length: Int, setMissing: Boolean = false) =
    arrayRep.initialize(aoff, length, setMissing)

  override def stagedInitialize(
    cb: EmitCodeBuilder,
    aoff: Code[Long],
    length: Code[Int],
    setMissing: Boolean = false,
  ): Unit =
    arrayRep.stagedInitialize(cb, aoff, length, setMissing)

  override def zeroes(region: Region, length: Int): Long =
    arrayRep.zeroes(region, length)

  override def zeroes(cb: EmitCodeBuilder, region: Value[Region], length: Code[Int]): Code[Long] =
    arrayRep.zeroes(cb, region, length)

  override def hasMissingValues(sourceOffset: Code[Long]): Code[Boolean] =
    arrayRep.hasMissingValues(sourceOffset)

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering =
    unsafeOrdering(sm, this)

  override def unsafeOrdering(sm: HailStateManager, rightType: PType): UnsafeOrdering =
    arrayRep.unsafeOrdering(sm, rightType)

  override def _copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long =
    arrayRep.copyFromAddress(
      sm,
      region,
      srcPType.asInstanceOf[PArrayBackedContainer].arrayRep,
      srcAddress,
      deepCopy,
    )

  override def nextElementAddress(currentOffset: Long): Long =
    arrayRep.nextElementAddress(currentOffset)

  override def nextElementAddress(currentOffset: Code[Long]): Code[Long] =
    arrayRep.nextElementAddress(currentOffset)

  override def incrementElementOffset(currentOffset: Long, increment: Int): Long =
    arrayRep.incrementElementOffset(currentOffset, increment)

  override def incrementElementOffset(currentOffset: Code[Long], increment: Code[Int]): Code[Long] =
    arrayRep.incrementElementOffset(currentOffset, increment)

  override def pastLastElementOffset(aoff: Long, length: Int): Long =
    arrayRep.pastLastElementOffset(aoff, length)

  override def pastLastElementOffset(aoff: Code[Long], length: Value[Int]): Code[Long] =
    arrayRep.pastLastElementOffset(aoff, length)

  override def unstagedStoreAtAddress(
    sm: HailStateManager,
    addr: Long,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Unit =
    arrayRep.unstagedStoreAtAddress(
      sm,
      addr,
      region,
      srcPType.asInstanceOf[PArrayBackedContainer].arrayRep,
      srcAddress,
      deepCopy,
    )

  override def sType: SIndexablePointer =
    SIndexablePointer(setRequired(false).asInstanceOf[PArrayBackedContainer])

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SIndexablePointerValue = {
    val a = cb.memoize(addr)
    val length = cb.memoize(loadLength(a))
    val elementsAddr = cb.memoize(firstElementOffset(a, length))
    new SIndexablePointerValue(sType, a, length, elementsAddr)
  }

  override def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean)
    : Value[Long] =
    arrayRep.store(cb, region, value.asIndexable.castToArray(cb), deepCopy)

  override def storeAtAddress(
    cb: EmitCodeBuilder,
    addr: Code[Long],
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): Unit =
    arrayRep.storeAtAddress(cb, addr, region, value.asIndexable.castToArray(cb), deepCopy)

  override def loadFromNested(addr: Code[Long]): Code[Long] = arrayRep.loadFromNested(addr)

  override def unstagedLoadFromNested(addr: Long): Long = arrayRep.unstagedLoadFromNested(addr)

  override def unstagedStoreJavaObjectAtAddress(
    sm: HailStateManager,
    addr: Long,
    annotation: Annotation,
    region: Region,
  ): Unit =
    Region.storeAddress(addr, unstagedStoreJavaObject(sm, annotation, region))
}
