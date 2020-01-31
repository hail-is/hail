package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeOrdering}
import is.hail.asm4s.{Code, MethodBuilder}

abstract class ComplexPType extends PType {
  val representation: PType

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override def fundamentalType: PType = representation.fundamentalType

  override def containsPointers: Boolean = representation.containsPointers

  def storeShallowAtOffset(dstAddress: Code[Long], valueAddress: Code[Long]): Code[Unit] =
    this.representation.storeShallowAtOffset(dstAddress, valueAddress)

  def storeShallowAtOffset(dstAddress: Long, valueAddress: Long) {
    this.representation.storeShallowAtOffset(dstAddress, valueAddress)
  }

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] = {
    assert(this isOfType srcPType)

    val srcRepPType = srcPType.asInstanceOf[ComplexPType].representation

    this.representation.copyFromType(mb, region, srcRepPType, srcAddress, forceDeep)
  }

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], forceDeep)

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long = {
    assert(this isOfType srcPType)

    val srcRepPType = srcPType.asInstanceOf[ComplexPType].representation

    this.representation.copyFromType(region, srcRepPType, srcAddress, forceDeep)
  }
}
