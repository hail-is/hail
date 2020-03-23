package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeOrdering}
import is.hail.asm4s.{Code, MethodBuilder, Value}
import is.hail.expr.ir.EmitMethodBuilder

abstract class ComplexPType extends PType {
  val representation: PType

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override def fundamentalType: PType = representation.fundamentalType

  override def containsPointers: Boolean = representation.containsPointers

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    assert(this isOfType srcPType)

    val srcRepPType = srcPType.asInstanceOf[ComplexPType].representation

    this.representation.copyFromType(mb, region, srcRepPType, srcAddress, deepCopy)
  }

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    this.representation.copyFromTypeAndStackValue(mb, region, srcPType.asInstanceOf[ComplexPType].representation, stackValue, deepCopy)

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    assert(this isOfType srcPType)

    val srcRepPType = srcPType.asInstanceOf[ComplexPType].representation

    this.representation.copyFromType(region, srcRepPType, srcAddress, deepCopy)
  }

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] =
    this.representation.constructAtAddress(mb, addr, region, srcPType.fundamentalType, srcAddress, deepCopy)

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    this.representation.constructAtAddress(addr, region, srcPType.fundamentalType, srcAddress, deepCopy)

  override def constructAtAddressFromValue(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, src: Code[_], deepCopy: Boolean): Code[Unit] =
    this.representation.constructAtAddressFromValue(mb, addr, region, srcPType.fundamentalType, src, deepCopy)
}