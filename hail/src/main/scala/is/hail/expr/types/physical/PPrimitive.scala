package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.utils._

trait PPrimitive extends PType {
  def byteSize: Long

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long = {
    if (forceDeep) {
      val addr = region.allocate(byteSize, byteSize)
      constructAtAddress(addr, region, srcPType, srcAddress, forceDeep)
      addr
    } else srcAddress
  }

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] = {
    if (forceDeep) {
      val addr = mb.newLocal[Long]
      Code(
        addr := region.allocate(byteSize, byteSize),
        constructAtAddress(mb, addr, region, srcPType, srcAddress, forceDeep),
        addr
      )
    } else srcAddress
  }

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    stackValue

  def constructAtAddress(mb: MethodBuilder, addr: Code[Long], region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Unit] = {
    assert(srcPType.isOfType(this))
    Region.copyFrom(srcAddress, addr, byteSize)
  }

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Unit = {
    assert(srcPType.isOfType(this))
    Region.copyFrom(srcAddress, addr, byteSize)
  }
}
