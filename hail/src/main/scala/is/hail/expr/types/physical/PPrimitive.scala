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

  def setRequired(required: Boolean) = {
    if (required == this.required)
      this
    else
      this match {
        case PBoolean(_) => PBoolean(required)
        case PInt32(_) => PInt32(required)
        case PInt64(_) => PInt64(required)
        case PFloat32(_) => PFloat32(required)
        case PFloat64(_) => PFloat64(required)
      }
  }
}
