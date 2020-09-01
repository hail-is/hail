package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.utils._

trait PPrimitive extends PType {
  def byteSize: Long

  def _construct(mb: EmitMethodBuilder[_], region: Value[Region], pc: PCode): PCode = pc

  override def containsPointers: Boolean = false

  override def encodableType: PType = this

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    if (!deepCopy)
      return srcAddress

    // FIXME push down
    val addr = region.allocate(byteSize, byteSize)
    unstagedStoreAtAddress(addr, region, srcPType, srcAddress, deepCopy)
    addr
  }


  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    assert(srcPType.isOfType(this))
    Region.copyFrom(srcAddress, addr, byteSize)
  }

  def copyFromType(cb: EmitCodeBuilder, region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    assert(this.isOfType(srcPType))
    if (deepCopy) {
      val addr = cb.newLocal[Long]("primitive_copyfromtype_addr", region.allocate(alignment, byteSize))
      storeAtAddress(cb, addr, region, srcPType.getPointerTo(cb, srcAddress), deepCopy)
      addr.load()
    } else srcAddress
  }

  def store(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): Code[Long] = {
    val newAddr = cb.newLocal[Long]("pprimitive_store_addr", region.allocate(alignment, byteSize))
    storeAtAddress(cb, newAddr, region, value, deepCopy)
    newAddr
  }


  override def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: PCode, deepCopy: Boolean): Unit = {
    storePrimitiveAtAddress(cb, addr, value)
  }

  def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: PCode): Unit

  override def getPointerTo(cb: EmitCodeBuilder, addr: Code[Long]): PCode = {
    sType.loadFrom(cb, null, this, addr)
  }

  def setRequired(required: Boolean): PPrimitive = {
    if (required == this.required)
      this
    else
      this match {
        case _: PBoolean => PBoolean(required)
        case _: PInt32 => PInt32(required)
        case _: PInt64 => PInt64(required)
        case _: PFloat32 => PFloat32(required)
        case _: PFloat64 => PFloat64(required)
      }
  }
}
