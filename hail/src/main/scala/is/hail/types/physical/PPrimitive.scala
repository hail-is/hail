package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.asm4s.{Code, _}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.{SCode, SValue}
import is.hail.utils._

trait PPrimitive extends PType {
  def byteSize: Long

  def _construct(mb: EmitMethodBuilder[_], region: Value[Region], pc: SValue): SValue = pc

  override def containsPointers: Boolean = false

  def _copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long = {
    if (!deepCopy)
      return srcAddress

    // FIXME push down
    val addr = region.allocate(byteSize, byteSize)
    unstagedStoreAtAddress(sm, addr, region, srcPType, srcAddress, deepCopy)
    addr
  }

  def unstagedStoreAtAddress(
    sm: HailStateManager,
    addr: Long,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Unit = {
    assert(srcPType.isOfType(this))
    Region.copyFrom(srcAddress, addr, byteSize)
  }

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean)
    : Value[Long] = {
    val newAddr = cb.memoize(region.allocate(alignment, byteSize))
    storeAtAddress(cb, newAddr, region, value, deepCopy)
    newAddr
  }

  override def storeAtAddress(
    cb: EmitCodeBuilder,
    addr: Code[Long],
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): Unit =
    storePrimitiveAtAddress(cb, addr, value)

  def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SValue): Unit

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region)
    : Long = {
    val addr = region.allocate(this.byteSize)
    unstagedStoreJavaObjectAtAddress(sm, addr, annotation, region)
    addr
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

  def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def unstagedLoadFromNested(addr: Long): Long = addr

  override def copiedType: PType = this
}
