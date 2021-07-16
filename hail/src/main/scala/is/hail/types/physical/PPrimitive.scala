package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.utils._

trait PPrimitive extends PType {
  def byteSize: Long

  def _construct(mb: EmitMethodBuilder[_], region: Value[Region], pc: SCode): SCode = pc

  override def containsPointers: Boolean = false

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    val newAddr = cb.newLocal[Long]("pprimitive_store_addr", region.allocate(alignment, byteSize))
    storeAtAddress(cb, newAddr, region, value, deepCopy)
    newAddr
  }


  override def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    storePrimitiveAtAddress(cb, addr, value)
  }

  def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SCode): Unit

  def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    val addr = region.allocate(this.byteSize)
    unstagedStoreJavaObjectAtAddress(addr, annotation, region)
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
}
