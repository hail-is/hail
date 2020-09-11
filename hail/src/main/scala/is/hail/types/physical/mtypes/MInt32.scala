package is.hail.types.physical.mtypes

import is.hail.annotations.Region
import is.hail.asm4s.{Value, _}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SInt32Value, SValue}

case object MInt32 extends MType {
  def byteSize: Long = 4

  def alignment: Long = 4

  def storeFromSValue(cb: EmitCodeBuilder, memory: UninitializedMValue, value: SValue): MValue = {
    cb.append(Region.storeInt(memory.addr, value.asInstanceOf[SInt32Value].intValue))
    memory.toMValue
  }

  def storeFromMValue(cb: EmitCodeBuilder, memory: UninitializedMValue, value: MValue): MValue = {
    assert(value.typ == this)
    cb.append(Region.storeInt(memory.addr, Region.loadInt(value.addr)))
    memory.toMValue
  }

  def coerceOrCopyMValue(cb: EmitCodeBuilder, region: Value[Region], value: MValue, deep: Boolean): MValue = {
    assert(value.typ == this)
    value
  }
}
