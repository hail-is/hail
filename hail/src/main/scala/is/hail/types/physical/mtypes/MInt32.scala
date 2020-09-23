package is.hail.types.physical.mtypes

import is.hail.annotations.Region
import is.hail.asm4s.{Value, _}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SInt32, SInt32Value, SType, SValue}

case object MInt32 extends MType {
  def byteSize: Long = 4

  def alignment: Long = 4

  def storeFromSValue(cb: EmitCodeBuilder, memory: UninitializedMValue, value: SValue, region: Value[Region], deepCopy: Boolean): MValue = {
    cb.append(Region.storeInt(memory.addr, value.asInstanceOf[SInt32Value].intValue))
    memory.toMValue
  }

  override def pointerType: SType = SInt32
}
