package is.hail.types.physical.mtypes

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SInt32Value, SStringPointer, SStringPointerCode, SStringPointerValue, SType, SValue}
import is.hail.utils._

trait MString extends MType {
  override def pointerType: SType = SStringPointer(this)
}



case object MCanonicalString extends MString with MPointer {
  def storeFromSValue(cb: EmitCodeBuilder, memory: UninitializedMValue, value: SValue, region: Value[Region], deepCopy: Boolean): MValue = {
    value.typ match {
      case SStringPointer(MCanonicalString) =>
        if (deepCopy) {
          val dstAddress = cb.newLocal[Long]("addr")
          val str = value.asInstanceOf[SStringPointerValue]
          cb.assign(dstAddress, region.allocate(1L, str.length()))
          cb.append(Region.copyFrom(str.value.storeBytes(dstAddress, byteRep, 0L, byteRep.length().toL))
          cb.append(Region.storeAddress(memory.addr, dstAddress))

        } else {

        }
      case t =>
        val dstAddress = cb.newLocal[Long]("addr")
        val byteRep = cb.newLocal[Array[Byte]]("string_bytes")

        cb.assign(byteRep, value.asString.stringValue.invoke[Array[Byte]]("getBytes"))
        cb.assign(dstAddress, region.allocate(1L, byteRep.length()))
        cb.append(Region.storeBytes(dstAddress, byteRep, 0L, byteRep.length().toL))
        cb.append(Region.storeAddress(memory.addr, dstAddress))
    }
    memory.toMValue
  }


}
