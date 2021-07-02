package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{Code, FloatInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PCode, PFloat32, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

case class SFloat32(required: Boolean) extends SPrimitive {
  def ti: TypeInfo[_] = FloatInfo

  override def pType: PFloat32  = PFloat32(required)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SFloat32(r) =>
        if (r == required)
          value
        else
          new SFloat32Code(required, value.asInstanceOf[SFloat32Code].code)
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(FloatInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    pt match {
      case _: PFloat32 =>
        new SFloat32Code(required, Region.loadFloat(addr))
    }
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SFloat32Settable = {
    val IndexedSeq(x: Settable[Float@unchecked]) = settables
    assert(x.ti == FloatInfo)
    new SFloat32Settable(required, x)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SFloat32Code = {
    val IndexedSeq(x: Code[Float@unchecked]) = codes
    assert(x.ti == FloatInfo)
    new SFloat32Code(required, x)
  }

  def canonicalPType(): PType = pType
}

trait PFloat32Value extends PValue {
  def floatCode(cb: EmitCodeBuilder): Code[Float]

}

class SFloat32Code(required: Boolean, val code: Code[Float]) extends PCode with SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  val pt: PFloat32 = PFloat32(required)

  def st: SFloat32 = SFloat32(required)

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PFloat32Value = {
    val s = new SFloat32Settable(required, sb.newSettable[Float]("sint64_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PFloat32Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PFloat32Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def floatCode(cb: EmitCodeBuilder): Code[Float] = code
}

object SFloat32Settable {
  def apply(sb: SettableBuilder, name: String, required: Boolean): SFloat32Settable = {
    new SFloat32Settable(required, sb.newSettable[Float](name))
  }
}

class SFloat32Settable(required: Boolean, x: Settable[Float]) extends PFloat32Value with PSettable {
  val pt: PFloat32 = PFloat32(required)

  def st: SFloat32 = SFloat32(required)

  def store(cb: EmitCodeBuilder, v: PCode): Unit = cb.assign(x, v.asFloat.floatCode(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: PCode = new SFloat32Code(required, x)

  def floatCode(cb: EmitCodeBuilder): Code[Float] = x
}