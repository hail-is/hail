package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{Code, IntInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PCode, PInt32, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

case class SInt32(required: Boolean) extends SPrimitive {
  def ti: TypeInfo[_] = IntInfo

  override def pType: PInt32  = PInt32(required)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SInt32(r) =>
        if (r == required)
          value
        else
          new SInt32Code(required, value.asInstanceOf[SInt32Code].code)
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(IntInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    pt match {
      case _: PInt32 =>
        new SInt32Code(required, Region.loadInt(addr))
    }
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SInt32Settable = {
    val IndexedSeq(x: Settable[Int@unchecked]) = settables
    assert(x.ti == IntInfo)
    new SInt32Settable(required, x)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SInt32Code = {
    val IndexedSeq(x: Code[Int@unchecked]) = codes
    assert(x.ti == IntInfo)
    new SInt32Code(required, x)
  }

  def canonicalPType(): PType = pType
}

trait PInt32Value extends PValue {
  def intCode(cb: EmitCodeBuilder): Code[Int]
}

class SInt32Code(required: Boolean, val code: Code[Int]) extends PCode with SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  val pt: PInt32 = PInt32(required)

  def st: SInt32 = SInt32(required)

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PInt32Value = {
    val s = new SInt32Settable(required, sb.newSettable[Int]("sInt32_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PInt32Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PInt32Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def intCode(cb: EmitCodeBuilder): Code[Int] = code
}

object SInt32Settable {
  def apply(sb: SettableBuilder, name: String, required: Boolean): SInt32Settable = {
    new SInt32Settable(required, sb.newSettable[Int](name))
  }
}

class SInt32Settable(required: Boolean, x: Settable[Int]) extends PInt32Value with PSettable {
  val pt: PInt32 = PInt32(required)

  def st: SInt32 = SInt32(required)

  def store(cb: EmitCodeBuilder, v: PCode): Unit = cb.assign(x, v.asInt.intCode(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: PCode = new SInt32Code(required, x)

  def intCode(cb: EmitCodeBuilder): Code[Int] = x
}