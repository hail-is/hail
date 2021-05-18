package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{Code, DoubleInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PFloat64, PType}
import is.hail.types.virtual.{TFloat64, Type}
import is.hail.utils.FastIndexedSeq

case object SFloat64 extends SPrimitive {
  def ti: TypeInfo[_] = DoubleInfo

  lazy val virtualType: Type = TFloat64

  override def castRename(t: Type): SType = this

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SFloat64 => value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(DoubleInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    pt match {
      case _: PFloat64 =>
        new SFloat64Code(Region.loadDouble(addr))
    }
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SFloat64Settable = {
    val IndexedSeq(x: Settable[Double@unchecked]) = settables
    assert(x.ti == DoubleInfo)
    new SFloat64Settable(x)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SFloat64Code = {
    val IndexedSeq(x: Code[Double@unchecked]) = codes
    assert(x.ti == DoubleInfo)
    new SFloat64Code(x)
  }

  def canonicalPType(): PType = PFloat64()
}

trait SFloat64Value extends SValue {
  def doubleCode(cb: EmitCodeBuilder): Code[Double]

}

object SFloat64Code {
  def apply(code: Code[Double]): SFloat64Code = new SFloat64Code(code)
}

class SFloat64Code(val code: Code[Double]) extends SCode with SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  def st: SFloat64.type = SFloat64

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SFloat64Value = {
    val s = new SFloat64Settable(sb.newSettable[Double]("sint64_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SFloat64Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SFloat64Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def doubleCode(cb: EmitCodeBuilder): Code[Double] = code
}

object SFloat64Settable {
  def apply(sb: SettableBuilder, name: String): SFloat64Settable = {
    new SFloat64Settable(sb.newSettable[Double](name))
  }
}

class SFloat64Settable(x: Settable[Double]) extends SFloat64Value with SSettable {
  val pt: PFloat64 = PFloat64(false)

  def st: SFloat64.type = SFloat64

  def store(cb: EmitCodeBuilder, v: SCode): Unit = cb.assign(x, v.asDouble.doubleCode(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: SCode = new SFloat64Code(x)

  def doubleCode(cb: EmitCodeBuilder): Code[Double] = x
}