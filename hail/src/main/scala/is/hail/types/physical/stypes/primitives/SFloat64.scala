package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.Code.invokeStatic1
import is.hail.asm4s.{DoubleInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.{PFloat64, PType}
import is.hail.types.virtual.{TFloat64, Type}
import is.hail.utils.FastSeq

case object SFloat64 extends SPrimitive {
  override def ti: TypeInfo[_] = DoubleInfo

  override lazy val virtualType: Type = TFloat64

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value.st match {
      case SFloat64 => value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(DoubleInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SFloat64Settable = {
    val IndexedSeq(x: Settable[Double@unchecked]) = settables
    assert(x.ti == DoubleInfo)
    new SFloat64Settable(x)
  }

  override def fromValues(settables: IndexedSeq[Value[_]]): SFloat64Value = {
    val IndexedSeq(x: Value[Double@unchecked]) = settables
    assert(x.ti == DoubleInfo)
    new SFloat64Value(x)
  }

  override def storageType(): PType = PFloat64()
}

object SFloat64Value {
  def apply(code: Value[Double]): SFloat64Value = new SFloat64Value(code)
}

class SFloat64Value(val value: Value[Double]) extends SPrimitiveValue {
  val pt: PFloat64 = PFloat64(false)

  override def valueTuple: IndexedSeq[Value[_]] = FastSeq(value)

  override def st: SFloat64.type = SFloat64

  override def _primitiveValue: Value[_] = value

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(cb.memoize(invokeStatic1[java.lang.Double, Double, Int]("hashCode", value)))

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = new SInt64Value(8L)
}

object SFloat64Settable {
  def apply(sb: SettableBuilder, name: String): SFloat64Settable = {
    new SFloat64Settable(sb.newSettable[Double](name))
  }
}

final class SFloat64Settable(x: Settable[Double]) extends SFloat64Value(x) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(x)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(x, v.asInstanceOf[SFloat64Value].value)
}
