package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{Code, FloatInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.{PFloat32, PType}
import is.hail.types.virtual.{TFloat32, Type}
import is.hail.utils.FastSeq

case object SFloat32 extends SPrimitive {
  override def ti: TypeInfo[_] = FloatInfo

  override lazy val virtualType: Type = TFloat32

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value.st match {
      case SFloat32 => value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(FloatInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SFloat32Settable = {
    val IndexedSeq(x: Settable[Float@unchecked]) = settables
    assert(x.ti == FloatInfo)
    new SFloat32Settable(x)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SFloat32Value = {
    val IndexedSeq(x: Value[Float@unchecked]) = values
    assert(x.ti == FloatInfo)
    new SFloat32Value(x)
  }

  override def storageType(): PType = PFloat32()
}

class SFloat32Value(val value: Value[Float]) extends SPrimitiveValue {
  val pt: PFloat32 = PFloat32()

  override def valueTuple: IndexedSeq[Value[_]] = FastSeq(value)

  override def st: SFloat32.type = SFloat32

  override def _primitiveValue: Value[_] = value

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(cb.memoize(Code.invokeStatic1[java.lang.Float, Float, Int]("floatToIntBits", value)))

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = new SInt64Value(4L)
}

object SFloat32Settable {
  def apply(sb: SettableBuilder, name: String): SFloat32Settable = {
    new SFloat32Settable(sb.newSettable[Float](name))
  }
}

final class SFloat32Settable(x: Settable[Float]) extends SFloat32Value(x) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(x)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(x, v.asInstanceOf[SFloat32Value].value)
}
