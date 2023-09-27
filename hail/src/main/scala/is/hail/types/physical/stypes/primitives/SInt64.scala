package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.Code.invokeStatic1
import is.hail.asm4s.{LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.{PInt64, PType}
import is.hail.types.virtual.{TInt64, Type}
import is.hail.utils.FastSeq

case object SInt64 extends SPrimitive {
  override def ti: TypeInfo[_] = LongInfo

  override lazy val virtualType: Type = TInt64

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value.st match {
      case SInt64 => value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SInt64Settable = {
    val IndexedSeq(x: Settable[Long@unchecked]) = settables
    assert(x.ti == LongInfo)
    new SInt64Settable(x)
  }

  override def fromValues(settables: IndexedSeq[Value[_]]): SInt64Value = {
    val IndexedSeq(x: Value[Long@unchecked]) = settables
    assert(x.ti == LongInfo)
    new SInt64Value(x)
  }

  override def storageType(): PType = PInt64()
}

class SInt64Value(val value: Value[Long]) extends SPrimitiveValue {
  val pt: PInt64 = PInt64(false)

  override def valueTuple: IndexedSeq[Value[_]] = FastSeq(value)

  override def st: SInt64.type = SInt64

  override def _primitiveValue: Value[_] = value

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(cb.memoize(invokeStatic1[java.lang.Long, Long, Int]("hashCode", value)))

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = new SInt64Value(this.st.storageType().asInstanceOf[PInt64].byteSize)
}

object SInt64Settable {
  def apply(sb: SettableBuilder, name: String): SInt64Settable = {
    new SInt64Settable(sb.newSettable[Long](name))
  }
}

final class SInt64Settable(x: Settable[Long]) extends SInt64Value(x) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(x)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(x, v.asInstanceOf[SInt64Value].value)
}
