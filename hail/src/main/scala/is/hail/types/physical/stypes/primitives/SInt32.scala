package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{IntInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.{PInt32, PType}
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils.FastSeq

case object SInt32 extends SPrimitive {
  override def ti: TypeInfo[_] = IntInfo

  override lazy val virtualType: Type = TInt32

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    value.st match {
      case SInt32 => value
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(IntInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SInt32Settable = {
    val IndexedSeq(x: Settable[Int @unchecked]) = settables
    assert(x.ti == IntInfo)
    new SInt32Settable(x)
  }

  override def fromValues(settables: IndexedSeq[Value[_]]): SInt32Value = {
    val IndexedSeq(x: Value[Int @unchecked]) = settables
    assert(x.ti == IntInfo)
    new SInt32Value(x)
  }

  override def storageType(): PType = PInt32()

  override def isIsomorphicTo(st: SType): Boolean =
    this == st
}

class SInt32Value(val value: Value[Int]) extends SPrimitiveValue {
  val pt: PInt32 = PInt32(false)

  override def valueTuple: IndexedSeq[Value[_]] = FastSeq(value)

  override def st: SInt32.type = SInt32

  override def _primitiveValue: Value[_] = value

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(value)

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = new SInt64Value(4L)
}

object SInt32Settable {
  def apply(sb: SettableBuilder, name: String): SInt32Settable =
    new SInt32Settable(sb.newSettable[Int](name))
}

final class SInt32Settable(x: Settable[Int]) extends SInt32Value(x) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(x)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(x, v.asInstanceOf[SInt32Value].value)
}
