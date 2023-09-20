package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{BooleanInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.{PBoolean, PType}
import is.hail.types.virtual.{TBoolean, Type}
import is.hail.utils.FastSeq


case object SBoolean extends SPrimitive {
  override def ti: TypeInfo[_] = BooleanInfo

  override lazy val virtualType: Type = TBoolean

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value.st match {
      case SBoolean =>
        value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(BooleanInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SBooleanSettable = {
    val IndexedSeq(x: Settable[Boolean@unchecked]) = settables
    assert(x.ti == BooleanInfo)
    new SBooleanSettable( x)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SBooleanValue = {
    val IndexedSeq(x: Value[Boolean@unchecked]) = values
    assert(x.ti == BooleanInfo)
    new SBooleanValue( x)
  }

  override def storageType(): PType = PBoolean()
}

class SBooleanValue(val value: Value[Boolean]) extends SPrimitiveValue {
  val pt: PBoolean = PBoolean()

  override def st: SBoolean.type = SBoolean

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(value)

  override def _primitiveValue: Value[_] = value

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(cb.memoize(value.toI))

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = new SInt64Value(pt.byteSize)
}

object SBooleanSettable {
  def apply(sb: SettableBuilder, name: String): SBooleanSettable = {
    new SBooleanSettable( sb.newSettable[Boolean](name))
  }
}

class SBooleanSettable(x: Settable[Boolean]) extends SBooleanValue(x) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(x)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(x, v.asInstanceOf[SBooleanValue].value)
}
