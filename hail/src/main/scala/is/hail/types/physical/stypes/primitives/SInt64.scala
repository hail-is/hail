package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.Code.invokeStatic1
import is.hail.asm4s.{Code, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PInt64, PType}
import is.hail.types.virtual.{TInt64, Type}
import is.hail.utils.FastIndexedSeq

case object SInt64 extends SPrimitive {
  override def ti: TypeInfo[_] = LongInfo

  override lazy val virtualType: Type = TInt64

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value.st match {
      case SInt64 => value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

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

class SInt64Code(val code: Code[Long]) extends SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  def st: SInt64.type = SInt64

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SInt64Value = {
    val s = new SInt64Settable(sb.newSettable[Long]("sint64_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SInt64Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SInt64Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def longCode(cb: EmitCodeBuilder): Code[Long] = code
}

class SInt64Value(x: Value[Long]) extends SPrimitiveValue {
  val pt: PInt64 = PInt64(false)

  override def valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(x)

  override def st: SInt64.type = SInt64

  override def get: SCode = new SInt64Code(x)

  def longCode(cb: EmitCodeBuilder): Value[Long] = x

  override def hash(cb: EmitCodeBuilder): SInt32Code =
    new SInt32Code(invokeStatic1[java.lang.Long, Long, Int]("hashCode", longCode(cb)))
}

object SInt64Settable {
  def apply(sb: SettableBuilder, name: String): SInt64Settable = {
    new SInt64Settable(sb.newSettable[Long](name))
  }
}

final class SInt64Settable(x: Settable[Long]) extends SInt64Value(x) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = cb.assign(x, v.asLong.longCode(cb))
}
