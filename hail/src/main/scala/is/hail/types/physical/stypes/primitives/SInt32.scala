package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{Code, IntInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PInt32, PType}
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils.FastIndexedSeq

case object SInt32 extends SPrimitive {
  override def ti: TypeInfo[_] = IntInfo

  override lazy val virtualType: Type = TInt32

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value.st match {
      case SInt32 => value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(IntInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SInt32Settable = {
    val IndexedSeq(x: Settable[Int@unchecked]) = settables
    assert(x.ti == IntInfo)
    new SInt32Settable(x)
  }

  override def fromValues(settables: IndexedSeq[Value[_]]): SInt32Value = {
    val IndexedSeq(x: Value[Int@unchecked]) = settables
    assert(x.ti == IntInfo)
    new SInt32Value(x)
  }

  override def storageType(): PType = PInt32()
}

class SInt32Code(val code: Code[Int]) extends SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  def st: SInt32.type = SInt32

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SInt32Value = {
    val s = new SInt32Settable(sb.newSettable[Int]("sInt32_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SInt32Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SInt32Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def intCode(cb: EmitCodeBuilder): Code[Int] = code
}

class SInt32Value(x: Value[Int]) extends SPrimitiveValue {
  val pt: PInt32 = PInt32(false)

  override def valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(x)

  override def st: SInt32.type = SInt32

  override def _primitiveValue: Value[_] = x

  override def get: SCode = new SInt32Code(x)

  def intCode(cb: EmitCodeBuilder): Value[Int] = x

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(intCode(cb))
}

object SInt32Settable {
  def apply(sb: SettableBuilder, name: String): SInt32Settable = {
    new SInt32Settable(sb.newSettable[Int](name))
  }
}

final class SInt32Settable(x: Settable[Int]) extends SInt32Value(x) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = cb.assign(x, v.asInt.intCode(cb))
}
