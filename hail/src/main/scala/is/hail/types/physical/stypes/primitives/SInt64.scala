package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{Code, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PInt64, PType}
import is.hail.types.virtual.{TInt64, Type}
import is.hail.utils.FastIndexedSeq

case object SInt64 extends SPrimitive {
  def ti: TypeInfo[_] = LongInfo

  lazy val virtualType: Type = TInt64

  override def castRename(t: Type): SType = this

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SInt64 => value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def fromSettables(settables: IndexedSeq[Settable[_]]): SInt64Settable = {
    val IndexedSeq(x: Settable[Long@unchecked]) = settables
    assert(x.ti == LongInfo)
    new SInt64Settable(x)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SInt64Code = {
    val IndexedSeq(x: Code[Long@unchecked]) = codes
    assert(x.ti == LongInfo)
    new SInt64Code(x)
  }

  def canonicalPType(): PType = PInt64()
}

trait SInt64Value extends SValue {
  def longCode(cb: EmitCodeBuilder): Code[Long]
}

class SInt64Code(val code: Code[Long]) extends SCode with SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  def st: SInt64.type = SInt64

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SInt64Value = {
    val s = new SInt64Settable(sb.newSettable[Long]("sint64_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SInt64Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SInt64Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def longCode(cb: EmitCodeBuilder): Code[Long] = code
}

object SInt64Settable {
  def apply(sb: SettableBuilder, name: String): SInt64Settable = {
    new SInt64Settable(sb.newSettable[Long](name))
  }
}

class SInt64Settable(x: Settable[Long]) extends SInt64Value with SSettable {
  val pt: PInt64 = PInt64(false)

  def st: SInt64.type = SInt64

  def store(cb: EmitCodeBuilder, v: SCode): Unit = cb.assign(x, v.asLong.longCode(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: SCode = new SInt64Code(x)

  def longCode(cb: EmitCodeBuilder): Code[Long] = x
}