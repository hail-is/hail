package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{Code, IntInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PInt32, PType}
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils.FastIndexedSeq

case object SInt32 extends SPrimitive {
  def ti: TypeInfo[_] = IntInfo

  lazy val virtualType: Type = TInt32

  override def castRename(t: Type): SType = this

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SInt32 => value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(IntInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    pt match {
      case _: PInt32 =>
        new SInt32Code(Region.loadInt(addr))
    }
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SInt32Settable = {
    val IndexedSeq(x: Settable[Int@unchecked]) = settables
    assert(x.ti == IntInfo)
    new SInt32Settable(x)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SInt32Code = {
    val IndexedSeq(x: Code[Int@unchecked]) = codes
    assert(x.ti == IntInfo)
    new SInt32Code(x)
  }

  def canonicalPType(): PType = PInt32()
}

trait SInt32Value extends SValue {
  def intCode(cb: EmitCodeBuilder): Code[Int]
}

class SInt32Code(val code: Code[Int]) extends SCode with SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  def st: SInt32.type = SInt32

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SInt32Value = {
    val s = new SInt32Settable(sb.newSettable[Int]("sInt32_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SInt32Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SInt32Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def intCode(cb: EmitCodeBuilder): Code[Int] = code
}

object SInt32Settable {
  def apply(sb: SettableBuilder, name: String): SInt32Settable = {
    new SInt32Settable(sb.newSettable[Int](name))
  }
}

class SInt32Settable(x: Settable[Int]) extends SInt32Value with SSettable {
  val pt: PInt32 = PInt32(false)

  def st: SInt32.type = SInt32

  def store(cb: EmitCodeBuilder, v: SCode): Unit = cb.assign(x, v.asInt.intCode(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: SCode = new SInt32Code(x)

  def intCode(cb: EmitCodeBuilder): Code[Int] = x
}