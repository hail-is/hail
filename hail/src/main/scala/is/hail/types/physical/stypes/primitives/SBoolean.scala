package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{BooleanInfo, Code, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PBoolean, PType}
import is.hail.types.virtual.{TBoolean, Type}
import is.hail.utils.FastIndexedSeq


case object SBoolean extends SPrimitive {
  def ti: TypeInfo[_] = BooleanInfo

  lazy val virtualType: Type = TBoolean

  override def castRename(t: Type): SType = this

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SBoolean =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(BooleanInfo)

  def fromSettables(settables: IndexedSeq[Settable[_]]): SBooleanSettable = {
    val IndexedSeq(x: Settable[Boolean@unchecked]) = settables
    assert(x.ti == BooleanInfo)
    new SBooleanSettable( x)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SBooleanCode = {
    val IndexedSeq(x: Code[Boolean@unchecked]) = codes
    assert(x.ti == BooleanInfo)
    new SBooleanCode(x)
  }

  def canonicalPType(): PType = PBoolean()
}

class SBooleanCode(val code: Code[Boolean]) extends SCode with SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  def st: SBoolean.type = SBoolean

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SBooleanSettable = {
    val s = new SBooleanSettable(sb.newSettable[Boolean]("sboolean_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SBooleanSettable = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SBooleanSettable = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def boolCode(cb: EmitCodeBuilder): Code[Boolean] = code
}

object SBooleanSettable {
  def apply(sb: SettableBuilder, name: String): SBooleanSettable = {
    new SBooleanSettable( sb.newSettable[Boolean](name))
  }
}

class SBooleanSettable(x: Settable[Boolean]) extends SValue with SSettable {
  val pt: PBoolean = PBoolean()

  def st: SBoolean.type = SBoolean

  def store(cb: EmitCodeBuilder, v: SCode): Unit = cb.assign(x, v.asBoolean.boolCode(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: SCode = new SBooleanCode(x)

  def boolCode(cb: EmitCodeBuilder): Code[Boolean] = x
}