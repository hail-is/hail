package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{BooleanInfo, Code, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PBoolean, PCode, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq


case class SBoolean(required: Boolean) extends SPrimitive {
  def ti: TypeInfo[_] = BooleanInfo

  override def pType: PBoolean = PBoolean(required)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SBoolean(_) =>
        value.asInstanceOf[SBooleanCode]
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(BooleanInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    pt match {
      case PBoolean(_) =>
        new SBooleanCode(required: Boolean, Region.loadBoolean(addr))
    }
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SBooleanSettable = {
    val IndexedSeq(x: Settable[Boolean@unchecked]) = settables
    assert(x.ti == BooleanInfo)
    new SBooleanSettable(required, x)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SBooleanCode = {
    val IndexedSeq(x: Code[Boolean@unchecked]) = codes
    assert(x.ti == BooleanInfo)
    new SBooleanCode(required, x)
  }

  def canonicalPType(): PType = pType
}

class SBooleanCode(required: Boolean, val code: Code[Boolean]) extends PCode with SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  val pt: PBoolean = PBoolean(required)

  def st: SBoolean = SBoolean(required)

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SBooleanSettable = {
    val s = new SBooleanSettable(required, sb.newSettable[Boolean]("sboolean_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SBooleanSettable = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SBooleanSettable = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def boolCode(cb: EmitCodeBuilder): Code[Boolean] = code
}

object SBooleanSettable {
  def apply(sb: SettableBuilder, name: String, required: Boolean): SBooleanSettable = {
    new SBooleanSettable(required, sb.newSettable[Boolean](name))
  }
}

class SBooleanSettable(required: Boolean, x: Settable[Boolean]) extends PValue with PSettable {
  val pt: PBoolean = PBoolean(required)

  def st: SBoolean = SBoolean(required)

  def store(cb: EmitCodeBuilder, v: PCode): Unit = cb.assign(x, v.asBoolean.boolCode(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: PCode = new SBooleanCode(required, x)

  def boolCode(cb: EmitCodeBuilder): Code[Boolean] = x
}