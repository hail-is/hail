package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{BooleanInfo, Code, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PBoolean, PCanonicalCall, PCode, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

trait SBoolean extends SType

case object SCanonicalBoolean extends SBoolean {
  override def pType: PType = PBoolean(false)

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    value.st match {
      case SCanonicalBoolean =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(BooleanInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    pt match {
      case PBoolean(_) =>
        new SCanonicalBooleanCode(Region.loadBoolean(addr))
    }
  }
}

trait PBooleanCode extends PCode {
  def boolValue(cb: EmitCodeBuilder): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): PBooleanValue
}

trait PBooleanValue extends PValue {
  def boolValue(cb: EmitCodeBuilder): Code[Boolean]

}

class SCanonicalBooleanCode(val code: Code[Boolean]) extends PBooleanCode {
  val pt: PBoolean = PBoolean()

  def st: SBoolean = SCanonicalBoolean

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PBooleanValue = {
    val s = new SCanonicalBooleanSettable(sb.newSettable[Boolean]("sboolean_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PBooleanValue = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PBooleanValue = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def boolValue(cb: EmitCodeBuilder): Code[Boolean] = code
}

object SCanonicalBooleanSettable {
  def apply(sb: SettableBuilder, name: String): SCanonicalBooleanSettable = {
    new SCanonicalBooleanSettable(sb.newSettable[Boolean](name))
  }
}

class SCanonicalBooleanSettable(x: Settable[Boolean]) extends PBooleanValue with PSettable {
  val pt: PBoolean = PBoolean()

  def st: SBoolean = SCanonicalBoolean

  def store(cb: EmitCodeBuilder, v: PCode): Unit = cb.assign(x, v.asBoolean.boolValue(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: PCode = new SCanonicalBooleanCode(x)

  def boolValue(cb: EmitCodeBuilder): Code[Boolean] = x
}