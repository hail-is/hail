package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, IntInfo, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PCanonicalCall, PCode, PInt64, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

trait SInt64 extends SType

case object SCanonicalInt64 extends SInt64 {
  override def pType: PType = PCanonicalCall(false)

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = PInt64(false).codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    value.st match {
      case SCanonicalInt64 =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    pt match {
      case PInt64(_) =>
        new SCanonicalInt64Code(Region.loadLong(addr))
    }
  }
}

trait PInt64Code extends PCode {
  def longValue(cb: EmitCodeBuilder): Code[Long]

  def memoize(cb: EmitCodeBuilder, name: String): PInt64Value
}

trait PInt64Value extends PValue {
  def longValue(cb: EmitCodeBuilder): Code[Long]

}

class SCanonicalInt64Code(val code: Code[Long]) extends PInt64Code {
  val pt: PInt64 = PInt64()

  def st: SInt64 = SCanonicalInt64

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PInt64Value = {
    val s = new SCanonicalInt64Settable(sb.newSettable[Long]("sint64_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PInt64Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PInt64Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def longValue(cb: EmitCodeBuilder): Code[Long] = code
}

object SCanonicalInt64Settable {
  def apply(sb: SettableBuilder, name: String): SCanonicalInt64Settable = {
    new SCanonicalInt64Settable(sb.newSettable[Long](name))
  }
}

class SCanonicalInt64Settable(x: Settable[Long]) extends PInt64Value with PSettable {
  val pt: PInt64 = PInt64()

  def st: SInt64 = SCanonicalInt64

  def store(cb: EmitCodeBuilder, v: PCode): Unit = cb.assign(x, v.asLong.longValue(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: PCode = new SCanonicalInt64Code(x)

  def longValue(cb: EmitCodeBuilder): Code[Long] = x
}