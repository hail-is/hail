package is.hail.types.physical.stypes.primitives

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, LineNumber, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PCode, PInt64, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

case class SInt64(required: Boolean) extends SType {
  override def pType: PInt64  = PInt64(required)

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean)(implicit line: LineNumber): SCode = {
    value.st match {
      case SInt64(r) =>
        if (r == required)
          value
        else
          new SInt64Code(required, value.asInstanceOf[SInt64Code].code)
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long])(implicit line: LineNumber): SCode = {
    pt match {
      case _: PInt64 =>
        new SInt64Code(required, Region.loadLong(addr))
    }
  }
}

trait PInt64Value extends PValue {
  def longCode(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Long]

}

class SInt64Code(required: Boolean, val code: Code[Long]) extends PCode {
  val pt: PInt64 = PInt64(required)

  def st: SInt64 = SInt64(required)

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder)(implicit line: LineNumber): PInt64Value = {
    val s = new SInt64Settable(required, sb.newSettable[Long]("sint64_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PInt64Value =
    memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PInt64Value =
    memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def longCode(cb: EmitCodeBuilder): Code[Long] = code
}

object SInt64Settable {
  def apply(sb: SettableBuilder, name: String, required: Boolean): SInt64Settable = {
    new SInt64Settable(required, sb.newSettable[Long](name))
  }
}

class SInt64Settable(required: Boolean, x: Settable[Long]) extends PInt64Value with PSettable {
  val pt: PInt64 = PInt64(required)

  def st: SInt64 = SInt64(required)

  def store(cb: EmitCodeBuilder, v: PCode)(implicit line: LineNumber): Unit = cb.assign(x, v.asLong.longCode(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get(implicit line: LineNumber): PCode = new SInt64Code(required, x)

  def longCode(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Long] = x
}