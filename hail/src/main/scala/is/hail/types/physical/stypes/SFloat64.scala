package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, LongInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PCanonicalCall, PCode, PFloat64, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

trait SFloat64 extends SType

case object SCanonicalFloat64 extends SFloat64 {
  override def pType: PType = PCanonicalCall(false)

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = PFloat64(false).codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    value.st match {
      case SCanonicalFloat64 =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    pt match {
      case PFloat64(_) =>
        new SCanonicalFloat64Code(Region.loadDouble(addr))
    }
  }
}

trait PFloat64Code extends PCode {
  def doubleValue(cb: EmitCodeBuilder): Code[Double]

  def memoize(cb: EmitCodeBuilder, name: String): PFloat64Value
}

trait PFloat64Value extends PValue {
  def doubleValue(cb: EmitCodeBuilder): Code[Double]

}

class SCanonicalFloat64Code(val code: Code[Double]) extends PFloat64Code {
  val pt: PFloat64 = PFloat64()
  def st: SFloat64 = SCanonicalFloat64

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PFloat64Value = {
    val s = new SCanonicalFloat64Settable(sb.newSettable[Double]("sfloat64_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PFloat64Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PFloat64Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def doubleValue(cb: EmitCodeBuilder): Code[Double] = code
}

object SCanonicalFloat64Settable {
  def apply(sb: SettableBuilder, name: String): SCanonicalFloat64Settable = {
    new SCanonicalFloat64Settable(sb.newSettable[Double](name))
  }
}

class SCanonicalFloat64Settable(x: Settable[Double]) extends PFloat64Value with PSettable {
  val pt: PFloat64 = PFloat64()

  def st: SFloat64 = SCanonicalFloat64

  def store(cb: EmitCodeBuilder, v: PCode): Unit = cb.assign(x, v.asDouble.doubleValue(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: PCode = new SCanonicalFloat64Code(x)

  def doubleValue(cb: EmitCodeBuilder): Code[Double] = x
}