package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, IntInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PCanonicalCall, PCode, PInt32, PInt64, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

trait SInt32 extends SType

case object SCanonicalInt32 extends SInt32 {
  override def pType: PType = PCanonicalCall(false)

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = PInt32(false).codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    value.st match {
      case SCanonicalInt32 =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(IntInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    pt match {
      case PInt32(_) =>
        new SCanonicalInt32Code(Region.loadInt(addr))
    }
  }
}

trait PInt32Code extends PCode {
  def intValue(cb: EmitCodeBuilder): Code[Int]

  def memoize(cb: EmitCodeBuilder, name: String): PInt32Value
}

trait PInt32Value extends PValue {
  def intValue(cb: EmitCodeBuilder): Code[Int]

}

class SCanonicalInt32Code(val code: Code[Int]) extends PInt32Code {
  val pt: PInt32 = PInt32()
  def st: SInt32 = SCanonicalInt32

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PInt32Value = {
    val s = new SCanonicalInt32Settable(sb.newSettable[Int]("sint32_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PInt32Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PInt32Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def intValue(cb: EmitCodeBuilder): Code[Int] = code
}

object SCanonicalInt32Settable {
  def apply(sb: SettableBuilder, name: String): SCanonicalInt32Settable = {
    new SCanonicalInt32Settable(sb.newSettable[Int](name))
  }
}

class SCanonicalInt32Settable(x: Settable[Int]) extends PInt32Value with PSettable {
  val pt: PInt32 = PInt32()

  def st: SInt32 = SCanonicalInt32

  def store(cb: EmitCodeBuilder, v: PCode): Unit = cb.assign(x, v.asInt.intValue(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: PCode = new SCanonicalInt32Code(x)

  def intValue(cb: EmitCodeBuilder): Code[Int] = x
}