package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, FloatInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PCanonicalCall, PCode, PFloat32, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

trait SFloat32 extends SType

case object SCanonicalFloat32 extends SFloat32 {
  override def pType: PType = PFloat32(false)

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = PFloat32(false).codeOrdering(mb, other.pType, so)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    value.st match {
      case SCanonicalFloat32 =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(FloatInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    pt match {
      case PFloat32(_) =>
        new SCanonicalFloat32Code(Region.loadFloat(addr))
    }
  }
}

trait PFloat32Code extends PCode {
  def floatValue(cb: EmitCodeBuilder): Code[Float]

  def memoize(cb: EmitCodeBuilder, name: String): PFloat32Value
}

trait PFloat32Value extends PValue {
  def floatValue(cb: EmitCodeBuilder): Code[Float]

}

class SCanonicalFloat32Code(val code: Code[Float]) extends PFloat32Code {
  val pt: PFloat32 = PFloat32()
  def st: SFloat32 = SCanonicalFloat32

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PFloat32Value = {
    val s = new SCanonicalFloat32Settable(sb.newSettable[Float]("sfloat32_memoize"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PFloat32Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PFloat32Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def floatValue(cb: EmitCodeBuilder): Code[Float] = code
}

object SCanonicalFloat32Settable {
  def apply(sb: SettableBuilder, name: String): SCanonicalFloat32Settable = {
    new SCanonicalFloat32Settable(sb.newSettable[Float](name))
  }
}


class SCanonicalFloat32Settable(x: Settable[Float]) extends PFloat32Value with PSettable {
  val pt: PFloat32 = PFloat32()
  def st: SFloat32 = SCanonicalFloat32

  def store(cb: EmitCodeBuilder, v: PCode): Unit = cb.assign(x, v.asFloat.floatValue(cb))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  def get: PCode = new SCanonicalFloat32Code(x)

  def floatValue(cb: EmitCodeBuilder): Code[Float] = x
}