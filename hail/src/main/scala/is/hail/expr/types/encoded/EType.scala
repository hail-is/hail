package is.hail.expr.types.encoded

import is.hail.expr.types.physical._

object EType {
  // does basic conversion from PType to EType
  def fromPType(t: PType): EType = t match {
    case t: PInt32 => EInt32(t.required)
    case t: PInt64 => EInt64(t.required)
    case t: PFloat32 => EFloat32(t.required)
    case t: PFloat64 => EFloat64(t.required)
    case t: PBoolean => EBoolean(t.required)
    case t: PBinary => EBinary(t.required)
    case t: PString => EString(t.required)
    case PLocus(rg, req) => ELocus(rg, req)
    case PInterval(pt, req) => EInterval(fromPType(pt), req)
    case PArray(et, req) => EArray(fromPType(et), req)
    case PSet(et, req) => ESet(fromPType(et), req)
    case PDict(kt, vt, req) => EDict(fromPType(kt), fromPType(vt), req)
    case PVoid => EVoid
  }
}

abstract class EType extends Serializable {
  def required: Boolean

  def fundamentalType: EType = this

  def toPType(): PType
}

case object EVoid extends EType {
  override val required = true

  def toPType(): PType = PVoid
}
