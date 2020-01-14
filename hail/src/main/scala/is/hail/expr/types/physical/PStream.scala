package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TStream

trait PStreamable extends PIterable {
  def asPArray: PArray = PArray(this.elementType, this.required)
  def copyStreamable(elt: PType, req: Boolean = required): PStreamable = {
    this match {
      case _: PArray => PArray(elt, req)
      case _: PStream => PStream(elt, req)
    }
  }
}

final case class PStream(elementType: PType, override val required: Boolean = false) extends PStreamable {
  lazy val virtualType: TStream = TStream(elementType.virtualType, required)

  override val fundamentalType: PStream = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  def _asIdent = s"stream_of_${elementType.asIdent}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PStream[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering =
    throw new UnsupportedOperationException("Stream comparison is currently undefined.")
}

