package is.hail.expr.types.physical

import is.hail.asm4s.Code
import is.hail.expr.types.virtual.{TStream, Type}
import is.hail.expr.ir.{EmitCode, EmitMethodBuilder, EmitStream, Stream}

final case class PCanonicalStream(elementType: PType, required: Boolean = false) extends PStream {
  def mux(mb: EmitMethodBuilder[_], cond: Code[Boolean], ifT: PUnrealizableCode, ifF: PUnrealizableCode): PUnrealizableCode =
    (ifT, ifF) match {
      case (PCanonicalStreamCode(tTyp, tStream), PCanonicalStreamCode(fTyp, fStream)) =>
        assert(tTyp == fTyp)
        PCanonicalStreamCode(tTyp, EmitStream.mux(mb, elementType, cond, tStream, fStream))
    }

  override val fundamentalType: PStream = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCStream[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def defaultValue: PCanonicalStreamCode =
    PCanonicalStreamCode(this, Stream.empty(EmitCode.missing(elementType)))

  override def deepRename(t: Type) = deepRenameStream(t.asInstanceOf[TStream])

  private def deepRenameStream(t: TStream): PStream =
    PCanonicalStream(this.elementType.deepRename(t.elementType), this.required)

  def setRequired(required: Boolean): PCanonicalStream = if(required == this.required) this else this.copy(required = required)
}

final case class PCanonicalStreamCode(pt: PCanonicalStream, stream: Stream[EmitCode]) extends PStreamCode
