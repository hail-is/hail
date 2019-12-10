package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TSet

final case class PSet(elementType: PType, required: Boolean = false) extends PArrayBackedContainer(PCanonicalArray(elementType, required )) {
  lazy val virtualType: TSet = TSet(elementType.virtualType, required)

  override val fundamentalType: PArray = PCanonicalArray(elementType.fundamentalType, required)

  def _asIdent = s"set_of_${elementType.asIdent}"
  def _toPretty = s"Set[$elementType]"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("set<")
    elementType.pyString(sb)
    sb.append('>')
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Set[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.setOrdering(this, other.asInstanceOf[PSet], mb)
  }
}
