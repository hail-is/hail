package is.hail.expr.types.physical

final case class PCanonicalSet(elementType: PType,  required: Boolean = false) extends PSet with PArrayBackedContainer {
  val arrayRep = PCanonicalArray(elementType, required)

  def copy(elementType: PType = this.elementType, required: Boolean = this.required): PSet = PCanonicalSet(elementType, required)

  def _asIdent = s"set_of_${elementType.asIdent}"
  def _toPretty = s"Set[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Set[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }
}
