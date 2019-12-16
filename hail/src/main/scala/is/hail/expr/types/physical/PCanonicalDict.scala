package is.hail.expr.types.physical

final case class PCanonicalDict(keyType: PType, valueType: PType, required: Boolean = false) extends PDict with PArrayBackedContainer {
  val elementType = PStruct(required = true, "key" -> keyType, "value" -> valueType)

  val arrayRep: PArray = PCanonicalArray(elementType, required)

  def copy(keyType: PType = this.keyType, valueType: PType = this.valueType, required: Boolean = this.required): PDict =
    PCanonicalDict(keyType, valueType, required)

  def _asIdent = s"dict_of_${keyType.asIdent}AND${valueType.asIdent}"
  def _toPretty = s"Dict[$keyType, $valueType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Dict[")
    keyType.pretty(sb, indent, compact)
    if (compact)
      sb += ','
    else
      sb.append(", ")
    valueType.pretty(sb, indent, compact)
    sb.append("]")
  }
}
