package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.expr.ir.{PCanonicalIndexableCode, PCode}
import is.hail.expr.types.virtual.{TArray, TDict, Type}

final case class PCanonicalDict(keyType: PType, valueType: PType, required: Boolean = false) extends PDict with PArrayBackedContainer {
  val elementType = PStruct(required = true, "key" -> keyType, "value" -> valueType)

  val arrayRep: PArray = PCanonicalArray(elementType, required)

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalDict(keyType, valueType, required)

  def _asIdent = s"dict_of_${keyType.asIdent}AND${valueType.asIdent}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCDict[")
    keyType.pretty(sb, indent, compact)
    if (compact)
      sb += ','
    else
      sb.append(", ")
    valueType.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def deepRename(t: Type) = deepRenameDict(t.asInstanceOf[TDict])

  private def deepRenameDict(t: TDict) =
    PCanonicalDict(this.keyType.deepRename(t.keyType), this.valueType.deepRename(t.valueType), this.required)

  override def load(src: Code[Long]): PCode =
    new PCanonicalIndexableCode(this, Region.loadAddress(src))
}
