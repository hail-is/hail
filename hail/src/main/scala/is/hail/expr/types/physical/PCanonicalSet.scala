package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.expr.ir.{PCanonicalIndexableCode, PCode}
import is.hail.expr.types.virtual.{TSet, Type}

final case class PCanonicalSet(elementType: PType,  required: Boolean = false) extends PSet with PArrayBackedContainer {
  val arrayRep = PCanonicalArray(elementType, required)

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalSet(elementType, required)

  def _asIdent = s"set_of_${elementType.asIdent}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCSet[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def deepRename(t: Type) = deepRenameSet(t.asInstanceOf[TSet])

  private def deepRenameSet(t: TSet) =
    PCanonicalSet(this.elementType.deepRename(t.elementType),  this.required)

  override def load(src: Code[Long]): PCode =
    new PCanonicalIndexableCode(this, Region.loadAddress(src))
}
