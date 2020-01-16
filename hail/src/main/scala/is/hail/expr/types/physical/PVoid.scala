package is.hail.expr.types.physical
import is.hail.annotations.{CodeOrdering, ExtendedOrdering}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.{TVoid, Type}

case object PVoid extends PType {
  def virtualType: Type = TVoid

  override val required = true

  def _asIdent = "void"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PVoid")

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = null
}
