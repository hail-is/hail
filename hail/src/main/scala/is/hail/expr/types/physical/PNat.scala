package is.hail.expr.types.physical
import is.hail.annotations.CodeOrdering
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.{TNat, Type}

final case class PNat(n: Int, override val required: Boolean = false) extends PType {
  override def virtualType: Type = TNat(n)

  override def _toPretty: String = n.toString

  override def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = throw new UnsupportedOperationException
}
