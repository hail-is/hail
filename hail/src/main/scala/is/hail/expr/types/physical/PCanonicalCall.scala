package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.variant.Genotype

final case class PCanonicalCall(required: Boolean = false) extends PCall {
    def _asIdent = "call"

    override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCCall")

    val representation: PType = PInt32(required)

    def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
      assert(other isOfType this)
      PInt32().codeOrdering(mb)
    }

    def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalCall(required)
}
