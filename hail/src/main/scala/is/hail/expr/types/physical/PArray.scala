package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TArray
object PArray {
  def apply(elementType: PType, required: Boolean = false) = new PCanonicalArray(elementType, required)
}

abstract class PArray extends PContainer with PStreamable {
  lazy val virtualType: TArray = TArray(elementType.virtualType, required)

  def copy(elementType: PType = this.elementType, required: Boolean = this.required): PArray

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(this isOfType other)
    CodeOrdering.iterableOrdering(this, other.asInstanceOf[PArray], mb)
  }
}