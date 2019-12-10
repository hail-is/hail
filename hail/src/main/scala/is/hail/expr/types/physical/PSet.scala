package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TSet

object PSet {
  def apply(elementType: PType, required: Boolean = false) = PCanonicalSet(elementType, required)
}

abstract class PSet extends PContainer with PArrayBackedContainer {
  lazy val virtualType: TSet = TSet(elementType.virtualType, required)

  // TODO: Fix
//  override val fundamentalType: PArray = PCanonicalArray(elementType.fundamentalType, required)

  def copy(elementType: PType = this.elementType, required: Boolean = this.required): PSet

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.setOrdering(this, other.asInstanceOf[PSet], mb)
  }
}
