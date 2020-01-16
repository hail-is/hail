package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TSet

object PSet {
  def apply(elementType: PType, required: Boolean = false) = PCanonicalSet(elementType, required)
}

abstract class PSet extends PContainer {
  lazy val virtualType: TSet = TSet(elementType.virtualType, required)

  def copy(elementType: PType = this.elementType, required: Boolean = this.required): PSet

  def arrayFundamentalType: PArray = fundamentalType.asInstanceOf[PArray]

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.setOrdering(this, other.asInstanceOf[PSet], mb)
  }
}
