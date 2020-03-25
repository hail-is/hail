package is.hail.expr.types.physical

import is.hail.annotations.{Annotation, CodeOrdering}
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TArray
object PArray {
  def apply(elementType: PType, required: Boolean = false) = new PCanonicalArray(elementType, required)
}

trait PArrayIterator {
  def hasNext: Boolean
  def isDefined: Boolean
  def value: Long
  def iterate(): Unit
}

abstract class PArray extends PContainer {
  lazy val virtualType: TArray = TArray(elementType.virtualType)

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(this isOfType other)
    CodeOrdering.iterableOrdering(this, other.asInstanceOf[PArray], mb)
  }

  def elementIterator(aoff: Long, length: Int): PArrayIterator

  override def genNonmissingValue: Gen[IndexedSeq[Annotation]] =
    Gen.buildableOf[Array](elementType.genValue).map(x => x: IndexedSeq[Annotation])
}
