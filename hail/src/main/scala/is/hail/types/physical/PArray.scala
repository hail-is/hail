package is.hail.types.physical

import is.hail.annotations.Annotation
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.types.virtual.TArray

trait PArrayIterator {
  def hasNext: Boolean
  def isDefined: Boolean
  def value: Long
  def iterate(): Unit
}

abstract class PArray extends PContainer {
  lazy val virtualType: TArray = TArray(elementType.virtualType)
  final protected[physical] val elementRequired = elementType.required

  def elementIterator(aoff: Long, length: Int): PArrayIterator

  override def genNonmissingValue(sm: HailStateManager): Gen[IndexedSeq[Annotation]] =
    Gen.buildableOf[Array](elementType.genValue(sm)).map(x => x: IndexedSeq[Annotation])
}
