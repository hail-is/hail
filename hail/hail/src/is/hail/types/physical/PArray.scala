package is.hail.types.physical

import is.hail.types.virtual.TArray

trait PArrayIterator {
  def hasNext: Boolean
  def isDefined: Boolean
  def value: Long
  def iterate(): Unit
}

abstract class PArray extends PArrayBackedContainer {
  lazy val virtualType: TArray = TArray(elementType.virtualType)
  final protected[physical] val elementRequired = elementType.required

  def elementIterator(aoff: Long, length: Int): PArrayIterator
}
