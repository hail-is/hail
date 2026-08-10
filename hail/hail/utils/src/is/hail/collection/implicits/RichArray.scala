package is.hail.collection.implicits

import scala.collection.immutable.ArraySeq
import scala.collection.mutable

class RichArray[T](val a: Array[T]) extends AnyVal {
  def index: Map[T, Int] = a.zipWithIndex.toMap

  def unsafeToArraySeq: ArraySeq[T] = ArraySeq.unsafeWrapArray(a)

  // Predef.wrapRefArray returns a shared ArraySeq backed by Array[AnyRef] when
  // the array is empty, so the stdlib `.sortInPlace*(...).array` idiom throws
  // ClassCastException; these sort `a` in place and return it.
  def sortedInPlace()(implicit ord: Ordering[T]): Array[T] = {
    if (a.length > 1) { val _ = mutable.ArraySeq.make(a).sortInPlace() }
    a
  }

  def sortedInPlaceBy[B](f: T => B)(implicit ord: Ordering[B]): Array[T] =
    sortedInPlace()(ord.on(f))
}
