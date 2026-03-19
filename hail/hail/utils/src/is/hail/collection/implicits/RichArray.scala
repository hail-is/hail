package is.hail.collection.implicits

import is.hail.collection.compat.immutable.ArraySeq

class RichArray[T](val a: Array[T]) extends AnyVal {
  def index: Map[T, Int] = a.zipWithIndex.toMap

  def unsafeToArraySeq: ArraySeq[T] = ArraySeq.unsafeWrapArray(a)
}
