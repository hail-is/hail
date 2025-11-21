package is.hail.utils.richUtils

import is.hail.utils.compat._
import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.mutable
import scala.reflect.ClassTag

object RichArraySeqCompanion {
  def newSortedBuilder[T: ClassTag: Ordering]: mutable.Builder[T, ArraySeq[T]] =
    Array.newBuilder[T].mapResult(a =>
      ArraySeq.unsafeWrapArray(a.sortInPlace().array.asInstanceOf[Array[T]])
    )

  def newSortedByBuilder[A]: SortedByBuilderFactory[A] = new SortedByBuilderFactory[A](())

  class SortedByBuilderFactory[A](val x: Unit) extends AnyVal {
    def apply[B](f: A => B)(implicit ct: ClassTag[A], ord: Ordering[B])
      : mutable.Builder[A, ArraySeq[A]] =
      Array.newBuilder[A].mapResult(a =>
        ArraySeq.unsafeWrapArray(a.sortInPlaceBy(f).array.asInstanceOf[Array[A]])
      )
  }
}
