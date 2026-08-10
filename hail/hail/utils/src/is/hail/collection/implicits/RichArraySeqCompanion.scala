package is.hail.collection.implicits

import scala.collection.{mutable, IterableOnce}
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag

object RichArraySeqCompanion {
  def sorted[A: ClassTag: Ordering](it: IterableOnce[A]): ArraySeq[A] =
    it.iterator.toArray[A].sortedInPlace().unsafeToArraySeq

  def sortedBy[A: ClassTag, B: Ordering](it: IterableOnce[A])(f: A => B): ArraySeq[A] =
    it.iterator.toArray[A].sortedInPlaceBy(f).unsafeToArraySeq

  def newSortedBuilder[A: ClassTag: Ordering]: mutable.Builder[A, ArraySeq[A]] =
    Array.newBuilder[A].mapResult(a => ArraySeq.unsafeWrapArray(a.sortedInPlace()))

  def newSortedByBuilder[A]: SortedByBuilderFactory[A] = new SortedByBuilderFactory[A](())

  class SortedByBuilderFactory[A](val x: Unit) extends AnyVal {
    def apply[B](f: A => B)(implicit ct: ClassTag[A], ord: Ordering[B])
      : mutable.Builder[A, ArraySeq[A]] =
      Array.newBuilder[A].mapResult(a => ArraySeq.unsafeWrapArray(a.sortedInPlaceBy(f)))
  }
}
