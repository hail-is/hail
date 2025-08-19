package is.hail.collection.implicits

import is.hail.collection.compat._
import is.hail.collection.compat.immutable.ArraySeq

import scala.collection.mutable
import scala.reflect.ClassTag
import org.typelevel.scalaccompat.annotation.nowarn213

import scala.collection.compat.IterableOnce

object RichArraySeqCompanion {
  @nowarn213("msg=method toArray in class IterableOnceExtensionMethods is deprecated")
  def sorted[A: ClassTag: Ordering](it: IterableOnce[A]): ArraySeq[A] =
    ArraySeq.unsafeWrapArray(
      it.toArray[A].sortInPlace().array
    )

  @nowarn213("msg=method toArray in class IterableOnceExtensionMethods is deprecated")
  def sortedBy[A: ClassTag, B: Ordering](it: IterableOnce[A])(f: A => B): ArraySeq[A] =
    ArraySeq.unsafeWrapArray(
      it.toArray[A].sortInPlaceBy(f).array
    )

  def newSortedBuilder[A: ClassTag: Ordering]: mutable.Builder[A, ArraySeq[A]] =
    Array.newBuilder[A].mapResult(a =>
      ArraySeq.unsafeWrapArray(a.sortInPlace().array)
    )

  def newSortedByBuilder[A]: SortedByBuilderFactory[A] = new SortedByBuilderFactory[A](())

  class SortedByBuilderFactory[A](val x: Unit) extends AnyVal {
    def apply[B](f: A => B)(implicit ct: ClassTag[A], ord: Ordering[B])
    : mutable.Builder[A, ArraySeq[A]] =
      Array.newBuilder[A].mapResult(a =>
        ArraySeq.unsafeWrapArray(a.sortInPlaceBy(f).array)
      )
  }
}
