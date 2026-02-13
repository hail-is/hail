package is.hail.collection

import is.hail.collection.compat.immutable.ArraySeq

import scala.collection.compat.Factory
import scala.collection.mutable.WrappedArray
import scala.reflect.ClassTag

package compat {
  class ArrayOps[A](val a: WrappedArray[A]) extends AnyVal {
    def sortInPlace()(implicit at: ClassTag[A], ord: Ordering[A]): WrappedArray[A] = {
      scala.util.Sorting.stableSort(a.array)
      a
    }

    def sortInPlaceBy[B](f: A => B)(implicit at: ClassTag[A], ord: Ordering[B]): WrappedArray[A] =
      sortInPlace()(at, ord = ord on f)
  }
}

package object compat {
  implicit def arraySeqbf[A: ClassTag](ob: ArraySeq.type): Factory[A, ArraySeq[A]] =
    ob.canBuildFrom[A]

  implicit def toArrayOps[A: ClassTag](a: Array[A]): ArrayOps[A] = new ArrayOps(a)
}
