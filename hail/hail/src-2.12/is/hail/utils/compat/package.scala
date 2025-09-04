package is.hail.utils

import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.compat.Factory
import scala.collection.mutable.WrappedArray
import scala.reflect.ClassTag

package compat {
  class ArrayOps[A: ClassTag](val a: WrappedArray[A]) {
    def sortInPlace[B >: A]()(implicit ct: ClassTag[B], ord: Ordering[B])
      : WrappedArray[A] = {
      scala.util.Sorting.stableSort(a.array.asInstanceOf[Array[B]])
      a
    }

    def sortInPlaceBy[B](f: A => B)(implicit ord: Ordering[B]): WrappedArray[A] =
      sortInPlace()(implicitly, ord = ord on f)
  }
}

package object compat {
  implicit def arraySeqbf[A: ClassTag](ob: ArraySeq.type): Factory[A, ArraySeq[A]] =
    ob.canBuildFrom[A]

  implicit def toArrayOps[A: ClassTag](a: Array[A]): ArrayOps[A] = new ArrayOps(a)
}
