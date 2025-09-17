package is.hail.utils

import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.compat.Factory
import scala.collection.mutable.WrappedArray
import scala.reflect.ClassTag

package object compat {
  implicit def arraySeqbf[A: ClassTag](ob: ArraySeq.type): Factory[A, ArraySeq[A]] =
    ob.canBuildFrom[A]

  implicit class ArrayOps[A](private val a: Array[A]) extends AnyVal {
    def sortInPlace[B >: A]()(implicit ct: ClassTag[B], ord: Ordering[B]): WrappedArray[A] = {
      scala.util.Sorting.stableSort(a.asInstanceOf[Array[B]])
      WrappedArray.make(a)
    }

    def sortInPlaceBy[B](f: A => B)(implicit ord: Ordering[B], ct: ClassTag[A]): WrappedArray[A] =
      sortInPlace()(ct, ord on f)
  }
}
