package is.hail.utils

import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.compat.Factory
import scala.reflect.ClassTag

package object compat {
  implicit def arraySeqbf[A: ClassTag](ob: ArraySeq.type): Factory[A, ArraySeq[A]] =
    ob.canBuildFrom[A]

  implicit class ArrayOps[A](private val a: Array[A]) extends AnyVal {
    def sortInPlace[B >: A]()(implicit ct: ClassTag[B], ord: Ordering[B]): a.type = {
      scala.util.Sorting.stableSort(a.asInstanceOf[Array[B]])
      a
    }

    def sortInPlaceBy[B](f: A => B)(implicit ord: Ordering[B], ct: ClassTag[A]): a.type =
      sortInPlace()(ct, ord on f)
  }
}
