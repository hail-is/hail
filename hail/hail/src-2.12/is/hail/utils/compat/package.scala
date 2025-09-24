package is.hail.utils

import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.compat.Factory
import scala.reflect.ClassTag

package object compat {
  implicit def arraySeqbf[A: ClassTag](ob: ArraySeq.type): Factory[A, ArraySeq[A]] =
    ob.canBuildFrom[A]
}
