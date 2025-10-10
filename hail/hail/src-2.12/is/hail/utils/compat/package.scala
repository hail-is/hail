package is.hail.utils

import is.hail.utils.compat.immutable.ArraySeq

import scala.reflect.ClassTag

package object compat {

  type BuildFrom[-From, -To, C] = scala.collection.compat.BuildFrom[From, To, C]
  type IterableOnce[+A] = scala.collection.compat.IterableOnce[A]
  type Factory[-A, +To] = scala.collection.compat.Factory[A, To]

  val BuildFrom: scala.collection.compat.BuildFrom.type =
    scala.collection.compat.BuildFrom

  val IterableOnce: scala.collection.compat.IterableOnce.type =
    scala.collection.compat.IterableOnce

  implicit def arraySeqbf[A: ClassTag](ob: ArraySeq.type): Factory[A, ArraySeq[A]] =
    ob.canBuildFrom[A]
}
