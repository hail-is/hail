package is.hail.utils

import scala.collection.mutable

package object compat {
  type BuildFrom[-From, -To, C] = scala.collection.BuildFrom[From, To, C]
  type Factory[-A, +To] = scala.collection.Factory[A, To]
  type GrowableCompat[-A] = mutable.Growable[A]
  type IterableOnce[+A] = scala.collection.IterableOnce[A]
  type ShrinkableCompat[-A] = mutable.Shrinkable[A]

  val BuildFrom: scala.collection.BuildFrom.type =
    scala.collection.BuildFrom

  val IterableOnce: scala.collection.IterableOnce.type =
    scala.collection.IterableOnce
}
