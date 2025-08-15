package is.hail.utils

import scala.collection.mutable

package object compat {
  type ArraySeq[+A] = scala.collection.immutable.ArraySeq[A]
  type GrowableCompat[-A] = mutable.Growable[A]
  type ShrinkableCompat[-A] = mutable.Shrinkable[A]
}

package compat {
  package object immutable {
    val ArraySeq: scala.collection.compat.immutable.ArraySeq.type =
      scala.collection.compat.immutable.ArraySeq
  }
}
