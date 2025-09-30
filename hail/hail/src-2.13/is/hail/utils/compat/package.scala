package is.hail.utils

import scala.collection.mutable

package object compat {
  type GrowableCompat[-A] = mutable.Growable[A]
  type ShrinkableCompat[-A] = mutable.Shrinkable[A]
}
