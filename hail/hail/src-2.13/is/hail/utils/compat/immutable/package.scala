package is.hail.utils.compat

import scala.collection.mutable

package object immutable {
  type ArraySeq[+A] = scala.collection.immutable.ArraySeq[A]

  val ArraySeq: scala.collection.compat.immutable.ArraySeq.type =
    scala.collection.compat.immutable.ArraySeq
}
