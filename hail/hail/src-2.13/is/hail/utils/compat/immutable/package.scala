package is.hail.utils.compat

package object immutable {
  type ArraySeq[+A] = scala.collection.immutable.ArraySeq[A]
  val ArraySeq = scala.collection.immutable.ArraySeq
}
