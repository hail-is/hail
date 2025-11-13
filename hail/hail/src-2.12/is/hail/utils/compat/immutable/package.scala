package is.hail.utils.compat

package object immutable {
  // scalafix:off ForbiddenSymbol
  type ArraySeq[A] = scala.collection.compat.immutable.ArraySeq[A]
}
