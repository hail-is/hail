package is.hail.utils.compat

package object mutable {
  // scalafix:off ForbiddenSymbol
  type AnyRefMap[K <: AnyRef, V] = scala.collection.mutable.AnyRefMap[K, V]
  val AnyRefMap = scala.collection.mutable.AnyRefMap

  type Growable[-A] = scala.collection.generic.Growable[A]
  type Shrinkable[-A] = scala.collection.generic.Shrinkable[A]
}
