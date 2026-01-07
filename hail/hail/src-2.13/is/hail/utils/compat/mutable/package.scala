package is.hail.utils.compat

package object mutable {
  type AnyRefMap[K, V] = scala.collection.mutable.HashMap[K, V]
  val AnyRefMap = scala.collection.mutable.HashMap

  type Growable[-A] = scala.collection.mutable.Growable[A]
  type GrowableCompat[-A] = Growable[A]
  type Shrinkable[-A] = scala.collection.mutable.Shrinkable[A]
  type ShrinkableCompat[-A] = Shrinkable[A]
}
