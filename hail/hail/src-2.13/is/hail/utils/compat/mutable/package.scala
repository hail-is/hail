package is.hail.utils.compat

package object mutable {
  type Growable[-A] = scala.collection.mutable.Growable[A]
  type GrowableCompat[-A] = Growable[A]
  type Shrinkable[-A] = scala.collection.mutable.Shrinkable[A]
  type ShrinkableCompat[-A] = Shrinkable[A]
}
