package is.hail.utils.compat

package object mutable {
  type Growable[-A] = scala.collection.mutable.Growable[A]
  type Shrinkable[-A] = scala.collection.mutable.Shrinkable[A]
}
