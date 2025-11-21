package is.hail.utils.compat

package object mutable {
  type Growable[-A] = scala.collection.generic.Growable[A]
  type Shrinkable[-A] = scala.collection.generic.Shrinkable[A]
}
