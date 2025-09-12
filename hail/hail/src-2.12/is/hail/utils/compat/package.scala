package is.hail.utils

import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.compat.Factory
import scala.reflect.ClassTag

package object compat {
  implicit def arraySeqbf[A: ClassTag](ob: ArraySeq.type): Factory[A, ArraySeq[A]] =
    ob.canBuildFrom[A]

  // re-exports

//  implicit def MutableTreeMapExtensions2(fact: m.TreeMap.type): c.MutableTreeMapExtensions2 =
//    new c.MutableTreeMapExtensions2(fact)
//
//  implicit def MutableSortedMapExtensions(fact: m.SortedMap.type): c.MutableSortedMapExtensions =
//    new c.MutableSortedMapExtensions(fact)
//
//  implicit def genericOrderedCompanionToCBF[A, CC[X] <: Traversable[X]](
//    fact: GenericOrderedCompanion[CC]
//  )(implicit ordering: Ordering[A]
//  ): CanBuildFrom[Any, A, CC[A]] =
//    c.genericOrderedCompanionToCBF(fact)(ordering)
//
//  implicit def canBuildFromIterableViewMapLike[K, V, L, W, CC[X, Y] <: Map[X, Y]]
//    : CanBuildFrom[IterableView[(K, V), CC[K, V]], (L, W), IterableView[(L, W), CC[L, W]]] =
//    c.canBuildFromIterableViewMapLike
//
//  implicit def toTraversableLikeExtensionMethods[Repr](
//    self: Repr
//  )(implicit
//    traversable: IsTraversableLike[Repr]
//  ): c.TraversableLikeExtensionMethods[traversable.A, Repr] =
//    new c.TraversableLikeExtensionMethods[traversable.A, Repr](traversable.conversion(self))
//
//  implicit def toSeqExtensionMethods[A](self: Seq[A]): c.SeqExtensionMethods[A] =
//    new c.SeqExtensionMethods[A](self)
//
//  implicit def toTrulyTraversableLikeExtensionMethods[T1, El1, Repr1](
//    self: T1
//  )(implicit w1: T1 => TraversableLike[El1, Repr1]
//  ): c.TrulyTraversableLikeExtensionMethods[El1, Repr1] =
//    new c.TrulyTraversableLikeExtensionMethods[El1, Repr1](w1(self))
//
//  implicit def toTuple2ZippedExtensionMethods[El1, Repr1, El2, Repr2](
//    self: Tuple2Zipped[El1, Repr1, El2, Repr2]
//  ): c.Tuple2ZippedExtensionMethods[El1, Repr1, El2, Repr2] =
//    new c.Tuple2ZippedExtensionMethods[El1, Repr1, El2, Repr2](self)
//
//  implicit def toImmutableQueueExtensionMethods[A](
//    self: i.Queue[A]
//  ): c.ImmutableQueueExtensionMethods[A] =
//    new c.ImmutableQueueExtensionMethods[A](self)
//
//  implicit def toMutableQueueExtensionMethods[A](
//    self: m.Queue[A]
//  ): c.MutableQueueExtensionMethods[A] =
//    new c.MutableQueueExtensionMethods[A](self)
//
//  implicit def toMapViewExtensionMethods[K, V, C <: scala.collection.Map[K, V]](
//    self: IterableView[(K, V), C]
//  ): c.MapViewExtensionMethods[K, V, C] =
//    new c.MapViewExtensionMethods[K, V, C](self)
}
