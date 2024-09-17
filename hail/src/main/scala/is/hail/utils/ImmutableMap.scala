package is.hail.utils

import scala.collection.mutable

case class ImmutableMap[K, V](m: Map[K, V]) extends mutable.AbstractMap[K, V] {
  override def +=(kv: (K, V)): ImmutableMap.this.type =
    throw new UnsupportedOperationException()

  override def -=(key: K): ImmutableMap.this.type =
    throw new UnsupportedOperationException()

  override def get(key: K): Option[V] =
    m.get(key)

  override def iterator: Iterator[(K, V)] =
    m.iterator

  override def toMap[T, U](implicit ev: (K, V) <:< (T, U)): Map[T, U] =
    m.toMap
}

object ImmutableMap {
  def empty[K, V]: ImmutableMap[K, V] =
    ImmutableMap(Map.empty)
}
