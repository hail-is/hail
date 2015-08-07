package org.broadinstitute.k3

import scala.language.implicitConversions

// FIXME don't zip, write direct iterators
class RichVector[T](v: Vector[T]) {
  def zipWith[T2, V](v2: Vector[T2], f: (T, T2) => V): Vector[V] = {
    v.iterator
    .zip(v2.iterator)
    .map(f.tupled)
    .toVector
  }

  def zipWithAndIndex[T2, V](v2: Vector[T2], f: (T, T2, Int) => V): Vector[V] = {
    v.iterator
    .zip(v2.iterator)
    .zipWithIndex
    .map { case ((e1, e2), i) => f(e1, e2, i) }
    .toVector
  }

  def zipWith[T2, T3, V](v2: Vector[T2], v3: Vector[T3], f: (T, T2, T3) => V): Vector[V] = {
    v.iterator
    .zip(v2.iterator)
    .zip(v3.iterator)
    .map { case ((e1, e2), e3) => f(e1, e2, e3) }
    .toVector
  }
}

object Utils {
  implicit def toRichVector[T](v: Vector[T]): RichVector[T] = new RichVector(v)
}