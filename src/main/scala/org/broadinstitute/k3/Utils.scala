package org.broadinstitute.k3

import scala.collection.mutable.ArrayBuilder
import scala.language.implicitConversions

// FIXME don't zip, write direct iterators
class RichVector[T](v: Vector[T]) {
  // FIXME add zipExactWith
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

class RichTuple2[T](t: (T, T)) {
  def at(i: Int): T = {
    i match {
      case 1 => t._1
      case 2 => t._2
    }
  }
}

class RichTuple3[T](t: (T, T, T)) {
  def at(i: Int): T = {
    i match {
      case 1 => t._1
      case 2 => t._2
      case 3 => t._3
    }
  }
}

object Utils {
  implicit def toRichVector[T](v: Vector[T]): RichVector[T] = new RichVector(v)

  implicit def toRichTuple2[T](t: (T, T)): RichTuple2[T] = new RichTuple2(t)

  implicit def toRichTuple3[T](t: (T, T, T)): RichTuple3[T] = new RichTuple3(t)

  def writeULEB128(b: ArrayBuilder[Byte], x0: Int) {
    require(x0 >= 0)

    // FIXME functionalize
    var x = x0
    var more = true
    while (more) {
      var c = x & 0x7F
      x = x >>> 7

      if (x == 0)
        more = false
      else
        c = c | 0x80

      assert(c >= 0 && c <= 255)
      b += c.toByte
    }
  }

  def fail(): Unit = {
    assert(false)
  }
}
