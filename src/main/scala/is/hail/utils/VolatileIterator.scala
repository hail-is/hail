package is.hail.utils

import scala.reflect.ClassTag

// volatility means it is undefined behavior to use the value returned by
// `next` after a subsequent call is made to `next`.
trait VolatileIterator[T] { self =>
  def hasNext(): Boolean

  def next(): T

  def map[U](f: T => U): VolatileIterator[U] = new VolatileIterator[U]() {
    def hasNext() = self.hasNext

    def next(): U = f(self.next)
  }

  def filter(f: T => Boolean): VolatileIterator[T] = new VolatileIterator[T]() {
    private var v: T = _
    private var stale: Boolean = false
    private var present: Boolean = self.hasNext
    if (present)
      v = self.next

    private def advance() {
      if (self.hasNext) {
        v = self.next
        var keep = f(v)
        while (self.hasNext && !keep) {
          v = self.next
          keep = f(v)
        }
        present = keep
      } else
        present = false
      stale = false
    }

    def hasNext(): Boolean = {
      if (stale)
        advance()
      present
    }

    def next(): T = {
      if (stale)
        advance()
      stale = true
      v
    }
  }

  def forall(f: T => Boolean): Boolean = {
    while (hasNext)
      if (!f(next()))
        return false
    return true
  }

  def exists(f: T => Boolean): Boolean = {
    while (hasNext)
      if (f(next()))
        return true
    return false
  }

  def foreach(f: T => Unit) {
    while (hasNext()) {
      f(next())
    }
  }

  private def unsafeMin(lt: (T, T) => Boolean): T = {
    if (hasNext) {
      var min = self.next
      foreach { x =>
        if (lt(x, min))
          min = x
      }
      min
    } else
      throw new RuntimeException("minimum of empty iterator is undefined")
  }

  def min(implicit ord: Ordering[T], prim: Primitive[T]): T =
    unsafeMin(ord.lt)

  def max(implicit ord: Ordering[T], prim: Primitive[T]): T =
    unsafeMin(ord.gt)

  private def unsafeToArray(implicit tct: ClassTag[T]): Array[T] = {
    val a = new ArrayBuilder[T]()
    var i = 0
    foreach { x =>
      a(i) = x
      i += 1
    }
    a.result()
  }

  def toArray(implicit prim: Primitive[T], tct: ClassTag[T]): Array[T] =
    unsafeToArray
}

sealed trait Primitive[T] { }

object VolatileIterator {
  implicit object StringPrimitive extends Primitive[String]
  implicit object IntPrimitive extends Primitive[Int]
  implicit object LongPrimitive extends Primitive[Long]
  implicit object FloatPrimitive extends Primitive[Float]
  implicit object DoublePrimitive extends Primitive[Double]
}
