package is.hail.utils

import scala.reflect.ClassTag

// volatility means it is undefined behavior to use the value returned by
// `apply` after a subsequent call is made to `apply`.
trait VolatileIndexedSeq[T] { self =>
  def length(): Int

  def apply(i: Int): T

  def map[U](f: T => U): VolatileIndexedSeq[U] = new VolatileIndexedSeq[U]() {
    def length = self.length
    def apply(i: Int): U = f(self.apply(i))
  }

  def volatileIterator: VolatileIterator[T] = new VolatileIterator[T]() {
    private var i: Int = 0
    def hasNext: Boolean = i < self.length
    def next: T = {
      val v = self(i)
      i += 1
      v
    }
  }

  def zipWithIndex: VolatileIndexedSeq[(T, Int)] = new VolatileIndexedSeq[(T, Int)]() {
    def length(): Int = self.length()
    def apply(i: Int): (T, Int) = (self.apply(i), i)
  }

  def filter(f: T => Boolean): VolatileIterator[T] =
    volatileIterator.filter(f)

  def forall(f: T => Boolean): Boolean = {
    var i = 0
    while (i < length) {
      if (!f(this.apply(i)))
        return false
      i += 1
    }
    return true
  }

  def exists(f: T => Boolean): Boolean = {
    var i = 0
    while (i < length) {
      if (f(this.apply(i)))
        return true
      i += 1
    }
    return false
  }

  def foreach(f: T => Unit) {
    var i = 0
    while (i < length) {
      f(this.apply(i))
      i += 1
    }
  }

  // copy *must not* store a copy to a reference of T unless T is immutable
  def toArray[U](copy: T => U)(implicit tct: ClassTag[U]): Array[U] = {
    var i = 0
    val a = new Array[U](length)
    while (i < length) {
      a(i) = copy(this.apply(i))
      i += 1
    }
    a
  }

  def toArray(implicit tct: ClassTag[T], prim: NonVolatilePrimitive[T]): Array[T] =
    toArray(x => x)
}

class ConcreteVolatileIndexedSeq[T](ts: IndexedSeq[T]) extends VolatileIndexedSeq[T] {
  val length = ts.length
  def apply(i: Int): T = ts(i)
}
