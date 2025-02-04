package is.hail.utils

import scala.reflect.ClassTag

final class MissingArrayBuilder[@specialized T](initialCapacity: Int)(implicit tct: ClassTag[T]) {
  private var b: Array[T] = new Array[T](initialCapacity)
  private var missing: Array[Boolean] = new Array[Boolean](initialCapacity)
  private var size_ : Int = 0

  def this()(implicit tct: ClassTag[T]) = this(BoxedArrayBuilder.defaultInitialCapacity)

  def size: Int = size_

  def length: Int = size_

  def isEmpty: Boolean = size_ == 0

  def apply(i: Int): T = {
    require(i >= 0 && i < size)
    b(i)
  }

  def update(i: Int, x: T): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[T](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
      val newmissing = new Array[Boolean](newCapacity)
      Array.copy(missing, 0, newmissing, 0, size_)
      missing = newmissing
    }
  }

  def clear(): Unit =
    size_ = 0

  def +=(x: T): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    missing(size_) = false
    size_ += 1
  }

  def ++=(s: Seq[T]): Unit = s.foreach(x => this += x)

  def ++=(a: Array[T]): Unit = ++=(a, a.length)

  def ++=(a: Array[T], length: Int): Unit = {
    require(length >= 0 && length <= a.length)
    ensureCapacity(size_ + length)
    System.arraycopy(a, 0, b, size_, length)
    var i = 0
    while (i < length) {
      missing(size_ + i) = false
      i += 1
    }
    size_ += length
  }

  def underlying(): Array[T] = b

  def isMissing(i: Int): Boolean = {
    require(i >= 0 && i < size)
    missing(i)
  }

  def setMissing(i: Int, m: Boolean): Unit = {
    require(i >= 0 && i < size)
    missing(i) = m
  }

  def addMissing(): Unit = {
    ensureCapacity(size_ + 1)
    missing(size_) = true
    size_ += 1
  }
}
