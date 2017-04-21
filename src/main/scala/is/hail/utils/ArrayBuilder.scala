package is.hail.utils

import scala.reflect.ClassTag

class ArrayBuilder[@specialized T](initialCapacity: Int)(implicit tct: ClassTag[T]) {
  private var b: Array[T] = new Array[T](initialCapacity)
  private var size_ : Int = 0

  def this()(implicit tct: ClassTag[T]) = this(16)

  def size: Int = size_

  def length: Int = size_

  def apply(i: Int): T = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int) {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[T](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def clear(): Unit = {
    size_ = 0
  }

  def +=(x: T) {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def ++=(x: Traversable[T]) {
    x.foreach(t => this.+=(t))
  }

  def result(): Array[T] = {
    val r = new Array[T](size_)
    Array.copy(b, 0, r, 0, size_)
    r
  }
}
