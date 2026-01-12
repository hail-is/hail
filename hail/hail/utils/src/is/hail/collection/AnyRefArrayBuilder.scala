package is.hail.collection

import scala.reflect.ClassTag

final class AnyRefArrayBuilder[T <: AnyRef](initialCapacity: Int = 16)(implicit ct: ClassTag[T]) {

  var size_ : Int = 0
  var b: Array[T] = new Array[T](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): T = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[T](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: T): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def +=(x: T): Unit = add(x)

  def update(i: Int, x: T): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[T] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[T](initialCapacity)
  }

  def appendFrom(ab2: AnyRefArrayBuilder[T]): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): T = {
    size_ -= 1
    b(size)
  }

  def clearAndSetNull(): Unit = {
    clear()
    var i = 0
    while (i < b.length) {
      b(i) = (null.asInstanceOf[T])
      i += 1
    }
  }

  def setSizeUninitialized(size: Int): Unit = {
    ensureCapacity(size)
    size_ = size
  }
}
