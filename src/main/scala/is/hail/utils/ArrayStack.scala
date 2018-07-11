package is.hail.utils

import scala.reflect.ClassTag

final class ArrayStack[@specialized T](hintSize: Int = 16)(implicit tct: ClassTag[T]) {
  private var a = new Array[T](hintSize)

  private[this] var size_ = 0

  def size: Int = size_

  def capacity: Int = a.length

  def isEmpty: Boolean = size_ == 0

  def nonEmpty: Boolean = size_ > 0

  def clear(): Unit = {
    size_ = 0
  }

  def top: T = {
    assert(size_ > 0)
    a(size_ - 1)
  }

  def push(x: T) {
    if (size_ == a.length) {
      val newA = new Array[T](size_ * 2)
      System.arraycopy(a, 0, newA, 0, size_)
      a = newA
    }
    a(size_) = x
    size_ += 1
  }

  def pop(): T = {
    assert(size_ > 0)
    size_ -= 1
    val x = a(size_)
    a(size_) = uninitialized[T]
    x
  }

  def update(i: Int, x: T) {
    assert(i >= 0 && i < size_)
    a(size_ - i - 1) = x
  }

  def apply(i: Int): T = {
    assert(i >= 0 && i < size_)
    a(size_ - i - 1)
  }
}
