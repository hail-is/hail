package is.hail.utils

import scala.reflect.ClassTag

class ArrayStack[@specialized T](hintSize: Int = 16)(implicit tct: ClassTag[T]) {
  var a = new Array[T](hintSize)
  var size = 0

  def isEmpty: Boolean = size == 0

  def nonEmpty: Boolean = size > 0

  def head: T = {
    assert(size > 0)
    a(size - 1)
  }

  def push(x: T) {
    if (size == a.length) {
      val newA = new Array[T](size * 2)
      System.arraycopy(a, 0, newA, 0, size)
      a = newA
    }
    a(size) = x
    size += 1
  }

  def pop(): T = {
    assert(size > 0)
    size -= 1
    val x = a(size)
    a(size) = uninitialized[T]
    x
  }

  def update(i: Int, x: T) {
    assert(i >= 0 && i < size)
    a(size - i - 1) = x
  }

  def apply(i: Int): T = {
    assert(i >= 0 && i < size)
    a(size - i - 1)
  }
}
