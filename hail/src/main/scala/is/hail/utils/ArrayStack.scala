package is.hail.utils

import scala.reflect.ClassTag

final class ObjectArrayStack[T <: AnyRef](hintSize: Int = 16)(implicit tct: ClassTag[T]) {
  private var a = new Array[T](hintSize)

  private[this] var size_ = 0

  def size: Int = size_

  def capacity: Int = a.length

  def isEmpty: Boolean = size_ == 0

  def nonEmpty: Boolean = size_ > 0

  def clear(): Unit =
    size_ = 0

  def top: T = {
    assert(size_ > 0)
    a(size_ - 1)
  }

  def topOption: Option[T] = if (size_ > 0) Some(top) else None

  def push(x: T): Unit = {
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

  def update(i: Int, x: T): Unit = {
    assert(i >= 0 && i < size_)
    a(size_ - i - 1) = x
  }

  def apply(i: Int): T = {
    assert(i >= 0 && i < size_)
    a(size_ - i - 1)
  }

  def toArray: Array[T] = (0 until size).map(apply).toArray
}

final class LongArrayStack(hintSize: Int = 16) {
  private var a = new Array[Long](hintSize)

  private[this] var size_ = 0

  def size: Int = size_

  def capacity: Int = a.length

  def isEmpty: Boolean = size_ == 0

  def nonEmpty: Boolean = size_ > 0

  def clear(): Unit =
    size_ = 0

  def top: Long = {
    assert(size_ > 0)
    a(size_ - 1)
  }

  def topOption: Option[Long] = if (size_ > 0) Some(top) else None

  def push(x: Long): Unit = {
    if (size_ == a.length) {
      val newA = new Array[Long](size_ * 2)
      System.arraycopy(a, 0, newA, 0, size_)
      a = newA
    }
    a(size_) = x
    size_ += 1
  }

  def pop(): Long = {
    assert(size_ > 0)
    size_ -= 1
    val x = a(size_)
    a(size_) = uninitialized[Long]
    x
  }

  def update(i: Int, x: Long): Unit = {
    assert(i >= 0 && i < size_)
    a(size_ - i - 1) = x
  }

  def apply(i: Int): Long = {
    assert(i >= 0 && i < size_)
    a(size_ - i - 1)
  }

  def toArray: Array[Long] = (0 until size).map(apply).toArray
}

final class IntArrayStack(hintSize: Int = 16) {
  private var a = new Array[Int](hintSize)

  private[this] var size_ = 0

  def size: Int = size_

  def capacity: Int = a.length

  def isEmpty: Boolean = size_ == 0

  def nonEmpty: Boolean = size_ > 0

  def clear(): Unit =
    size_ = 0

  def top: Int = {
    assert(size_ > 0)
    a(size_ - 1)
  }

  def topOption: Option[Int] = if (size_ > 0) Some(top) else None

  def push(x: Int): Unit = {
    if (size_ == a.length) {
      val newA = new Array[Int](size_ * 2)
      System.arraycopy(a, 0, newA, 0, size_)
      a = newA
    }
    a(size_) = x
    size_ += 1
  }

  def pop(): Int = {
    assert(size_ > 0)
    size_ -= 1
    val x = a(size_)
    a(size_) = uninitialized[Int]
    x
  }

  def update(i: Int, x: Int): Unit = {
    assert(i >= 0 && i < size_)
    a(size_ - i - 1) = x
  }

  def apply(i: Int): Int = {
    assert(i >= 0 && i < size_)
    a(size_ - i - 1)
  }

  def toArray: Array[Int] = (0 until size).map(apply).toArray
}
