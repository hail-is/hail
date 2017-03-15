package is.hail.utils

final class ByteArrayBuilder(initialCapacity: Int) extends ArrayBuilder[Byte] {
  def this() = this(16)
}

final class IntArrayBuilder(initialCapacity: Int) {
  private var a: Array[Int] = new Array[Int](initialCapacity)
  private var size_ : Int = 0

  def this() = this(16)

  def size: Int = size_

  def length: Int = size_

  def apply(i: Int): Int = {
    require(i >= 0 && i < size)
    a(i)
  }

  def ensureCapacity(n: Int) {
    if (a.length < n) {
      val newCapacity = (a.length * 2).max(n)
      val newa = new Array[Int](newCapacity)
      Array.copy(a, 0, newa, 0, size_)
      a = newa
    }
  }

  def clear(): Unit = {
    size_ = 0
  }

  def +=(x: Int) {
    ensureCapacity(size_ + 1)
    a(size_) = x
    size_ += 1
  }

  def result(): Array[Int] = {
    val r = new Array[Int](size_)
    Array.copy(a, 0, r, 0, size_)
    r
  }
}

final class DoubleArrayBuilder(initialCapacity: Int) {
  private var a: Array[Double] = new Array[Double](initialCapacity)
  private var size_ : Int = 0

  def this() = this(16)

  def size: Int = size_

  def length: Int = size_

  def apply(i: Int): Double = {
    require(i >= 0 && i < size)
    a(i)
  }

  def ensureCapacity(n: Int) {
    if (a.length < n) {
      val newCapacity = (a.length * 2).max(n)
      val newa = new Array[Double](newCapacity)
      Array.copy(a, 0, newa, 0, size_)
      a = newa
    }
  }

  def clear(): Unit = {
    size_ = 0
  }

  def +=(x: Double) {
    ensureCapacity(size_ + 1)
    a(size_) = x
    size_ += 1
  }

  def result(): Array[Double] = {
    val r = new Array[Double](size_)
    Array.copy(a, 0, r, 0, size_)
    r
  }
}