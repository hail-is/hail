package is.hail.utils

final class ByteArrayBuilder(initialCapacity: Int) {
  private var b: Array[Byte] = new Array[Byte](initialCapacity)
  private var size_ : Int = 0

  def this() = this(16)

  def size: Int = size_

  def length: Int = size_

  def apply(i: Int): Byte = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int) {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Byte](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def clear(): Unit = {
    size_ = 0
  }

  def +=(x: Byte) {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def result(): Array[Byte] = {
    val r = new Array[Byte](size_)
    Array.copy(b, 0, r, 0, size_)
    r
  }
}
