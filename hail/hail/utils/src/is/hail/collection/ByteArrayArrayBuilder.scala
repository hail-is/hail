package is.hail.collection

final class ByteArrayArrayBuilder(initialCapacity: Int) {

  var size_ : Int = 0
  private var b: Array[Array[Byte]] = new Array[Array[Byte]](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Array[Byte] = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Array[Byte]](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: Array[Byte]): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Array[Byte]): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Array[Byte]] = b.slice(0, size_)
}
