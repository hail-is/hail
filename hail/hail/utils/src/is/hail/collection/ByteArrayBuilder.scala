package is.hail.collection

final class ByteArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Byte] = new Array[Byte](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def setSizeUnchecked(n: Int): Unit =
    size_ = n

  def apply(i: Int): Byte = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Byte](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def +=(x: Byte) = add(x)

  def ++=(xs: Array[Byte]) = {
    val newLen = size + xs.length
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, xs.length)
    size_ = newLen
  }

  def ++=(xs: Array[Byte], n: Int) = {
    val newLen = size + n
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, n)
    size_ = newLen
  }

  def add(x: Byte): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Byte): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Byte] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Byte](initialCapacity)
  }

  def appendFrom(ab2: ByteArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Byte = {
    size_ -= 1
    b(size)
  }
}
