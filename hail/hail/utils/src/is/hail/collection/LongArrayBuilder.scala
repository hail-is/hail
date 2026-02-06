package is.hail.collection

final class LongArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Long] = new Array[Long](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Long = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, (b.length * 2))
      val newb = new Array[Long](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def +=(x: Long): Unit = add(x)

  def ++=(xs: Array[Long]) = {
    val newLen = size + xs.length
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, xs.length)
    size_ = newLen
  }

  def ++=(xs: Array[Long], n: Int) = {
    val newLen = size + n
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, n)
    size_ = newLen
  }

  def add(x: Long): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Long): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Long] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Long](initialCapacity)
  }

  def appendFrom(ab2: LongArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Long = {
    size_ -= 1
    b(size)
  }
}
