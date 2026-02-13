package is.hail.collection

final class BooleanArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Boolean] = new Array[Boolean](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Boolean = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Boolean](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def +=(x: Boolean) = add(x)

  def add(x: Boolean): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Boolean): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Boolean] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Boolean](initialCapacity)
  }

  def appendFrom(ab2: BooleanArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Boolean = {
    size_ -= 1
    b(size)
  }
}
