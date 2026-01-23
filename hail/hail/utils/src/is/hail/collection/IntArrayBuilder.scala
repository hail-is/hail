package is.hail.collection

final class IntArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Int] = new Array[Int](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Int = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Int](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def setSizeUninitialized(n: Int) = {
    ensureCapacity(n)
    size_ = n
  }

  def +=(x: Int): Unit = add(x)

  def ++=(xs: Array[Int]) = {
    val newLen = size + xs.length
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, xs.length)
    size_ = newLen
  }

  def ++=(xs: Array[Int], n: Int) = {
    val newLen = size + n
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, n)
    size_ = newLen
  }

  def add(x: Int): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Int): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Int] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Int](initialCapacity)
  }

  def appendFrom(ab2: IntArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Int = {
    size_ -= 1
    b(size)
  }
}
