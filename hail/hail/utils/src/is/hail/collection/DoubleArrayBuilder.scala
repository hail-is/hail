package is.hail.collection

final class DoubleArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Double] = new Array[Double](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Double = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Double](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def setSizeUninitialized(n: Int) = {
    ensureCapacity(n)
    size_ = n
  }

  def +=(x: Double): Unit = add(x)

  def ++=(xs: Array[Double]) = {
    val newLen = size + xs.length
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, xs.length)
    size_ = newLen
  }

  def ++=(xs: Array[Double], n: Int) = {
    val newLen = size + n
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, n)
    size_ = newLen
  }

  def add(x: Double): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Double): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Double] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Double](initialCapacity)
  }

  def appendFrom(ab2: DoubleArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Double = {
    size_ -= 1
    b(size)
  }
}
