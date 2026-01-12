package is.hail.collection

final class StringArrayBuilder(initialCapacity: Int = 16) {
  var size_ : Int = 0
  var b: Array[String] = new Array[String](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): String = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[String](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def +=(x: String) = add(x)

  def add(x: String): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: String): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[String] = {
    val a = new Array[String](size_)
    System.arraycopy(b, 0, a, 0, size_)
    a
  }

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[String](initialCapacity)
  }

  def appendFrom(ab2: StringArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): String = {
    size_ -= 1
    b(size)
  }
}
