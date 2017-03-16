package is.hail.utils

abstract class IntIterator {
  def nextInt(): Int

  def hasNext: Boolean

  def toArray: Array[Int] = {
    val b = new ArrayBuilder[Int]()
    while (hasNext)
      b += nextInt()
    b.result()
  }

  def countNonNegative(): Int = {
    var count = 0
    while (hasNext) {
      if (nextInt() >= 0) count += 1
    }
    count
  }
}
