package is.hail.utils

import scala.collection.mutable

abstract class IntIterator {
  self =>

  def nextInt(): Int

  def hasNext: Boolean

  // requires that hasNext is called exactly once before each call to nextInt
  // requires that `this` and `that` are of equal length
  def unsafeFilter(that: Iterator[Boolean]): IntIterator = new IntIterator {
    def nextInt(): Int = self.nextInt()

    def hasNext: Boolean = {
      while (that.hasNext) {
        if (that.next())
          return true
        else
          self.nextInt()
      }
      false
    }
  }

  def toArray: Array[Int] = {
    val b = new mutable.ArrayBuilder.ofInt
    while (hasNext)
      b += nextInt()
    b.result()
  }

  def foreach(f: Int => Unit) {
    while (hasNext)
      f(nextInt())
  }

  def countNonNegative(): Int = {
    var count = 0
    while (hasNext) {
      if (nextInt() >= 0) count += 1
    }
    count
  }
}
