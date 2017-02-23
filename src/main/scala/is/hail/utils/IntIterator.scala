package is.hail.utils

import scala.collection.mutable

abstract class IntIterator {
  def nextInt(): Int
  def hasNext: Boolean

  // requires that hasNext is called exactly once between each call to nextInt
  def unsafeFilter(that: Iterator[Boolean]): IntIterator = {
    val it = this

    new IntIterator {
      def nextInt(): Int = it.nextInt()

      def hasNext: Boolean = {
        while (that.hasNext && it.hasNext) {
          if (that.next())
            return true
          else
            it.nextInt()
        }
        false
      }
    }
  }

  def toArray: Array[Int] = {
    val b = new mutable.ArrayBuilder.ofInt
    while (hasNext)
      b += nextInt()
    b.result()
  }

  def toBoxedIntIterator: Iterator[Int] = {
    val it = this
    new Iterator[Int] {
      def hasNext: Boolean = it.hasNext
      def next(): Int = it.nextInt()
    }
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
