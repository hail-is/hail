package is.hail.utils

import scala.collection.mutable

abstract class IntIterator {
  def nextInt(): Int
  def hasNext: Boolean

  // requires that hasNext is called exactly once between each call to nextInt
  def unsafeFilter(that: Iterator[Boolean]): IntIterator = new IntIterator {
      def nextInt(): Int = IntIterator.this.nextInt()

      def hasNext: Boolean = {
        while (that.hasNext && IntIterator.this.hasNext) {
          if (that.next())
            return true
          else
            IntIterator.this.nextInt()
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

  def toBoxedIntIterator: Iterator[Int] = new Iterator[Int] {
      def hasNext: Boolean = IntIterator.this.hasNext
      def next(): Int = IntIterator.this.nextInt()
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
