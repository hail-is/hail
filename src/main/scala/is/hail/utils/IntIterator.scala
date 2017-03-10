package is.hail.utils

import scala.collection.mutable

abstract class IntIterator {
  self =>

  def nextInt(): Int

  def hasNext: Boolean

  def toArray: Array[Int] = {
    val b = new mutable.ArrayBuilder.ofInt
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
