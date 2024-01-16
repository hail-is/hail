package is.hail.utils

import scala.collection.mutable

class BytePacker {
  val slots = new mutable.TreeSet[(Long, Long)]

  def insertSpace(size: Long, start: Long): Unit = {
    slots += size -> start
  }

  def getSpace(size: Long, alignment: Long): Option[Long] = {

    // disregard spaces smaller than size
    slots.iteratorFrom(size -> 0)
      .foreach { x =>
        val spaceSize = x._1
        val start = x._2

        // find a start position with the proper alignment
        var i = 0L
        while (i + size <= spaceSize) {
          // space found
          if ((start + i) % alignment == 0) {
            // remove this space from the tree of open slots
            slots -= x
            if (i > 0) {
              // append any space before the insertion location
              slots += i -> start
            }
            val endGap = spaceSize - size - i
            if (endGap > 0) {
              // append any space after the insertion location
              slots += endGap -> (start + i + size)
            }
            return Some(start + i)
          }
          i += 1
        }
      }
    None
  }
}
