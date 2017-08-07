package is.hail.utils

import scala.collection.mutable

class BytePacker {
  val slots = new mutable.TreeSet[(Int, Int)]

  def insertSpace(size: Int, start: Int) {
    slots += size -> start
  }

  def getSpace(size: Int, alignment: Int): Option[Int] = {

    // disregard spaces smaller than size
    slots.iteratorFrom(size -> 0)
      .foreach { x =>
        val spaceSize = x._1
        val start = x._2

        // find a start position with the proper alignment
        var i = 0
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