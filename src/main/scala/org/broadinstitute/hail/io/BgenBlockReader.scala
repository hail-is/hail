package org.broadinstitute.hail.io

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.mapred.{InvalidFileTypeException, FileSplit}
import scala.collection.mutable.ArrayBuffer
import org.apache.hadoop.mapred._

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader(job, split) {

  override def getIndices(start: Long, end: Long): Array[BlockIndex] = {
    val path = job.get("idx")
    val bfr = new BinaryFileReader(path)
    val posAndInd = ArrayBuffer[(Long, Int)]()

    val nVariants = bfr.readIntBE()
    var i: Int = 0
    var pos = bfr.readLongBE()
    var firstI: Int = Integer.MAX_VALUE
    var lastI: Int = -1
    while (i < nVariants & pos <= end) {
      if (pos >= start) {
        if (i < firstI)
          firstI = i
        if (i > lastI)
          lastI = i
        posAndInd += ((pos, i))
      }
      pos = bfr.readLongBE()
      i += 1
    }
    println(s"Indices $firstI-$lastI")
    // Now we need one more so that we know how far to read from the last index
    posAndInd += ((bfr.readLongBE(), i+1))

    // - 1 because we don't need the unread portion for this array
    val result: Array[BlockIndex] = Array.ofDim[BlockIndex](posAndInd.length - 1)
    for (j <- result.indices) {
      val thisPos: Long = posAndInd(j)._1
      val nextPos: Long = posAndInd(j+1)._1
      val thisInd: Int = posAndInd(j)._2
      result(j) = BlockIndex(thisPos, (nextPos-thisPos).toInt, thisInd)
    }

    if (result.length < 1)
      throw new InvalidFileTypeException()
    result
  }
}
