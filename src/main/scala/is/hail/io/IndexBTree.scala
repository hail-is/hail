package is.hail.io

import is.hail.utils._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs._

import scala.collection.mutable

object IndexBTree {

  def calcDepth(arr: Array[Long]) = math.max(1,(math.log10(arr.length) / math.log10(1024)).ceil.toInt) //max necessary for array of length 1 becomes depth=0

  def write(arr: Array[Long], fileName: String, hConf: Configuration) {
    require(arr.length > 0)

    hConf.writeDataFile(fileName) { w =>
      val depth = calcDepth(arr)

      // Write layers above last layer if needed -- padding of -1 included
      val layers = mutable.ArrayBuffer[IndexedSeq[Long]]()
      for (i <- 0 until depth - 1) {
        val multiplier = math.pow(1024, depth - 1 - i).toInt
        layers.append((0 until math.pow(1024, i + 1).toInt).map { j =>
          if (j * multiplier < arr.length)
            arr(j * multiplier)
          else
            -1L
        })
      }

      // Write last layer
      layers.append(arr)

      // Pad last layer so last block is 1024 elements (1024*8 bytes)
      val paddingRequired = 1024 - (arr.length % 1024)
      layers.append((0 until paddingRequired).map{_ => -1L})

      val bytes = layers.flatten.flatMap(l => Array[Byte](
        (l >>> 56).toByte,
        (l >>> 48).toByte,
        (l >>> 40).toByte,
        (l >>> 32).toByte,
        (l >>> 24).toByte,
        (l >>> 16).toByte,
        (l >>> 8).toByte,
        (l >>> 0).toByte)).toArray

      w.write(bytes)
    }
  }
}

class IndexBTree(indexFileName: String, hConf: Configuration) {
  val maxDepth = calcDepth()
  private val fs = try {
    hConf.fileSystem(indexFileName).open(new Path(indexFileName))
  } catch {
    case e: Exception => fatal("Could not find a BGEN .idx file at $indexFileName. Try running HailContext.index_bgen().", e)
  }

  def close() = fs.close()

  def calcDepth(): Int = {
    val numBtreeElements = hConf.getFileSize(indexFileName) / 8
    var depth = 1
    while (numBtreeElements > math.pow(1024,depth).toInt) {
      depth += 1
    }
    depth
  }

  private def getOffset(depth: Int): Long = {
    (1 until depth).map(math.pow(1024,_).toLong * 8).sum
  }

  private def getOffset(depth: Int, blockIndex: Long): Long = {
    getOffset(depth) + blockIndex * 8 * 1024
  }

  private def traverseTree(query: Long, startIndex: Long, currentDepth: Int): Long = {

    def searchBlock(): Long = {
      def read(prevValue: Long, prevPos: Long): Long = {
        val currValue = fs.readLong()

        if (currentDepth != maxDepth && query >= prevValue && (query < currValue || currValue == -1L))
          prevPos
        else if (currentDepth == maxDepth && query <= currValue || currValue == -1L)
          currValue
        else if (prevPos >= (startIndex + 1024*8))
          fatal("did not find query in block")
        else
          read(currValue, prevPos + 8)
      }

      fs.seek(startIndex)
      val firstValue = fs.readLong()
      if (currentDepth != maxDepth && query >= 0L && query <= firstValue)
        startIndex
      else if (currentDepth == maxDepth && query >= 0L && query <= firstValue)
        firstValue
      else
        read(firstValue, startIndex)
    }

    if (currentDepth == maxDepth)
      searchBlock()
    else {
      val matchPosition = searchBlock()
      val blockIndex = (matchPosition - getOffset(currentDepth)) / 8
      val newStart = getOffset(currentDepth + 1, blockIndex)
      traverseTree(query, newStart, currentDepth + 1)
    }
  }

  def queryIndex(query: Long): Option[Long] = {
    require( query >= 0 )

    val result = traverseTree(query, 0L, 1)

    if (result != -1L)
      Option(result)
    else
      None
  }
}
