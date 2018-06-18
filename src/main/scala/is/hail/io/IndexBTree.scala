package is.hail.io

import is.hail.utils._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs._

import scala.collection.mutable

object IndexBTree {
  private[io] def calcDepth(arr: Array[Long], branchingFactor: Int) =
    //max necessary for array of length 1 becomes depth=0
    math.max(1, (math.log10(arr.length) / math.log10(branchingFactor)).ceil.toInt)

  private[io] def btreeBytes(
    arr: Array[Long],
    branchingFactor: Int = 1024
  ): Array[Byte] = {
    require(arr.length > 0)

    val depth = calcDepth(arr, branchingFactor)

    // Write layers above last layer if needed -- padding of -1 included
    val layers = mutable.ArrayBuffer[IndexedSeq[Long]]()
    for (i <- 0 until depth - 1) {
      val multiplier = math.pow(branchingFactor, depth - 1 - i).toInt
      layers.append((0 until math.pow(branchingFactor, i + 1).toInt).map { j =>
        if (j * multiplier < arr.length)
          arr(j * multiplier)
        else
          -1L
      })
    }

    // Write last layer
    layers.append(arr)

    // Pad last layer so last block is branchingFactor elements (branchingFactor*8 bytes)
    val danglingElements = (arr.length % branchingFactor)
    val paddingRequired =
      if (danglingElements == 0) 0
      else branchingFactor - danglingElements
    layers.append((0 until paddingRequired).map { _ => -1L })

    val bytes = layers.flatten.flatMap(l => Array[Byte](
      (l >>> 56).toByte,
      (l >>> 48).toByte,
      (l >>> 40).toByte,
      (l >>> 32).toByte,
      (l >>> 24).toByte,
      (l >>> 16).toByte,
      (l >>> 8).toByte,
      (l >>> 0).toByte)).toArray

    bytes
  }

  def write(
    arr: Array[Long],
    fileName: String,
    hConf: Configuration,
    branchingFactor: Int = 1024
  ): Unit = hConf.writeDataFile(fileName) { w =>
    w.write(btreeBytes(arr, branchingFactor))
  }
}

class IndexBTree(indexFileName: String, hConf: Configuration, branchingFactor: Int = 1024) {
  val maxDepth = calcDepth()
  private val fs = try {
    hConf.fileSystem(indexFileName).open(new Path(indexFileName))
  } catch {
    case e: Exception => fatal(s"Could not find a BGEN .idx file at $indexFileName. Try running HailContext.index_bgen().", e)
  }

  def close() = fs.close()

  def calcDepth(): Int = {
    val numBtreeElements = hConf.getFileSize(indexFileName) / 8
    var depth = 1
    while (numBtreeElements > math.pow(branchingFactor, depth).toInt) {
      depth += 1
    }
    depth
  }

  private def getOffset(depth: Int): Long = {
    (1 until depth).map(math.pow(branchingFactor, _).toLong * 8).sum
  }

  private def getOffset(depth: Int, blockIndex: Long): Long = {
    getOffset(depth) + blockIndex * 8 * branchingFactor
  }

  private def traverseTree(query: Long, startIndex: Long, currentDepth: Int): Long = {

    def searchBlock(): Long = {
      def read(prevValue: Long, prevPos: Long): Long = {
        val currValue = fs.readLong()

        if (currentDepth != maxDepth && query >= prevValue && (query < currValue || currValue == -1L))
          prevPos
        else if (currentDepth == maxDepth && query <= currValue || currValue == -1L)
          currValue
        else if (prevPos >= (startIndex + branchingFactor * 8))
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
    require(query >= 0)

    val result = traverseTree(query, 0L, 1)

    if (result != -1L)
      Option(result)
    else
      None
  }
}

class IndexBTree2(indexFileName: String, hConf: Configuration, nVariants: Int) {
  private[this] def numLayers(size: Long): Int = {
    if (size <= 1024)
      1
    else
      (math.log(size) / math.log(1024)).ceil.toInt
  }

  private[this] def leadingElements(layer: Int): Long = {
    var i = 0
    var leadingElements = 0L
    while (i < layer - 1) {
      leadingElements = leadingElements * 1024L + 1024L
      i += 1
    }
    leadingElements
  }

  // the last layer is literally all the data inline
  private[this] val layers = numLayers(hConf.getFileSize(indexFileName) / 8)
  private[this] val junk = leadingElements(layers)
  private[this] val variantPosition = try {
    using(hConf.fileSystem(indexFileName).open(new Path(indexFileName))) { fs =>
      fs.seek(junk * 8) // we have junk Longs
      Array.tabulate(nVariants) { _ =>
        fs.readLong()
      }
    }
  } catch {
    case e: Exception =>
      fatal(s"Could not find a BGEN .idx file at $indexFileName. Try running HailContext.index_bgen().", e)
  }

  def positionOfVariant(index: Int): Long =
    variantPosition(index)
}
