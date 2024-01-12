package is.hail.io

import is.hail.io.fs.FS
import is.hail.utils._

import java.io.{Closeable, DataOutputStream}
import java.util.Arrays
import scala.collection.mutable

object IndexBTree {
  private[io] def calcDepth(internalAndExternalNodeCount: Long, branchingFactor: Int): Int = {
    var depth = 1
    var maximumTreeSize = branchingFactor.toLong
    while (internalAndExternalNodeCount > maximumTreeSize) {
      assert(depth <= 6) // 1024^7 > Long.MaxValue
      maximumTreeSize = maximumTreeSize * branchingFactor + branchingFactor
      depth += 1
    }
    depth
  }

  private[io] def calcDepth(arr: Array[Long], branchingFactor: Int) =
    // max necessary for array of length 1 becomes depth=0
    math.max(1, (math.log10(arr.length) / math.log10(branchingFactor)).ceil.toInt)

  private[io] def btreeLayers(
    arr: Array[Long],
    branchingFactor: Int = 1024,
  ): Array[Array[Long]] = {
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

    // Pad last layer so last block is branchingFactor elements (branchingFactor*8 bytes)
    val danglingElements = (arr.length % branchingFactor)
    val paddingRequired =
      if (danglingElements == 0) 0
      else branchingFactor - danglingElements
    val padding = (0 until paddingRequired).map(_ => -1L)
    // Write last layer
    layers.append(arr ++ padding)

    layers.map(_.toArray).toArray
  }

  private[io] def btreeBytes(
    arr: Array[Long],
    branchingFactor: Int = 1024,
  ): Array[Byte] = btreeLayers(arr, branchingFactor)
    .flatten
    .flatMap(l =>
      Array[Byte](
        (l >>> 56).toByte,
        (l >>> 48).toByte,
        (l >>> 40).toByte,
        (l >>> 32).toByte,
        (l >>> 24).toByte,
        (l >>> 16).toByte,
        (l >>> 8).toByte,
        (l >>> 0).toByte,
      )
    )
    .toArray

  def write(
    arr: Array[Long],
    fileName: String,
    fs: FS,
    branchingFactor: Int = 1024,
  ): Unit = using(new DataOutputStream(fs.create(fileName))) { w =>
    w.write(btreeBytes(arr, branchingFactor))
  }

  def toString(
    arr: Array[Long],
    branchingFactor: Int = 1024,
  ): String =
    btreeLayers(arr, branchingFactor).map(_.mkString("[", " ", "]")).mkString(
      "(BTREE\n",
      "\n",
      "\n)",
    )
}

class IndexBTree(indexFileName: String, fs: FS, branchingFactor: Int = 1024) extends Closeable {
  val maxDepth = calcDepth()

  private val is =
    try
      fs.openNoCompression(indexFileName)
    catch {
      case e: Exception => fatal(
          s"Could not find a BGEN .idx file at $indexFileName. Try running HailContext.index_bgen().",
          e,
        )
    }

  def close(): Unit = is.close()

  def calcDepth(): Int =
    IndexBTree.calcDepth(fs.getFileSize(indexFileName) / 8, branchingFactor)

  private def getOffset(depth: Int): Long =
    (1 until depth).map(math.pow(branchingFactor, _).toLong * 8).sum

  private def getOffset(depth: Int, blockIndex: Long): Long =
    getOffset(depth) + blockIndex * 8 * branchingFactor

  private def traverseTree(query: Long, startIndex: Long, currentDepth: Int): (Long, Long) = {

    def searchBlock(): Long = {
      def read(prevValue: Long, prevPos: Long): Long = {
        val currValue = is.readLong()

        if (
          currentDepth != maxDepth && query >= prevValue && (query < currValue || currValue == -1L)
        )
          prevPos
        else if (currentDepth == maxDepth && query <= currValue || currValue == -1L)
          currValue
        else if (prevPos >= (startIndex + branchingFactor * 8))
          fatal("did not find query in block")
        else
          read(currValue, prevPos + 8)
      }

      is.seek(startIndex)
      val firstValue = is.readLong()
      if (currentDepth != maxDepth && query >= 0L && query <= firstValue)
        startIndex
      else if (currentDepth == maxDepth && query >= 0L && query <= firstValue)
        firstValue
      else
        read(firstValue, startIndex)
    }

    def searchLastBlock(): (Long, Long) = {
      def read(prevValue: Long, prevPos: Long): (Long, Long) = {
        val currValue = is.readLong()

        if (query <= currValue || currValue == -1L)
          (prevPos + 8, currValue)
        else if (prevPos >= (startIndex + branchingFactor * 8))
          fatal("did not find query in block")
        else
          read(currValue, prevPos + 8)
      }

      is.seek(startIndex)
      val firstValue = is.readLong()
      if (query >= 0L && query <= firstValue)
        (startIndex, firstValue)
      else
        read(firstValue, startIndex)
    }

    if (currentDepth == maxDepth) {
      val (bytePosition, value) = searchLastBlock()
      val leadingBytes = getOffset(currentDepth)
      ((bytePosition - leadingBytes) / 8, value)
    } else {
      val matchPosition = searchBlock()
      val blockIndex = (matchPosition - getOffset(currentDepth)) / 8
      val newStart = getOffset(currentDepth + 1, blockIndex)
      traverseTree(query, newStart, currentDepth + 1)
    }
  }

  def queryIndex(query: Long): Option[Long] = {
    require(query >= 0)

    val (index, result) = traverseTree(query, 0L, 1)

    if (result != -1L)
      Option(result)
    else
      None
  }

  def queryArrayPositionAndFileOffset(query: Long): Option[(Long, Long)] = {
    require(query >= 0)

    val (index, result) = traverseTree(query, 0L, 1)

    if (result != -1L)
      Option((index, result))
    else
      None
  }
}

/** A BTree file of N elements is a sequence of layers containing 8-byte values.
  *
  * The size of layer i is {@code math.pow(branchingFactor, i + 1).toInt} . The last layer is the
  * first layer whose size is large enough to contain N elements. The final layer contains all N
  * elements, in their given order, followed by {@code branchingFactor - N} {@code -1} 's.
  */
// IndexBTree maps from a value to the next largest value, this treats the BTree
// like an on-disk array and looks up values by index
class OnDiskBTreeIndexToValue(
  path: String,
  fs: FS,
  branchingFactor: Int = 1024,
) extends AutoCloseable {
  private[this] def numLayers(size: Long): Int =
    IndexBTree.calcDepth(size, branchingFactor)

  private[this] def leadingElements(layer: Int): Long = {
    var i = 0
    var leadingElements = 0L
    while (i < layer) {
      leadingElements = leadingElements * branchingFactor + branchingFactor
      i += 1
    }
    leadingElements
  }

  private[this] val layers = numLayers(fs.getFileSize(path) / 8)
  private[this] val junk = leadingElements(layers - 1)

  private[this] var is =
    try {
      log.info("reading index file: " + path)
      fs.openNoCompression(path)
    } catch {
      case e: Exception =>
        fatal(s"Could not find a BGEN .idx file at $path. Try running HailContext.index_bgen().", e)
    }

  // WARNING: mutatively sorts the provided array
  def positionOfVariants(indices: Array[Int]): Array[Long] = {
    val a = new Array[Long](indices.length)
    if (indices.length == 0) {
      a
    } else {
      Arrays.sort(indices)
      is.seek((junk + indices(0)) * 8)
      a(0) = is.readLong()
      assert(a(0) != -1)
      var i = 1
      while (i < indices.length) {
        if (indices(i) == indices(i - 1)) {
          a(i) = a(i - 1)
        } else {
          val jump = (indices(i) - indices(i - 1) - 1) * 8
          assert(jump >= 0)
          is.skipBytes(jump)
          a(i) = is.readLong()
          assert(a(i) != -1)
        }
        i += 1
      }
      a
    }
  }

  override def close(): Unit = synchronized {
    if (is != null) {
      is.close()
      is = null
    }
  }
}
