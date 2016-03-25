package org.broadinstitute.hail.io

import java.io.{EOFException, OutputStream, InputStream}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream}
import scala.collection.mutable
import org.broadinstitute.hail.Utils._

object IndexBTree {

  def write(arr: Array[Long], fileName: String, hConf: Configuration) {
    require(arr.length > 0)

    writeDataFile(fileName, hConf) { w =>
      val fs = new FSDataOutputStream(w)
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
      val paddingRequired = if (arr.length < 1024) 1024 - arr.length else arr.length % 1024
      layers.append((0 until paddingRequired).map{j => -1L})

      val bytes = layers.flatten.flatMap(l => Array[Byte](
        (l >>> 56).toByte,
        (l >>> 48).toByte,
        (l >>> 40).toByte,
        (l >>> 32).toByte,
        (l >>> 24).toByte,
        (l >>> 16).toByte,
        (l >>> 8).toByte,
        (l >>> 0).toByte)).toArray

      fs.write(bytes)
    }
  }

  def calcDepth(arr: Array[Long]) = math.max(1,(math.log10(arr.length) / math.log10(1024)).ceil.toInt) //max necessary for array of length 1 becomes depth=0

  def calcDepth(fileName: String, hConf: Configuration): Int = {
    val numBtreeElements = hadoopGetFileSize(fileName, hConf) / 8
    var depth = 1
    while (numBtreeElements > math.pow(1024,depth).toInt) {
      depth += 1
    }
    depth
  }

  private def getOffset(depth: Int): Int = {
    if (depth == 1)
      0
    else
    (1 until depth).map(math.pow(1024,_).toInt * 8).sum
  }

  private def getOffset(depth: Int, blockIndex: Int): Int = {
    getOffset(depth) + blockIndex * 8 * 1024
  }

  private def searchInternalBlock(query: Long, startIndex: Int, fs: FSDataInputStream): Int = {
    def read(prevItem: Long, localIndex: Int): Int = {
      val currItem = fs.readLong()
      if (query >= prevItem && (query < currItem || currItem == -1L))
        localIndex - 1
      else if (localIndex - startIndex >= 1024)
        fatal("did not find query in block")
      else
        read(currItem, localIndex + 1)
    }
    fs.seek(startIndex)
    read(0L, startIndex)
  }

  private def searchOuterBlock(query: Long, startIndex: Int, fs: FSDataInputStream): Long = {
    def read(localIndex: Int): Long = {
      val currItem = fs.readLong()
      if (query <= currItem || currItem == -1L)
        currItem
      else if (localIndex - startIndex >= 1024)
        fatal("did not find query in block")
      else
        read(localIndex + 1)
    }
    fs.seek(startIndex)
    read(startIndex)
  }

  def traverseTree(query: Long, startIndex: Int, currentDepth: Int, maxDepth: Int, fs: FSDataInputStream): Long = {
    if (currentDepth == maxDepth) {
      searchOuterBlock(query, startIndex, fs)
    }
    else {
      val blockIndex = searchInternalBlock(query, startIndex, fs)
      val newStart = getOffset(currentDepth + 1, blockIndex)
      traverseTree(query, newStart, currentDepth + 1, maxDepth, fs)
    }
  }

  def queryIndex(query: Long, originalFileSize: Long, indexFileName: String, hConf: Configuration): Long = {
    val maxDepth = calcDepth(indexFileName, hConf)

    if (query < 0)
      fatal("Query cannot be negative")
    if (query >= originalFileSize)
      fatal("Query cannot be larger than file size")

    readFile(indexFileName, hConf) { s =>
      val fs = new FSDataInputStream(s)
      traverseTree(query, 0, 1, maxDepth, fs)
    }
  }
}