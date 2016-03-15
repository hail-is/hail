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

      // first calculate the necessary number of layers in the tree -- log1024(arr.length) rounded up
      val depth = calcDepth(arr)
      //println(s"write depth=$depth")

      // now downsample the array -- each downsampling should use the smallest index contained -- 0-1023 should use 0, etc
      val layers = mutable.ArrayBuffer[IndexedSeq[Long]]()
      for (i <- 0 until depth - 1) {

        val multiplier = math.pow(1024, depth - 1 - i).toInt
        //println(s"i = $i, depth=$depth mult = $multiplier")
        layers.append((0 until math.pow(1024, i + 1).toInt).map { j =>
          if (j * multiplier < arr.length) //should this be <=
            arr(j * multiplier)
          else
            -1
        })
      }
      layers.append(arr)

      //layers.zipWithIndex.foreach { case (a, i) => println(s"index $i size is ${a.size}"); println(s"first5:${a.take(5).mkString(",")}"); println(s"last5:${a.takeRight(5).mkString(",")}")}
          //println("After flatten: " + flat.size)
      val bytes = layers.flatten.flatMap(l => Array[Byte](
        (l >>> 56).toByte,
        (l >>> 48).toByte,
        (l >>> 40).toByte,
        (l >>> 32).toByte,
        (l >>> 24).toByte,
        (l >>> 16).toByte,
        (l >>> 8).toByte,
        (l >>> 0).toByte)).toArray
      //println(s"numLayers:${layers.size} bytesArraySize: ${bytes.length}")
      fs.writeLong(depth)
      fs.write(bytes)
    }
  }

  def calcDepth(arr: Array[Long]) = math.max(1,(math.log10(arr.length) / math.log10(1024)).ceil.toInt) //max necessary for array of length 1 becomes depth=0

//  def calcBlockStarts(depth: Int, layer: Int) = {
//    val multiplier = math.pow(1024, depth - 1 - layer).toInt
//    (0 until math.pow(1024, layer + 1).toInt).map { j => (j*multiplier, (j+1)*multiplier)}
//  }

//  private def searchBlock(query: Long, currentDepth: Int, startIndex: Int, endIndex: Int, fs: FSDataInputStream): Int = {
//    def getIndex(previousPosition: Long, index: Int): Int = {
//      val currentPosition: Long = fs.readLong()
//      println(s"query:$query startIndex:$startIndex endIndex:$endIndex previousPosition:$previousPosition currentPosition:$currentPosition")
//      if (query >= previousPosition && query < currentPosition)
//        index
//      else
//        getIndex(currentPosition, index+1)
//    }
//
//    fs.seek(getOffset(currentDepth,startIndex))
//    getIndex(0L, startIndex)
//  }

  private def getOffset(depth: Int): Int = {
    (0 until depth).map(math.pow(1024,_).toInt * 8).sum
  }

  private def getOffset(depth: Int, blockIndex: Int): Int = {
    getOffset(depth) + blockIndex * 8
  }

  private def searchBlock(query: Long, currentDepth: Int, maxDepth: Int, startIndex: Int, fs: FSDataInputStream): (Int, Long) = {
    def read(prevItem: Long, localIndex: Int): (Int,Long) = {
      val currItem = fs.readLong()
      println(s"read -- prevItem:$prevItem currItem:$currItem localIndex=$localIndex query:$query currDepth:$currentDepth maxDepth:$maxDepth startIndex:$startIndex")
      if (query >= prevItem && query < currItem)
        (localIndex, currItem)
      else if (localIndex - startIndex >= 1024 || currItem == -1)
        fatal("did not find query in block")
      else
        read(currItem, localIndex + 1)
    }
    val offset = getOffset(currentDepth,startIndex)
    println(s"searchBlock -- offset:$offset currDepth:$currentDepth maxDepth:$maxDepth startIndex:$startIndex")
    fs.seek(getOffset(currentDepth,startIndex))
    read(0L, startIndex)
  }

  def traverseTree(query: Long, startIndex: Int, currentDepth: Int, maxDepth: Int, fs: FSDataInputStream): Long = {
    println(s"traverseTree currentDepth:$currentDepth maxDepth:$maxDepth")
    if (currentDepth == maxDepth) {
      println("searching the last layer of tree")
      val result = searchBlock(query, currentDepth, maxDepth, startIndex, fs)
      result._2
    }
    else {
      val result = searchBlock(query, currentDepth, maxDepth, startIndex, fs)
      val newStart = getOffset(currentDepth, result._1*1024)
      traverseTree(query, newStart, currentDepth + 1, maxDepth, fs)
    }
  }



  def queryJackie(start: Long, fileName: String, hConf: Configuration): Long = {
    if (start < 0)
      fatal("Query cannot be negative")

    readFile(fileName, hConf) { s =>
      val fs = new FSDataInputStream(s)
      fs.seek(0)
      val depth = fs.readLong()
      println(s"depth==$depth")
      traverseTree(start, 0, 1, depth.toInt, fs)
    }
  }

  def query(start: Long, fileName: String, hConf: Configuration): Long = {
    //used to be queryStart
    //println(s"query: start=$start")
    readFile(fileName, hConf) { s =>
      val fs = new FSDataInputStream(s)
      fs.seek(0)
      val depth = fs.readLong()

      // keep track of depth and position in layer -- position = 1 + (i <- 1 to currentLayer).map(math.pow(1024,_)).sum
      var index: Int = 0
      var currentDepth = 1
      def getOffset(i: Int): Long = (0 to i).map(math.pow(1024, _).toLong).fold(0L)(_ + _) * 8
      var ret: Long = -1L
      //println(s"depth=$depth currentDepth=$currentDepth")
      while (currentDepth <= depth) {
        // if not on the last layer, find the largest value <= our start
        val read = fs.readLong()

        // check that read is not -1

        if (currentDepth < depth) {
          val offset = getOffset(currentDepth)
          if (read >= start || read == -1 || index > offset) {
            index = 1024 * math.max(index - 1, 0)
            try {
              fs.seek(offset + 8 * index)
            } catch {
              case e:EOFException => fatal(s"query=$start read=$read offset=$offset depth=$depth currentDepth=$currentDepth index=$index")
            }
            currentDepth += 1
          } else
            index += 1
        } else {
          if (read >= start) {
            ret = read
            currentDepth = Integer.MAX_VALUE
          }
          index += 1
        }
      }
      ret
    }
  }
}