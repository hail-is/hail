package org.broadinstitute.hail.io

import java.io.{OutputStream, InputStream}
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream}
import scala.collection.mutable

object IndexBTree {

  def write(arr: Array[Long], out: OutputStream) {
    val fs = new FSDataOutputStream(out)
    require(arr.length > 0)
    // first calculate the necessary number of layers in the tree -- log1024(arr.length) rounded up
    val depth = (math.log10(arr.length) / math.log10(1024)).ceil.toInt

    // now downsample the array -- each downsampling should use the smallest index contained -- 0-1023 should use 0, etc
    val layers = mutable.ArrayBuffer[IndexedSeq[Long]]()
    for (i <- 0 until depth - 1) {
      val multiplier = math.pow(1024, depth - 1 - i).toInt
      //      println(s"i = $i, mult = $multiplier")
      layers.append((0 until math.pow(1024, i + 1).toInt).map { j =>
        if (j * multiplier <= arr.length)
          arr(j * multiplier)
        else
          -1
      })
    }

    layers.append(arr)
    //    layers.zipWithIndex.foreach { case (a, i) => println(s"index $i size is ${a.size}")}
    //    println("After flatten: " + flat.size)
    val bytes = layers.flatten.flatMap(l => Array[Byte](
      (l >>> 56).toByte,
      (l >>> 48).toByte,
      (l >>> 40).toByte,
      (l >>> 32).toByte,
      (l >>> 24).toByte,
      (l >>> 16).toByte,
      (l >>> 8).toByte,
      (l >>> 0).toByte)).toArray
    fs.writeLong(depth)
    fs.write(bytes)
    fs.close()
  }

  def queryArr(start: Long, end: Long, arr: Array[Long]): Array[Long] = {
    println(arr.take(10).mkString(","))
    val depth = arr(0)

    // keep track of depth and position in layer -- position = 1 + (i <- 1 to currentLayer).map(math.pow(1024,_)).sum
    var layerOffset: Int = 0
    var layerPos: Int = 0
    var currentDepth = 1
    def getOffset(i: Int): Int = (0 to i).map(math.pow(1024, _).toInt).sum

    val ret = new mutable.ArrayBuffer[Long]()
    while (currentDepth <= depth) {
      // if not on the last layer, find the largest value <= our start
      if (currentDepth < depth) {
        if (arr(layerOffset + layerPos + 1) > start) {
          layerOffset = getOffset(currentDepth)
          currentDepth += 1
          layerPos = 0
        }
        else
          layerPos += 1
      }
      else {
        if (arr(layerOffset + layerPos + 1) > end)
          currentDepth = Integer.MAX_VALUE
        else if (arr(layerOffset + layerPos + 1) >= start) {
          layerPos += 1
          ret.+=(arr(layerOffset + layerPos))
        }
        else
          layerPos += 1
      }
    }
    ret.toArray
  }

  def query(start: Long, end: Long, in: InputStream): Array[Long] = {
    val fs = new FSDataInputStream(in)
    val depth = fs.readLong()

    // keep track of depth and position in layer -- position = 1 + (i <- 1 to currentLayer).map(math.pow(1024,_)).sum
    var layerPos: Long = 0
    var currentDepth = 1
    def getOffset(i: Int): Int = (0 to i).map(math.pow(1024, _).toInt).sum * 8
    val ret = new mutable.ArrayBuffer[Long]()

    while (currentDepth <= depth) {
      // if not on the last layer, find the largest value <= our start
      val read = fs.readLong()

      if (currentDepth < depth) {
        if (read > start) {
          fs.seek(getOffset(currentDepth) + 8192 * (layerPos))
          currentDepth += 1
          layerPos = 0
        }
        else {
          layerPos += 1
        }
      }
      else {
        if (read > end || read == -1)
          currentDepth = Integer.MAX_VALUE // exit the loop
        else if (read >= start) {
          ret += read
        }
      }
    }
    ret.toArray
  }

  def queryBlockIndices(start: Long, end: Long, in: InputStream): Array[BlockIndex] = {
    println(s"Got query: $start-$end")
    val fs = new FSDataInputStream(in)
    fs.seek(0)
    val depth = fs.readLong()
//    println(s"depth is $depth")

    // keep track of depth and position in layer -- position = 1 + (i <- 1 to currentLayer).map(math.pow(1024,_)).sum
    var index: Int = 0
    var currentDepth = 1
    def getOffset(i: Int): Int = (0 to i).map(math.pow(1024, _).toInt).sum * 8
    val ret = new mutable.ArrayBuffer[(Long, Int)]()

    while (currentDepth <= depth) {
      // if not on the last layer, find the largest value <= our start
      val read = fs.readLong()
//      println(s"read a value: $read")

      if (currentDepth < depth) {
//        println(s"read $read")
        if (read >= start) {
          index = 1024 * math.max(index-1, 0)
//          println(s"Incrementing depth, index=$index, read=$read -- going to ${getOffset(currentDepth) + 8 * index}")
          fs.seek(getOffset(currentDepth) + 8 * index)
          currentDepth += 1
        } else
          index += 1
      } else {
        if (read >= start)
          ret += ((read, index.toInt))
        if (read >= end)
          // exit the loop
          currentDepth = Integer.MAX_VALUE

        index += 1
      }
    }
    val indices = ret.take(ret.length-1)
      .zip(ret.takeRight(ret.length-1))
      .map {
        case ((pos1, ind1), (pos2, ind2)) =>
          BlockIndex(pos1, (pos2-pos1).toInt, ind1)
      }
      .toArray
//    println(s"number of blocks found: ${indices.length}")
//    println(s"first five indices: ${indices.map(_.toString).take(5).mkString("\n")}")
 //   println(s"last five indices: ${indices.map(_.toString).takeRight(5).mkString("\n")}")
//    indices.foreach(println)
    indices
  }

  def queryStart(start: Long, in: InputStream): Long = {
    val fs = new FSDataInputStream(in)
    fs.seek(0)
    val depth = fs.readLong()

    // keep track of depth and position in layer -- position = 1 + (i <- 1 to currentLayer).map(math.pow(1024,_)).sum
    var index: Int = 0
    var currentDepth = 1
    def getOffset(i: Int): Int = (0 to i).map(math.pow(1024, _).toInt).sum * 8
    var ret: Long = -1L

    while (currentDepth <= depth) {
      // if not on the last layer, find the largest value <= our start
      val read = fs.readLong()
      //      println(s"read a value: $read")

      if (currentDepth < depth) {
        //        println(s"read $read")
        if (read >= start || index > getOffset(currentDepth)) {
/*          if (start == 637534208) {
            println(s"index=$index, max = ${math.max(index-1, 0)}, mod = ${1024 * math.max(index-1, 0)}")
          }*/
          index = 1024 * math.max(index-1, 0)
/*          if (index == -1925316608) {
            println(s"depth=$currentDepth, depthOfTree=$depth, read=$read")
          }*/
//          println(s"Incrementing depth(d=$currentDepth -> $depth), index=$index, read=$read -- going to ${getOffset(currentDepth) + 8 * index}")
          fs.seek(getOffset(currentDepth) + 8 * index)
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