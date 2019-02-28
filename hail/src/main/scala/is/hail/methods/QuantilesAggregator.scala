package is.hail.methods

import is.hail.expr._
import is.hail.expr.types.virtual.Type
import is.hail.utils.ArrayBuilder

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.Sorting

class QuantilesAggregator(q: Double, bufSize: Int, t: Type, rand: Random) {

//  private val ord = t.ordering.toOrdering
//  private val comp = ord.lt _
  private val bufferPool: mutable.ArrayStack[Array[Long]] = mutable.ArrayStack()

  private val fullBuffers: ArrayBuffer[Array[Long]] = ArrayBuffer(null)
  private val bufferSizes: ArrayBuffer[Int] = ArrayBuffer()
  private val bufSize2 = bufSize * 2
  private var inputBuffer: Array[Long] = Array.ofDim[Long](bufSize2 + 1)
  private var inBufSize: Int = 0

  private var count: Long = 0

  def seqOp(x: Long): Unit = {
    inputBuffer(inBufSize) = x
    inBufSize += 1
    count += 1
    if (inBufSize == bufSize2)
      processFullInputBuffer()
  }

  def processFullInputBuffer(): Unit = {
    java.util.Arrays.sort(inputBuffer)
    val compacted = getBuffer()
    QuantilesAggregator.compactBuffer(inputBuffer, 0, bufSize2, compacted, 0, rand.nextBoolean())
    if (fullBuffers(0) == null)
      fullBuffers(0) = compacted
    else
      compact(0, compacted)
    inBufSize = 0
  }

  def compact(height: Int, other: Array[Long]): Unit = {
    var curHeight = height
    var right = other
    do {
      if (fullBuffers.size == curHeight + 1) {
        fullBuffers += null
      }
      val left = fullBuffers(curHeight)
      fullBuffers(curHeight) = null
      val out = getBuffer()
      QuantilesAggregator.mergeAndCompactBuffer(left, 0, bufSize, right, 0, bufSize, out, 0, rand.nextBoolean())
      bufferPool.push(left)
      bufferPool.push(right)
      curHeight += 1
      right = out
    } while (fullBuffers(curHeight) != null)
    fullBuffers(curHeight) = right
  }

  def getBuffer(): Array[Long] = {
    if (bufferPool.isEmpty)
      Array.ofDim[Long](bufSize + 1)
    else
      bufferPool.pop()
  }

  def initBufferSizes(): Unit = {
    if (bufferSizes.isEmpty) {
      var i = 0
      while (i < fullBuffers.size) {
        if (fullBuffers(i) == null) {
          bufferSizes += 0
          fullBuffers(i) = getBuffer()
        } else
          bufferSizes += bufSize
        i += 1
      }
    }
    assert(bufferSizes.size == fullBuffers.size)
  }

  def combOp(agg2: this.type): Unit = {
    initBufferSizes()
    agg2.initBufferSizes()
    java.util.Arrays.sort(inputBuffer, 0, inBufSize)
    java.util.Arrays.sort(agg2.inputBuffer, 0, agg2.inBufSize)
    var carry: Array[Long] = Array.ofDim[Long](bufSize2 + 1)
    var scratch = Array.ofDim[Long](bufSize2 + 1)
    var scratch2 = Array.ofDim[Long](bufSize2 + 1)
    var carrySize: Int = 0
    val totalInBufSize = inBufSize + agg2.inBufSize
    if (totalInBufSize < bufSize2) {
      QuantilesAggregator.merge(inputBuffer, 0, inBufSize, agg2.inputBuffer, 0, agg2.inBufSize, scratch, 0)
      val tmp = inputBuffer
      inputBuffer = scratch
      scratch = tmp
      inBufSize = totalInBufSize
    } else {
      val lStart = totalInBufSize  % 2
      carrySize = totalInBufSize / 2
      QuantilesAggregator.mergeAndCompactBuffer(inputBuffer, lStart, inBufSize, agg2.inputBuffer, 0, agg2.inBufSize, carry, 0, rand.nextBoolean())
      inBufSize = lStart
    }
    var i = 0
    while (i < agg2.fullBuffers.size || carrySize > 0) {
      if (i >= fullBuffers.size) {
        fullBuffers += getBuffer()
        bufferSizes += 0
      }
      val rightSize = if (i < agg2.fullBuffers.size) agg2.bufferSizes(i) else 0
      val rightBuffer = if (i < agg2.fullBuffers.size) agg2.fullBuffers(i) else null
      val totalSize = bufferSizes(i) + rightSize + carrySize
      if (totalSize <= bufSize) {
        var tmp = getBuffer()
        if (carrySize > 0 && bufferSizes(i) > 0 && rightSize > 0) {
          QuantilesAggregator.merge(fullBuffers(i), 0, bufferSizes(i), rightBuffer, 0, rightSize, scratch, 0)
          val scratchSize = bufferSizes(i) + rightSize
          QuantilesAggregator.merge(scratch, 0, scratchSize, carry, 0, carrySize, tmp, 0)
        } else if (bufferSizes(i) > 0 && rightSize > 0) {
          QuantilesAggregator.merge(fullBuffers(i), 0, bufferSizes(i), rightBuffer, 0, rightSize, tmp, 0)
        } else if (carrySize > 0 && rightSize > 0) {
          QuantilesAggregator.merge(carry, 0, carrySize, rightBuffer, 0, rightSize, tmp, 0)
        } else if (bufferSizes(i) > 0 && carrySize > 0) {
          QuantilesAggregator.merge(fullBuffers(i), 0, bufferSizes(i), carry, 0, carrySize, tmp, 0)
        } else if (bufferSizes(i) > 0) {
          val x = tmp
          tmp = fullBuffers(i)
          fullBuffers(i) = x
        } else if (rightSize > 0) {
          tmp = rightBuffer
        } else if (carrySize > 0) {
          System.arraycopy(carry, 0, tmp, 0, carrySize)
        }
        bufferPool.push(fullBuffers(i))
        fullBuffers(i) = tmp
        bufferSizes(i) = totalSize
        carrySize = 0
      } else {
        val lStart = totalSize % 2
        if (carrySize > 0 && bufferSizes(i) > 0 && rightSize > 0) {
          val scratchSize = bufferSizes(i) - lStart + rightSize
          QuantilesAggregator.merge(fullBuffers(i), lStart, bufferSizes(i), rightBuffer, 0, rightSize, scratch2, 0)
          QuantilesAggregator.mergeAndCompactBuffer(carry, 0, carrySize, scratch2, 0, scratchSize, scratch, 0, rand.nextBoolean())
          val tmp = carry
          carry = scratch
          scratch = tmp
        } else if (bufferSizes(i) > 0 && rightSize > 0) {
          QuantilesAggregator.mergeAndCompactBuffer(fullBuffers(i), lStart, bufferSizes(i), rightBuffer, 0, rightSize, carry, 0, rand.nextBoolean())
        } else if (carrySize > 0 && rightSize > 0) {
          fullBuffers(i)(0) = carry(0)
          QuantilesAggregator.mergeAndCompactBuffer(carry, lStart, carrySize, rightBuffer, 0, rightSize, scratch, 0, rand.nextBoolean())
          val tmp = carry
          carry = scratch
          scratch = tmp
        } else if (bufferSizes(i) > 0 && carrySize > 0) {
          QuantilesAggregator.mergeAndCompactBuffer(fullBuffers(i), lStart, bufferSizes(i), carry, 0, carrySize, scratch, 0, rand.nextBoolean())
          val tmp = carry
          carry = scratch
          scratch = tmp
        } else if (carrySize > 0) {
          fullBuffers(i)(0) = carry(0)
          QuantilesAggregator.compactBuffer(carry, lStart, carrySize, scratch, 0, rand.nextBoolean())
          val tmp = carry
          carry = scratch
          scratch = tmp
        } else {
          assert(false)
        }
        bufferSizes(i) = lStart
        carrySize = totalSize / 2
      }
      i += 1
    }
  }

  def compactAfterComb(height: Int, other: Array[Long], otherSize: Int): Unit = {
    var curHeight = height
    var right = other
    var rightSize = otherSize
    do {
      if (fullBuffers.size == curHeight + 1) {
        fullBuffers += getBuffer()
        bufferSizes += 0
      }
      val left = fullBuffers(curHeight)
      fullBuffers(curHeight) = null
      val out = getBuffer()
      QuantilesAggregator.mergeAndCompactBuffer(left, 0, bufSize, right, 0, bufSize, out, 0, rand.nextBoolean())
      bufferPool.push(left)
      bufferPool.push(right)
      curHeight += 1
      right = out
    } while (fullBuffers(curHeight) != null)
    fullBuffers(curHeight) = right
  }

  def cdf: Array[(Long, Long)] = {
    initBufferSizes()
    val builder: ArrayBuilder[(Long, Long)] = new ArrayBuilder(0)
    var height: Int = 0
    var n: Long = 0
    while (height < fullBuffers.size) {
      val buf = fullBuffers(height)
      var i: Int = 0
      while (i < bufferSizes(height)) {
        val weight: Long = 1 << (height + 1)
        builder += (weight, buf(i))
        i += 1
        n += weight
      }
      height += 1
    }
    var i: Int = 0
    while (i < inBufSize) {
      builder += (1, inputBuffer(i))
      i += 1
      n += 1
    }
//    builder.result().sortWith((p1, p2) => comp(p1._2, p2._2))
    builder.result().sortWith(_._2 < _._2)
  }

  def result = {
    val summary = cdf
    var i = 0
    var rank: Long = 0
    val desiredRank: Long = (q * count).toLong
    while (rank <= desiredRank) {
      rank += summary(i)._1
      i += 1
    }
    summary(i-1)._2
  }

  def copy(): QuantilesAggregator = new QuantilesAggregator(q, bufSize, t, rand)
}

object QuantilesAggregator {
  def mergeAndCompactBuffer(
    left: Array[Long], lStart: Int, lEnd: Int,
    right: Array[Long], rStart: Int, rEnd: Int,
    out: Array[Long], outStart: Int, skipFirst: Boolean
  ): Unit = {
    var i: Int = lStart
    var j: Int = rStart
    var o: Int = outStart

    left(lEnd) = Long.MaxValue
    right(rEnd) = Long.MaxValue
    val size: Int = (lEnd - lStart + rEnd - rStart) / 2
    if (skipFirst) {
      if (left(i) < right(j)) i += 1 else j += 1
    }
    while (o < size) {
      if (left(i) < right(j)) {
        out(o) = left(i)
        i += 1
      } else {
        out(o) = right(j)
        j += 1
      }
      o += 1
      if (left(i) < right(j)) i += 1 else j += 1
    }
  }

  def merge(
    left: Array[Long], lStart: Int, lEnd: Int,
    right: Array[Long], rStart: Int, rEnd: Int,
    out: Array[Long], outStart: Int
  ): Unit = {
    var i: Int = 0
    var j: Int = 0
    var o: Int = 0
    left(lEnd) = Long.MaxValue
    right(rEnd) = Long.MaxValue
    val n = lEnd - lStart + rEnd - rStart
    while (o < n) {
      if (left(i) < right(j)) {
        out(o) = left(i)
        i += 1
      } else {
        out(o) = right(j)
        j += 1
      }
      o += 1
    }
  }

  def compactBuffer(buf: Array[Long], inStart: Int, inEnd: Int, out: Array[Long], outStart: Int, skipFirst: Boolean): Unit = {
    var i = inStart
    var o = outStart
    if (skipFirst) {
      i += 1
    }
    while (i < inEnd) {
      out(o) = buf(i)
      i += 2
      o += 1
    }
  }
}

object Main {
  def time(block: => Unit): Long = {
    val t0 = System.nanoTime()
    block
    val t1 = System.nanoTime()
    t1 - t0
  }

  def main(args: Array[String]): Unit = {
    import is.hail.expr.types.virtual.TInt64
    import scala.math.{abs, pow}
    val seed = Random.nextInt()
    val rand = new Random(seed)
    val agg = new QuantilesAggregator(0.5, 100, TInt64(), rand)
//    val agg2 = new QuantilesAggregator(0.5, 100, TInt64(), rand)
    val n: Int = 150000

    val data = rand.shuffle(IndexedSeq.range(0, n)).toArray
//    val data = Array.range(0, n)
    var cdf: Array[(Long, Long)] = null
    val time1 = time {
      var i = 0
//      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
      while (i < n) { agg.seqOp(data(i)); i += 1 }
//      agg.combOp(agg2.asInstanceOf[agg.type])
      cdf = agg.cdf
    }
//    val time3 = time {
//      data.sorted
//    }
    val time2 = time {
      val data2 = Array.ofDim[Long](n)
      var i = 0
      while (i < n) {
        data2(i) = data(i)
        i += 1
      }
      Sorting.quickSort(data2)
    }
    val time4 = time {
      val data2 = Array.ofDim[Long](n)
      var i = 0
      while (i < n) {
        data2(i) = data(i)
        i += 1
      }
      java.util.Arrays.sort(data2)
    }
    val factor: Double = pow(10, 9)
    println(s"approx took ${ time1.toDouble / factor } s")
    println(s"quicksort took ${ time2.toDouble / factor } s")
//    println(s"stable sort took ${ time3.toDouble / factor } s")
    println(s"java sort took ${ time4.toDouble / factor } s")

//    for ((weight, value) <- cdf) println(s"$weight : $value")

    val epsilon: Double = 0.05
    var i: Int = 0
    var q: Double = epsilon
    var curRank: Long = 0
    var totalError: Long = 0
    var count: Long = 0
    var maxError: Long = 0
    while (q < 1) {
      val rank: Long = {
        val r = (q * n).floor.toLong
        if (r == n) r - 1 else r
      }
      while (curRank <= rank) {
        curRank += cdf(i)._1
        i += 1
      }
      val error = abs(cdf(i-1)._2.asInstanceOf[Long] - rank)
      totalError += error
      count += 1
      if (error > maxError) maxError = error

      q += epsilon
    }
    while (i < cdf.size) {
      curRank += cdf(i)._1
      i += 1
    }
    assert(curRank == n)
    println(s"average error = ${ (totalError.toDouble / n) / count }")
    println(s"max error = ${ maxError.toDouble / n }")
  }
}
