package is.hail.methods

import is.hail.expr._
import is.hail.expr.types.virtual.Type
import is.hail.utils.ArrayBuilder

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class QuantilesAggregator(q: Double, bufSize: Int, t: Type, rand: Random) extends TypedAggregator[Any] {

  private val comp = t.ordering.ltNonnull _
  private val bufferPool: mutable.ArrayStack[Array[Any]] = mutable.ArrayStack()

  private val fullBuffers: ArrayBuffer[Array[Any]] = ArrayBuffer(null)
  private var inputBuffer: Array[Any] = Array.ofDim[Any](bufSize)
  private var inBufSize: Int = 0

  private var count: Long = 0

  def seqOp(x: Any): Unit = {
    inputBuffer(inBufSize) = x
    inBufSize += 1
    count += 1
    if (inBufSize == bufSize) {
      inputBuffer = inputBuffer.sortWith(comp)
      if (fullBuffers(0) == null)
        fullBuffers(0) = inputBuffer
      else
        compact(0, inputBuffer)
      inputBuffer = getBuffer()
      inBufSize = 0
    }
  }

  def compact(height: Int, other: Array[Any]): Unit = {
    var curHeight = height
    var right = other
    do {
      if (fullBuffers.size == curHeight + 1) {
        fullBuffers += null
      }
      val left = fullBuffers(curHeight)
      fullBuffers(curHeight) = null
      var i: Int = 0
      var j: Int = 0
      val out = getBuffer()
      var o: Int = 0

      if (rand.nextBoolean()) {
        if (comp(left(i), right(j))) i += 1 else j += 1
      }
      while (i != bufSize && j != bufSize) {
        if (comp(left(i), right(j))) {
          out(o) = left(i)
          i += 1
          if (i != bufSize && comp(left(i), right(j))) {
            i += 1
          } else {
            j += 1
          }
        } else {
          out(o) = right(j)
          j += 1
          if (j != bufSize && comp(right(i), left(j))) {
            j += 1
          } else {
            i += 1
          }
        }
        o += 1
      }
      while (i < bufSize) {
        out(o) = left(i)
        i += 2
        o += 1
      }
      while (j < bufSize) {
        out(o) = right(j)
        j += 2
        o += 1
      }
      bufferPool.push(left)
      bufferPool.push(right)
      curHeight += 1
      right = out
    } while (fullBuffers(curHeight) != null)
    fullBuffers(curHeight) = right
  }

  def getBuffer(): Array[Any] = {
    if (bufferPool.isEmpty)
      Array.ofDim[Any](bufSize)
    else
      bufferPool.pop()
  }

  def combOp(agg2: this.type): Unit = {
    ???
  }

  def cdf: Array[(Long, Any)] = {
    val builder: ArrayBuilder[(Long, Any)] = new ArrayBuilder(0)
    var height: Int = 0
    var n: Long = 0
    while (height < fullBuffers.size) {
      if (fullBuffers(height) != null) {
        val buf = fullBuffers(height)
        var i: Int = 0
        while (i < bufSize) {
          val weight: Long = 1 << height
          builder += (weight, buf(i))
          i += 1
          n += weight
        }
      }
      height += 1
    }
    var i: Int = 0
    while (i < inBufSize) {
      builder += (1, inputBuffer(i))
      i += 1
      n += 1
    }
    builder.result().sortWith((p1, p2) => comp(p1._2, p2._2))
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
    val agg = new QuantilesAggregator(0.5, 512, TInt64(), rand)
    val n: Long = 1000000

    val data = rand.shuffle(IndexedSeq.range(0, n))
    val time1 = time {
      for (i <- data) agg.seqOp(i)
    }
    val ord = TInt64().ordering.ltNonnull _
    val time2 = time {
      data.sortWith(ord)
    }
    val factor: Double = pow(10, 9)
    println(s"approx took ${ time1.toDouble / factor } s")
    println(s"sort took ${ time2.toDouble / factor } s")

    val cdf = agg.cdf

    val epsilon: Double = 0.01
    var i: Int = 0
    var q: Double = epsilon
    var curRank: Long = 0
    var totalError: Long = 0
    var count: Long = 0
    var maxError: Long = 0
    while (q < 1) {
      val rank: Long = (q * n).round
      while (curRank < rank && i < cdf.size) {
        curRank += cdf(i)._1
        i += 1
      }
      val l = abs(curRank - cdf(i-1)._1 - rank)
      val r = abs(curRank - rank)
      val error: Long = if (l < r || i == cdf.size)
        abs(cdf(i-1)._2.asInstanceOf[Long] - rank)
      else
        abs(cdf(i)._2.asInstanceOf[Long] - rank)
      totalError += error
      count += 1
      if (error > maxError) maxError = error

      q += epsilon
    }
    println(s"average error = ${ (totalError.toDouble / n) / count }")
    println(s"max error = ${ maxError.toDouble / n }")
  }
}
