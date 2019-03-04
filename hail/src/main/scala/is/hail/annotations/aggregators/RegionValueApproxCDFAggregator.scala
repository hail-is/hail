package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types.physical._
import is.hail.utils.ArrayBuilder
import org.apache.spark.sql.Row

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import java.util.Arrays.sort

import net.sourceforge.jdistlib.rng.MersenneTwister

object Test {
  def main(args: Array[String]): Unit = {
    val k = 10000
    val agg = new RegionValueApproxCDFLongAggregator2(k, 8, false)
    for (i <- Range(0, 29))
      println(s"depth $i: ${agg.depthCapacity(i)} vs ${k * math.pow(2.0/3.0, i)}")
  }
}

class RegionValueApproxCDFLongAggregator2(val k: Int, val m: Int, eager: Boolean) { // extends RegionValueAggregator {
  private val growthRate: Int = 4

  // 0 <= power <= 30
  private val powersOfThree = Array[Long](1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683,
    59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721, 129140163,
    387420489, 1162261467, 3486784401L, 10460353203L, 31381059609L,
    94143178827L, 282429536481L, 847288609443L, 2541865828329L, 7625597484987L,
    22876792454961L, 68630377364883L, 205891132094649L)

  var n: Long = 0
  var numLevelsCapacity = findInitialNumLevels()
  var numLevels = 1
  var items: Array[Long] = Array.ofDim[Long](computeTotalCapacity(numLevelsCapacity))
  var levels: Array[Int] = Array.fill[Int](numLevelsCapacity + 1)(items.length)
  var base = if (eager) items.length - levelCapacity(0) else 0
  private val rand = new MersenneTwister()

  def size: Int = levels(numLevels) - levels(0)

  def memUsage: Int = items.length

  def seqOp(region: Region, x: Long, missing: Boolean) {
    if (!missing) _seqOp(x)
  }

  def _seqOp(x: Long) {
    if (levels(0) == base) {
      if (eager)
        compactEager()
      else
        compact()
    }
    n += 1
    val newBot = levels(0) - 1
    items(newBot) = x
    levels(0) = newBot
  }

  def combOp(other: RegionValueApproxCDFLongAggregator2) {
    assert (m == other.m)
    val finalN = n + other.n
    var i = other.levels(0)
    while (i < other.levels(1)) {
      _seqOp(other.items(i))
      i += 1
    }
    if (other.numLevels >= 2) mergeHigherLevels(other, finalN)
    n = finalN
    assertCorrectTotalWeight()
  }

  def clear() {
    n = 0
    numLevels = 1
    var i = 0
    while (i < levels.length) {
      levels(i) = items.length
      i += 1
    }
    base = if (eager) items.length - levelCapacity(0) else 0
  }

  def compact() {
    assert(levels(0) == 0)
    val level = findFullLevel()
    if (level == numLevels - 1) growFullSketch()

    val start = levels(level)
    val size = levels(level + 1) - levels(level)
    val adj = size % 2
    val halfSize = size / 2

    levels(level) += adj
    val end = compactLevel(items, levels, level)

    // If size was odd, need to copy uncompacted element up. If not, this is
    // still safe.
    items(end - 1) = items(start)
    levels(level) = levels(level + 1) - adj

    if (level > 0) {
      System.arraycopy(items, 0, items, halfSize, start)
      var lvl = 0
      while (lvl < level) {
        levels(lvl) += halfSize
        lvl += 1
      }
    }

    assertCorrectTotalWeight()
  }

  def compactEager() {
    assert(levels(1) - levels(0) >= levelCapacity(0))

    def compactAndGrow(level: Int, singletons: Int) {
      val curTotalCap = levels(numLevels)
      assert(items.length == curTotalCap)
      assert(level == numLevelsCapacity - 1)
      val size: Int = levels(level + 1) - levels(level)
      assert(size % 2 == 0)

      numLevels += 1
      val deltaCap = m * growthRate
      val newTotalCap = curTotalCap + deltaCap
      val newBuf = Array.ofDim[Long](newTotalCap)

      numLevelsCapacity += growthRate
      levels = levels.padTo(numLevelsCapacity + growthRate + 1, newTotalCap)

      val halfSize = size / 2
      QuantilesAggregator.compactBuffer(items, levels(level), levels(level + 1), newBuf, newTotalCap - halfSize, rand.nextBoolean())

      levels(level + 1) = newTotalCap - halfSize

      System.arraycopy(items, 0, newBuf, 0, singletons)
      items = newBuf
    }

    var level = 0
    var singletons = 0
    while (levels(level + 1) - levels(level) >= levelCapacity(level)) {
      val size = levels(level + 1) - levels(level)
      val adj = size % 2

      items(singletons) = items(levels(level))

      levels(level) += adj

      if (level >= numLevels - 1) {
        if (level == numLevelsCapacity - 1) {
          compactAndGrow(level, singletons)
        } else {
          numLevels += 1
          compactLevel(items, levels, level)
        }
      } else {
        compactLevel(items, levels, level)
      }

      levels(level) = singletons
      singletons += adj
      level += 1
    }
    val shift = levels(level) - singletons
    var i = 0
    while (i < level) {
      levels(i) += shift
      i += 1
    }
    System.arraycopy(items, 0, items, shift, singletons)
    base = levels(1) - levelCapacity(0)
  }

  def compactLevel(buf: Array[Long], levels: Array[Int], level: Int): Int = {
    val a = levels(level)
    val c = levels(level + 1)
    val d = levels(level + 2)
    val size = c - a
    val sizeAbove = d - c

    assert(size > 0 && (size & 1) == 0)

    val b = a + ((c - a) / 2)

    if (level == 0) sort(items, a, c)
    if (sizeAbove == 0) {
      QuantilesAggregator.compactBufferBackwards(items, a, c, items, c, rand.nextBoolean())
    } else {
      QuantilesAggregator.compactBuffer(items, a, c, items, a, rand.nextBoolean())
      QuantilesAggregator.mergeNoSentinals(items, a, b, items, c, d, items, b)
    }
    levels(level + 1) = b
    b
  }

  def findFullLevel(): Int = {
    var level: Int = 0
    while (levels(level + 1) - levels(level) < levelCapacity(level)) {
      level += 1
    }
    level
  }

  def levelCapacity(level: Int, numLevels: Int = numLevels): Int =
    math.max(m, depthCapacity(numLevels - level - 1))

  // compute k * (2/3)^depth rounded to the nearest integer
  def depthCapacity(depth: Int): Int = {
    assert(depth <= 30)
    val tmp = ((k.toLong << (depth + 1)) / powersOfThree(depth)).toInt
    val result = (tmp + 1) >> 1
    result
  }

  private def growFullSketch() {
    val curTotalCap = levels(numLevels)
    assert(levels(0) == 0)
    assert(items.length == curTotalCap)

    numLevels += 1
    if (numLevels > numLevelsCapacity) {
      numLevelsCapacity += growthRate
      levels = levels.padTo(numLevelsCapacity + growthRate + 1, curTotalCap)

      val deltaCap = m * growthRate
      val newTotalCap = curTotalCap + deltaCap
      val newBuf = Array.ofDim[Long](newTotalCap)

      System.arraycopy(items, 0, newBuf, deltaCap, curTotalCap)
      items = newBuf

      var i = 0
      while (i < levels.length) {
        levels(i) += deltaCap
        i += 1
      }
      assert(levels(numLevels) == newTotalCap)
    }
  }

  def mergeHigherLevels(other: RegionValueApproxCDFLongAggregator2, finalN: Long) {
    val workbuf = Array.ofDim[Long](size + other.sizeAboveLevel0)
    val ub = QuantilesAggregator.ubOnNumLevels(finalN)
    val worklevels = new Array[Int](ub + 2)
    // ub+1 does not work
    val outlevels = new Array[Int](ub + 2)

    val provisionalNumLevels = math.max(numLevels, other.numLevels)
    populateWorkArrays(other, workbuf, worklevels, provisionalNumLevels)

    // notice that workbuf is being used as both the input and output here
    val (finalNumLevels, finalCapacity, finalPop) = generalCompress(k, m, provisionalNumLevels, workbuf, worklevels, workbuf, outlevels)
    assert(finalNumLevels <= ub) // can sometimes be much bigger

    // now we need to transfer the results back into the "self" sketch
    val newbuf = if (finalCapacity == items.length) items else Array.ofDim[Long](finalCapacity)
    val freeSpaceAtBottom = finalCapacity - finalPop
    System.arraycopy(workbuf, outlevels(0), newbuf, freeSpaceAtBottom, finalPop)
    val offset = freeSpaceAtBottom - outlevels(0)
    if (levels.length < (finalNumLevels + 1)) levels = Array.ofDim[Int](finalNumLevels + 1)
    var lvl = 0
    while (lvl < (finalNumLevels + 1)) {
      levels(lvl) = outlevels(lvl) + offset
      lvl += 1
    }
    items = newbuf
    numLevels = finalNumLevels
  }

  private def populateWorkArrays(
    other: RegionValueApproxCDFLongAggregator2,
    workbuf: Array[Long],
    worklevels: Array[Int],
    provisionalNumLevels: Int
  ) {
    worklevels(0) = 0

    // Note: the level zero data from "other" was already inserted into "self"
    val selfPopZero = levelSize(0)
    System.arraycopy(items, levels(0), workbuf, worklevels(0), selfPopZero)
    worklevels(1) = worklevels(0) + selfPopZero
    var lvl = 1
    while (lvl < provisionalNumLevels) {
      val selfPop = levelSize(lvl)
      val otherPop = other.levelSize(lvl)
      worklevels(lvl + 1) = worklevels(lvl) + selfPop + otherPop

      if (selfPop > 0 && otherPop > 0)
        QuantilesAggregator.mergeNoSentinals(
          items, levels(lvl), levels(lvl + 1),
          other.items, other.levels(lvl), other.levels(lvl + 1),
          workbuf, worklevels(lvl))
      else if (selfPop > 0)
        System.arraycopy(items, levels(lvl), workbuf, worklevels(lvl), selfPop)
      else if (otherPop > 0)
        System.arraycopy(other.items, other.levels(lvl), workbuf, worklevels(lvl), otherPop)

      lvl += 1
    }
  }

  def generalCompress(
    k: Int, m: Int, numLevelsIn: Int,
    inBuf: Array[Long], inLevels: Array[Int],
    outBuf: Array[Long], outLevels: Array[Int]
  ): (Int, Int, Int) = {
    assert(numLevelsIn > 0) // things are too weird if zero levels are allowed

    var numLevels = numLevelsIn
    var currentItemCount = inLevels(numLevels) - inLevels(0) // decreases with each compaction
    var targetItemCount = computeTotalCapacity(numLevels) // increases if we add levels
    outLevels(0) = 0

    var curLevel = 0
    while (curLevel < numLevels) {
      // If we are at the current top level, add an empty level above it for convenience,
      // but do not increment numLevels until later
      if (curLevel == (numLevels - 1)) inLevels(curLevel + 2) = inLevels(curLevel + 1)
      val start = inLevels(curLevel)
      val size = inLevels(curLevel + 1) - start

      if ((currentItemCount < targetItemCount) || (size < levelCapacity(curLevel, numLevels))) { // copy level over as is
        // because inBuf and outBuf could be the same, make sure we are not moving data upwards!
        assert(start >= outLevels(curLevel))
        System.arraycopy(inBuf, start, outBuf, outLevels(curLevel), size)
        outLevels(curLevel + 1) = outLevels(curLevel) + size
      } else { // The sketch is too full AND this level is too full, so we compact it
        // Note: this can add a level and thus change the sketches capacities
        val adj = size % 2
        val halfSize = size / 2

        outBuf(outLevels(curLevel)) = inBuf(start)
        outLevels(curLevel + 1) = outLevels(curLevel) + adj

        inLevels(curLevel) += adj
        compactLevel(inBuf, inLevels, curLevel)

        currentItemCount -= halfSize
        if (curLevel == (numLevels - 1)) {
          numLevels += 1
          targetItemCount += levelCapacity(0, numLevels)
        }
      }
      curLevel += 1
    }
    assert((outLevels(numLevels) - outLevels(0)) == currentItemCount)
    (numLevels, targetItemCount, currentItemCount)
  }

  def cdf: (Array[Long], Array[Long]) = {
    val builder: ArrayBuilder[(Long, Long)] = new ArrayBuilder(0)
    var level = 0
    while (level < numLevels) {
      val weight: Long = 1 << (level)
      var i = levels(level)
      while (i < levels(level + 1)) {
        builder += (weight, items(i))
        i += 1
      }
      level += 1
    }
    val sorted = builder.result().sortBy(_._2)
    val values = Array.ofDim[Long](sorted.length)
    val ranks = Array.ofDim[Long](sorted.length + 1)
    var i = 0
    while (i < sorted.length) {
      values(i) = sorted(i)._2
      i += 1
    }
    i = 0
    var rank: Long = 0
    ranks(0) = 0
    while (i < sorted.length) {
      rank += sorted(i)._1
      i += 1
      ranks(i) = rank
    }
    (values, ranks)
  }

  private def assertCorrectTotalWeight() {
    var total = 0
    var weight = 1
    var lvl = 0
    while (lvl < numLevels) {
      total += weight * (levels(lvl + 1) - levels(lvl))
      weight *= 2
      lvl += 1
    }
    assert(total == n)
  }

  def computeTotalCapacity(numLevels: Int): Int = {
    var total = 0
    var h = 0
    while (h < numLevels) {
      total += levelCapacity(h, numLevels)
      h += 1
    }
    total
  }

  private def levelSize(level: Int) =
    if (level >= numLevels) 0 else levels(level + 1) - levels(level)

  def sizeAboveLevel0 = levels(numLevels) - levels(1)

  def findInitialNumLevels(): Int = {
    var numLevels = 0
    while (depthCapacity(numLevels) >= m) numLevels += 1
    numLevels + 1
  }
}

class RegionValueApproxCDFLongAggregator(bufSize: Int) extends RegionValueAggregator {

  val resultType: PType = PStruct("values" -> PArray(PInt64()), "ranks" -> PArray(PInt64()), "count" -> PInt64())
  private val bufferPool: mutable.ArrayStack[Array[Long]] = mutable.ArrayStack()

  private val fullBuffers: ArrayBuffer[Array[Long]] = ArrayBuffer(null)
  private val bufferSizes: ArrayBuffer[Int] = ArrayBuffer()
  private val bufSize2 = bufSize * 2
  private var inputBuffer: Array[Long] = Array.ofDim[Long](bufSize2 + 1)
  private var inBufSize: Int = 0
  private var hasCombined: Boolean = false

  private val rand = new MersenneTwister()

  def memUsage: Int = fullBuffers.size * (bufSize + 1) + bufSize2 + 1

  def seqOp(region: Region, x: Long, missing: Boolean) {
    if (!missing) _seqOp(x)
  }

  def _seqOp(x: Long){
    inputBuffer(inBufSize) = x
    inBufSize += 1
    if (inBufSize == bufSize2)
      processFullInputBuffer()
  }

  def processFullInputBuffer() {
    java.util.Arrays.sort(inputBuffer)
    val compacted = getBuffer()
    QuantilesAggregator.compactBuffer(inputBuffer, 0, bufSize2, compacted, 0, rand.nextBoolean())
    if (fullBuffers(0) == null)
      fullBuffers(0) = compacted
    else
      compact(0, compacted)
    inBufSize = 0
  }

  def compact(height: Int, other: Array[Long]) {
    assert(!hasCombined)
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
    if (bufferPool.isEmpty) {
      Array.ofDim[Long](bufSize + 1)
    } else
      bufferPool.pop()
  }

  def initBufferSizes() {
    if (bufferSizes.isEmpty) {
      hasCombined = true
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
  }

  def combOp(_that: RegionValueAggregator) {
    val that = _that.asInstanceOf[RegionValueApproxCDFLongAggregator]
    initBufferSizes()
    that.initBufferSizes()
    java.util.Arrays.sort(inputBuffer, 0, inBufSize)
    java.util.Arrays.sort(that.inputBuffer, 0, that.inBufSize)
    var carry: Array[Long] = Array.ofDim[Long](bufSize2 + 1)
    var scratch = Array.ofDim[Long](bufSize2 + 1)
    var scratch2 = Array.ofDim[Long](bufSize2 + 1)
    var carrySize: Int = 0
    val totalInBufSize = inBufSize + that.inBufSize
    if (totalInBufSize < bufSize2) {
      QuantilesAggregator.merge(inputBuffer, 0, inBufSize, that.inputBuffer, 0, that.inBufSize, scratch, 0)
      val tmp = inputBuffer
      inputBuffer = scratch
      scratch = tmp
      inBufSize = totalInBufSize
    } else {
      val lStart = totalInBufSize  % 2
      carrySize = totalInBufSize / 2
      QuantilesAggregator.mergeAndCompactBuffer(inputBuffer, lStart, inBufSize, that.inputBuffer, 0, that.inBufSize, carry, 0, rand.nextBoolean())
      inBufSize = lStart
    }
    var i = 0
    while (i < that.fullBuffers.size || carrySize > 0) {
      if (i >= fullBuffers.size) {
        fullBuffers += getBuffer()
        bufferSizes += 0
      }
      val rightSize = if (i < that.fullBuffers.size) that.bufferSizes(i) else 0
      val rightBuffer = if (i < that.fullBuffers.size) that.fullBuffers(i) else null
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

  def result(rvb: RegionValueBuilder): Unit = {
    val res = Row.fromTuple(cdf)
    rvb.addAnnotation(resultType.virtualType, res)
  }

  def cdf: (Array[Long], Array[Long]) = {
    initBufferSizes()
    val builder: ArrayBuilder[(Long, Long)] = new ArrayBuilder(0)
    var height: Int = 0
    while (height < fullBuffers.size) {
      val buf = fullBuffers(height)
      val weight: Long = 1 << (height + 1)
      var i: Int = 0
      while (i < bufferSizes(height)) {
        builder += (weight, buf(i))
        i += 1
      }
      height += 1
    }
    var i: Int = 0
    while (i < inBufSize) {
      builder += (1, inputBuffer(i))
      i += 1
    }
    val sorted = builder.result().sortBy(_._2)
    val values = Array.ofDim[Long](sorted.length)
    val ranks = Array.ofDim[Long](sorted.length + 1)
    i = 0
    while (i < sorted.length) {
      values(i) = sorted(i)._2
      i += 1
    }
    i = 0
    var rank: Long = 0
    ranks(0) = 0
    while (i < sorted.length) {
      rank += sorted(i)._1
      i += 1
      ranks(i) = rank
    }
    (values, ranks)
  }

  def copy(): RegionValueApproxCDFLongAggregator = {
    val rva = new RegionValueApproxCDFLongAggregator(bufSize)
    System.arraycopy(inputBuffer, 0, rva.inputBuffer, 0, inBufSize)
    rva.inBufSize = inBufSize
    if (bufferSizes.isEmpty) {
      var i = 0
      while (i < fullBuffers.size) {
        if (fullBuffers(i) == null)
          rva.fullBuffers += null
        else {
          rva.fullBuffers += Array.ofDim[Long](bufSize + 1)
          System.arraycopy(fullBuffers(i), 0, rva.fullBuffers(i), 0, bufSize)
        }
        i += 1
      }
    } else {
      var i = 0
      while (i < fullBuffers.size) {
        rva.fullBuffers += Array.ofDim[Long](bufSize + 1)
        System.arraycopy(fullBuffers(i), 0, rva.fullBuffers(i), 0, bufferSizes(i))
        rva.bufferSizes(i) = bufferSizes(i)
        i += 1
      }
    }
    rva
  }

  def newInstance(): RegionValueAggregator = new RegionValueApproxCDFLongAggregator(bufSize)

  def clear(): Unit = {
    inBufSize = 0
    var i = 0
    while (i < fullBuffers.size) {
      if (fullBuffers(i) != null)
        bufferPool.push(fullBuffers(i))
      i += 1
    }
    fullBuffers.clear()
    fullBuffers += null
    bufferSizes.clear()
  }
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
    var i: Int = lStart
    var j: Int = rStart
    var o: Int = outStart
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

  def mergeNoSentinals(
    left: Array[Long], lStart: Int, lEnd: Int,
    right: Array[Long], rStart: Int, rEnd: Int,
    out: Array[Long], outStart: Int
  ): Unit = {
    var i: Int = lStart
    var j: Int = rStart
    var o: Int = outStart
    while (i < lEnd && j < rEnd) {
      if (left(i) < right(j)) {
        out(o) = left(i)
        i += 1
      } else {
        out(o) = right(j)
        j += 1
      }
      o += 1
    }
    while (i < lEnd) {
      out(o) = left(i)
      i += 1
      o += 1
    }
    while (j < rEnd) {
      out(o) = right(j)
      j += 1
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

  def compactBufferBackwards(buf: Array[Long], inStart: Int, inEnd: Int, out: Array[Long], outEnd: Int, skipFirst: Boolean): Unit = {
    var i = inEnd - 1
    var o = outEnd - 1
    if (skipFirst) {
      i -= 1
    }
    while (i >= inStart) {
      out(o) = buf(i)
      i -= 2
      o -= 1
    }
  }

  def floorOfLog2OfFraction(numer: Long, denom: Long) = {
    var count = 0
    var d = denom << 1
    while (d <= numer) {
      d <<= 1
      count += 1
    }
    count
  }

  def ubOnNumLevels(n: Long) = 1 + floorOfLog2OfFraction(n, 1)
}

object Main {
  def time(block: => Unit) = {
    val t0 = System.nanoTime()
    block
    val t1 = System.nanoTime()
    t1 - t0
  }

  def computeErrors(values: Array[Long], ranks: Array[Long], epsilon: Double): (Double, Double) = {
    val n = ranks(values.length)
    var i = 0
    var q = epsilon
    var totalError: Long = 0
    var maxError: Long = 0
    var numErrors = 0
    while (q < 1) {
      val rank = {
        val r = (q * n).floor.toLong
        if (r == n) r - 1 else r
      }
      while (ranks(i) <= rank) i += 1
      val error = scala.math.abs(values(i-1) - rank)
      totalError += error
      if (error > maxError) maxError = error
      numErrors += 1

      q += epsilon
    }
    (totalError.toDouble / numErrors / n, maxError.toDouble / n)
  }

  def main(args: Array[String]) = {
    import scala.math.{abs, pow}
    val agg = new RegionValueApproxCDFLongAggregator(80)
    val kllAgg = new RegionValueApproxCDFLongAggregator2(220, 8, false)
    val kllEager = new RegionValueApproxCDFLongAggregator2(220, 8, true)
    val n = 150000

    import scala.util.{Random, Sorting}
    val rand = new Random()
    val data = rand.shuffle(IndexedSeq.range(0, n)).toArray
    //    val data = Array.range(0, n)
    var cdf: (Array[Long], Array[Long]) = null
    var kllTime = time {
      var i = 0
      //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
      while (i < n) { kllAgg._seqOp(data(i)); i += 1 }
      //      agg.combOp(agg2.asInstanceOf[agg.type])
    }
    kllAgg.clear()
    kllTime = time {
      var i = 0
      //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
      while (i < n) { kllAgg._seqOp(data(i)); i += 1 }
      //      agg.combOp(agg2.asInstanceOf[agg.type])
      cdf = kllAgg.cdf
    }
    val (kllValues, kllRanks) = cdf
    var aggTime = time {
      var i = 0
      //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
      while (i < n) { agg._seqOp(data(i)); i += 1 }
      //      agg.combOp(agg2.asInstanceOf[agg.type])
    }
    agg.clear()
    aggTime = time {
      var i = 0
      //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
      while (i < n) { agg._seqOp(data(i)); i += 1 }
      //      agg.combOp(agg2.asInstanceOf[agg.type])
      cdf = agg.cdf
    }
    val (values, ranks) = cdf
    var kllEagerTime = time {
      var i = 0
      //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
      while (i < n) { kllEager._seqOp(data(i)); i += 1 }
      //      agg.combOp(agg2.asInstanceOf[agg.type])
    }
    kllEager.clear()
    kllEagerTime = time {
      var i = 0
      //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
      while (i < n) { kllEager._seqOp(data(i)); i += 1 }
      //      agg.combOp(agg2.asInstanceOf[agg.type])
      cdf = kllEager.cdf
    }
    val (kllEagValues, kllEagRanks) = cdf
    val scalaStableSortTime = time {
      data.sorted
    }
    val scalaSortTime = time {
      val data2 = Array.ofDim[Long](n)
      var i = 0
      while (i < n) {
        data2(i) = data(i)
        i += 1
      }
      Sorting.quickSort(data2)
    }
    val javaSortTime = time {
      val data2 = Array.ofDim[Long](n)
      var i = 0
      while (i < n) {
        data2(i) = data(i)
        i += 1
      }
      java.util.Arrays.sort(data2)
    }
    val factor = pow(10, 9)
    println(s"approx took ${ aggTime.toDouble / factor } s")
    println(s"KLL took ${ kllTime.toDouble / factor } s")
    println(s"Eager KLL took ${ kllEagerTime.toDouble / factor } s")
    println(s"java sort took ${ javaSortTime.toDouble / factor } s")
    println(s"scala quicksort took ${ scalaSortTime.toDouble / factor } s")
    println(s"stable sort took ${ scalaStableSortTime.toDouble / factor } s")

    println()

    //    for ((weight, value) <- cdf) println(s"$weight : $value")
    val (avgError, maxError) = computeErrors(values, ranks, 0.01)
    println(s"average error = $avgError")
    println(s"max error = $maxError")
    println(s"memory used = ${agg.memUsage}")

    println()

    val (kllAvgError, kllMaxError) = computeErrors(kllValues, kllRanks, 0.01)
    println(s"KLL average error = $kllAvgError")
    println(s"KLL max error = $kllMaxError")
    println(s"KLL memory used = ${kllAgg.memUsage}")

    println()

    val (kllEagAvgError, kllEagMaxError) = computeErrors(kllEagValues, kllEagRanks, 0.01)
    println(s"Eager KLL average error = $kllEagAvgError")
    println(s"Eager KLL max error = $kllEagMaxError")
    println(s"Eager KLL memory used = ${kllEager.memUsage}")
  }
}
