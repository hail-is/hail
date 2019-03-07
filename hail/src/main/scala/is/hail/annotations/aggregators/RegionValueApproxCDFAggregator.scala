package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types.physical._
import is.hail.utils.ArrayBuilder
import org.apache.spark.sql.Row
import net.sourceforge.jdistlib.rng.MersenneTwister

import scala.reflect.ClassTag
import scala.math.Ordering

object Sorting {
  implicit object IntSorting extends Sorting[Int] {
    def compare(x: Int, y: Int): Int = java.lang.Integer.compare(x, y)
    def lt(x: Int, y: Int): Boolean = x < y
    def sort(a: Array[Int], begin: Int, end: Int) {
      java.util.Arrays.sort(a, begin, end)
    }
    def dummyValue: Int = 0
  }
  implicit object LongSorting extends Sorting[Long] {
    def compare(x: Long, y: Long): Int = java.lang.Long.compare(x, y)
    def lt(x: Long, y: Long): Boolean = x < y
    def sort(a: Array[Long], begin: Int, end: Int) {
      java.util.Arrays.sort(a, begin, end)
    }
    def dummyValue: Long = 0
  }
  implicit object FloatSorting extends Sorting[Float] {
    def compare(x: Float, y: Float): Int = java.lang.Float.compare(x, y)
    def lt(x: Float, y: Float): Boolean = x < y
    def sort(a: Array[Float], begin: Int, end: Int) {
      java.util.Arrays.sort(a, begin, end)
    }
    def dummyValue: Float = 0
  }
  implicit object DoubleSorting extends Sorting[Double] {
    def compare(x: Double, y: Double): Int = java.lang.Double.compare(x, y)
    def lt(x: Double, y: Double): Boolean = x < y
    def sort(a: Array[Double], begin: Int, end: Int) {
      java.util.Arrays.sort(a, begin, end)
    }
    def dummyValue: Double = 0
  }
}

abstract class Sorting[@specialized(Int, Long, Float, Double) T] {
  private val rand = new MersenneTwister()

  def dummyValue: T

  def compare(x: T, y: T): Int
  def lt(x: T, y: T): Boolean

  def sort(a: Array[T], begin: Int, end: Int): Unit

  def merge(
    left: Array[T], lStart: Int, lEnd: Int,
    right: Array[T], rStart: Int, rEnd: Int,
    out: Array[T], outStart: Int
  ): Unit = {
    var i: Int = lStart
    var j: Int = rStart
    var o: Int = outStart
    while (i < lEnd && j < rEnd) {
      val c = compare(left(i), right(j))
      if (c < 0) {
        out(o) = left(i)
        i += 1
      } else if (c > 0) {
        out(o) = right(j)
        j += 1
      } else {
        out(o) = left(i)
        o += 1
        out(o) = right(j)
        i += 1
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

  def compactBuffer(
    buf: Array[T], inStart: Int, inEnd: Int,
    out: Array[T], outStart: Int
  ): Unit = {
    var i = inStart
    var o = outStart
    if (rand.nextBoolean()) {
      i += 1
    }
    while (i < inEnd) {
      out(o) = buf(i)
      i += 2
      o += 1
    }
  }

  def compactBufferBackwards(
    buf: Array[T], inStart: Int, inEnd: Int,
    out: Array[T], outEnd: Int
  ): Unit = {
    var i = inEnd - 1
    var o = outEnd - 1
    if (rand.nextBoolean()) {
      i -= 1
    }
    while (i >= inStart) {
      out(o) = buf(i)
      i -= 2
      o -= 1
    }
  }

  /* Compact level `level`, merging the compacted results into level `level+1`.
   *
   * Assumes level `level` contains an even number of elements. Leaves `buf` and
   * `levels` correct at level `level+1` and above. Does not modify `levels` at
   * `level` or below; that is the caller's responsibility.
   *
   * Returns new start index of level `level+1`.
   */
  def compactLevel(buf: Array[T], levels: Array[Int], level: Int): Int = {
    val a = levels(level)
    val c = levels(level + 1)
    val d = levels(level + 2)
    val size = c - a
    val sizeAbove = d - c

    assert((size & 1) == 0)

    val b = a + (size / 2)

    if (level == 0) sort(buf, a, c)
    if (sizeAbove == 0) {
      compactBufferBackwards(buf, a, c, buf, c)
    } else {
      compactBuffer(buf, a, c, buf, a)
      merge(buf, a, b, buf, c, d, buf, b)
    }
    levels(level + 1) = b
    b
  }

  /* Takes `inBuf` containing sorted levels, with no bound on size, and compacts
   * and copies into `outBuf`, ensuring that `outBuf` contains
   *
   */
  def generalCompact(
    k: Int, m: Int, numLevelsIn: Int,
    inBuf: Array[T], inLevels: Array[Int],
    outBuf: Array[T], outLevels: Array[Int]
  ): (Int, Int, Int) = {
    var numLevels = numLevelsIn
    var currentItemCount = inLevels(numLevels) - inLevels(0) // decreases with each compaction

    var targetItemCount = // increases if we add levels
      QuantilesAggregator.computeTotalCapacity(numLevels, k, m)

    outLevels(0) = 0

    var curLevel = 0
    while (curLevel < numLevels) {
      val start = inLevels(curLevel)
      val size = inLevels(curLevel + 1) - start

      if (currentItemCount < targetItemCount
          || size < QuantilesAggregator.levelCapacity(curLevel, numLevels, k, m)) { // copy level over as is
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
          targetItemCount += QuantilesAggregator.levelCapacity(0, numLevels, k, m)
        }
      }
      curLevel += 1
    }
    assert((outLevels(numLevels) - outLevels(0)) == currentItemCount)
    (numLevels, targetItemCount, currentItemCount)
  }
}

/* Compute an approximation to the sorted sequence of values seen.
 *
 * The result of the aggregator is an array "values" of samples, in
 * non-decreasing order, and an array "ranks" of integers less than `n`, in
 * increasing order, such that:
 * - ranks.length = values.length + 1
 * - ranks(0) = 0
 * - ranks(values.length) = n
 * These represent a summary of the sorted list of values seen by the
 * aggregator. For example, values=[0,2,5,6,9] and ranks=[0,3,4,5,8,10]
 * represents the approximation [0,0,0,2,5,6,6,6,9,9], with the value
 * `values(i)` occupying indices `ranks(i)` to `ranks(i+1)` (again half-open).
 */
class RegionValueApproxCDFAggregator[@specialized(Int, Long, Float, Double) T: ClassTag : Ordering](
  val k: Int, val m: Int, growthRate: Int = 4, eager: Boolean
)(implicit sorting: Sorting[T]
) extends RegionValueAggregator {

  val resultType: PType = PStruct("values" -> PArray(PInt64()), "ranks" -> PArray(PInt64()))

  /* The sketch maintains a sample of items seen, organized into levels. The
   * levels are stored contiguously in `items`, right justified. Level i
   * starts at index `levels(i)` (inclusive), and goes to `levels(i+1)`
   * (exclusive). New items are added to level 0, from right to left. Levels
   * above 0 are maintained in sorted order.
   *
   * Samples in level i represent 2^i items from the original stream. Whenever
   * `items` fills up, we make room by "compacting" a full level. Compacting
   * means sorting (if the level wasn't already sorted), throwing away every
   * other sample (taking the evens or the odds with equal probability), and
   * adding the remaining samples to the level above (where now each kept sample
   * represents twice as many items).
   *
   * Let `levelCapacity(i)`=k*(2/3)^(numLevels-i). A compaction operation at
   * level i is correct if the level contains at least `levelCapacity(i)`
   * samples at the time of compaction. As long as this holds, the analysis from
   * the paper [KLL] applies. This leaves room for several compaction
   * strategies, of which we implement two, with the `eager` flag choosing
   * between them.
   *
   * To keep things simple, we require level i contains at least m samples at
   * the time of compaction, where `m` is a class parameter, m>=2, controlling
   * the minimum size of a compaction. Because of this minimum size, we must
   * (very slowly) grow the `items` buffer over time.
   *
   * To maintain the correct total weight, we only compact an even number of
   * samples. If a level contains an odd number of samples when compacting,
   * we leave one sample at the lower level.
   *
   * Invariants:
   * - `n` is the number of items seen.
   * - `items` contains the samples collected so far. Each level above 0 is
   *   always sorted in non-decreasing order.
   * - `levels` tracks the boundaries of the levels stored in `items`. It is
   *   always non-decreasing, and `levels(numLevels)` always equals `items.length`.
   * - `levelsCapacity` is the number of levels `items` and `levels` have room
   *   for before we need to reallocate.
   * - `items.length` is always the sum of all level capacities up to
   *   `levelsCapacity`, assuming `numLevels=levelsCapacity`. Thus if `items`
   *   is full, at least one level must be full.
   * - `numLevels` is the number of levels currently held. The top level is
   *   never empty, so this is also the greatest nonempty level.
   */

  var n: Long = 0
  var levelsCapacity = QuantilesAggregator.findInitialLevelsCapacity(k, m)
  var numLevels = 1
  var levels: Array[Int] = Array.fill[Int](levelsCapacity + 1)(QuantilesAggregator.computeTotalCapacity(levelsCapacity, k, m))
  var items: Array[T] = Array.ofDim[T](levels(0))

  private def size: Int = levels(numLevels) - levels(0)

  private[aggregators] def memUsage: Int = items.length

  def seqOp(region: Region, x: T, missing: Boolean) {
    if (!missing) _seqOp(x)
  }

  private[aggregators] def _seqOp(x: T) {
    if (levels(0) == 0) {
      if (eager)
        compactEager(sorting.dummyValue)
      else
        compact(sorting.dummyValue)
    }
    n += 1
    val newBot = levels(0) - 1
    items(newBot) = x
    levels(0) = newBot
  }

  def combOp(other: RegionValueAggregator) {
    _combOp(other.asInstanceOf[RegionValueApproxCDFAggregator[T]])
  }

  private[aggregators] def _combOp(other: RegionValueApproxCDFAggregator[T]) {
    assert (m == other.m)
    if (other.numLevels == 1) {
      var i = other.levels(0)
      while (i < other.levels(1)) {
        _seqOp(other.items(i))
        i += 1
      }
    } else {
      merge(other)
    }
  }

  def result(rvb: RegionValueBuilder): Unit = {
    val res = Row.fromTuple(cdf)
    rvb.addAnnotation(resultType.virtualType, res)
  }

  private[aggregators] def cdf: (Array[T], Array[Long]) = {
    val builder: ArrayBuilder[(Long, T)] = new ArrayBuilder(0)

    var level = 0
    while (level < numLevels) {
      val weight: Long = 1 << level
      var i = levels(level)
      while (i < levels(level + 1)) {
        builder += (weight, items(i))
        i += 1
      }
      level += 1
    }

    val sorted = builder.result().sortBy(_._2)

    val values = Array.ofDim[T](sorted.length)
    var i = 0
    while (i < sorted.length) {
      values(i) = sorted(i)._2
      i += 1
    }

    val ranks = Array.ofDim[Long](sorted.length + 1)
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

  def clear() {
    n = 0
    numLevels = 1
    var i = 0
    while (i < levels.length) {
      levels(i) = items.length
      i += 1
    }
  }

  def newInstance(): RegionValueApproxCDFAggregator[T] =
    new RegionValueApproxCDFAggregator[T](k, m, growthRate, eager)

  def copy(): RegionValueApproxCDFAggregator[T] = {
    val newAgg = newInstance()
    newAgg.n = n
    newAgg.levelsCapacity = levelsCapacity
    newAgg.numLevels = numLevels
    newAgg.items = items.clone()
    newAgg.levels = levels.clone()
    newAgg
  }

  private def findFullLevel(): Int = {
    var level: Int = 0
    while (levels(level + 1) - levels(level) < levelCapacity(level)) {
      level += 1
    }
    level
  }

  private def levelCapacity(level: Int, numLevels: Int = numLevels): Int =
    QuantilesAggregator.levelCapacity(level, numLevels, k, m)

  /* Compact the first over-capacity level. If that is the top level, grow the
   * sketch.
   */
  private def compact(dummy: T) {
    assert(levels(0) == 0)
    val level = findFullLevel()
    if (level == numLevels - 1) growFullSketch()

    val start = levels(level)
    val size = levels(level + 1) - levels(level)
    val adj = size % 2
    val halfSize = size / 2

    levels(level) += adj
    val end = sorting.compactLevel(items, levels, level)

    // If size was odd, need to copy extra element up. If not, this is still safe.
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
  }

  /* If we are following the eager compacting strategy, level 0 must be full
   * when starting a compaction. After each compaction, if the next level is
   * then full, we compact it. This strategy sacrifices some accuracy, but
   * avoids having to shift up items below the compacted level.
   */
  private def compactEager(dummy: T) {
    assert(levels(1) - levels(0) >= levelCapacity(0))

    def compactAndGrow(level: Int, singletons: Int) {
      val curTotalCap = levels(numLevels)
      assert(items.length == curTotalCap)
      assert(level == levelsCapacity - 1)
      val size: Int = levels(level + 1) - levels(level)
      assert(size % 2 == 0)

      numLevels += 1
      val deltaCap = m * growthRate
      val newTotalCap = curTotalCap + deltaCap
      val newBuf = Array.ofDim[T](newTotalCap)

      levelsCapacity += growthRate
      levels = levels.padTo(levelsCapacity + 1, newTotalCap)

      val halfSize = size / 2
      sorting.compactBuffer(items, levels(level), levels(level + 1), newBuf, newTotalCap - halfSize)

      levels(level + 1) = newTotalCap - halfSize

      System.arraycopy(items, 0, newBuf, 0, singletons)
      items = newBuf
    }

    var level = 0
    var singletons = 0
    var desiredFreeCapacity = 0
    var grew = false
    do {
      val size = levels(level + 1) - levels(level)
      assert(size >= levelCapacity(level))
      val adj = size % 2

      items(singletons) = items(levels(level))

      levels(level) += adj

      if (level >= numLevels - 1) {
        if (level == levelsCapacity - 1) {
          compactAndGrow(level, singletons)
        } else {
          numLevels += 1
          sorting.compactLevel(items, levels, level)
        }
        assert(levels(numLevels) >= computeTotalCapacity())
        grew = true
      } else {
        sorting.compactLevel(items, levels, level)
      }

      levels(level) = singletons
      singletons += adj
      desiredFreeCapacity += levelCapacity(level)
      level += 1
    } while (levels(level) < desiredFreeCapacity && !grew)
    val shift = levels(level) - singletons
    var i = 0
    while (i < level) {
      levels(i) += shift
      i += 1
    }
    System.arraycopy(items, 0, items, shift, singletons)
  }

  private def growFullSketch() {
    val curTotalCap = levels(numLevels)
    assert(levels(0) == 0)
    assert(items.length == curTotalCap)

    numLevels += 1
    if (numLevels > levelsCapacity) {
      levelsCapacity += growthRate
      levels = levels.padTo(levelsCapacity + growthRate + 1, curTotalCap)

      val deltaCap = m * growthRate
      val newTotalCap = curTotalCap + deltaCap
      val newBuf = Array.ofDim[T](newTotalCap)

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

  private def sortLevel0() { sorting.sort(items, levels(0), levels(1)) }

  private def merge(other: RegionValueApproxCDFAggregator[T]) {
    val finalN = n + other.n
    val workbuf = Array.ofDim[T](size + other.size)
    val ub = QuantilesAggregator.ubOnNumLevels(finalN)
    val worklevels = new Array[Int](ub + 2)
    // ub+1 does not work
    val outlevels = new Array[Int](ub + 2)

    val provisionalNumLevels = math.max(numLevels, other.numLevels)
    sortLevel0()
    other.sortLevel0()
    levelwiseMerge(other, workbuf, worklevels)

    // notice that workbuf is being used as both the input and output here
    val (finalNumLevels, finalCapacity, finalPop) =
      sorting.generalCompact(k, m, provisionalNumLevels,
        workbuf, worklevels,
        workbuf, outlevels)
    assert(finalNumLevels <= ub)

    if (finalNumLevels > levelsCapacity) {
      items = Array.ofDim[T](finalCapacity)
      levels = Array.ofDim[Int](finalNumLevels + 1)
      levelsCapacity = finalNumLevels
    }

    val freeSpaceAtBottom = items.length - finalPop
    System.arraycopy(workbuf, outlevels(0), items, freeSpaceAtBottom, finalPop)

    val offset = freeSpaceAtBottom - outlevels(0)
    var lvl = 0
    while (lvl < finalNumLevels + 1) {
      levels(lvl) = outlevels(lvl) + offset
      lvl += 1
    }
    while (lvl < levels.length) {
      levels(lvl) = items.length
      lvl += 1
    }

    numLevels = finalNumLevels
    n = finalN
  }

  private def levelwiseMerge(
    other: RegionValueApproxCDFAggregator[T],
    workbuf: Array[T],
    worklevels: Array[Int]
  ) {
    worklevels(0) = 0

    var lvl = 0
    while (lvl < worklevels.length - 1) {
      val selfPop = levelSize(lvl)
      val otherPop = other.levelSize(lvl)
      worklevels(lvl + 1) = worklevels(lvl) + selfPop + otherPop

      if (selfPop > 0 && otherPop > 0)
        sorting.merge(
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

  private def computeTotalCapacity(numLevels: Int = numLevels): Int =
    QuantilesAggregator.computeTotalCapacity(numLevels, k, m)

  private def levelSize(level: Int): Int =
    if (level >= numLevels) 0 else levels(level + 1) - levels(level)
}

object QuantilesAggregator {
  def floorOfLog2OfFraction(numer: Long, denom: Long): Int = {
    var count = 0
    var d = denom << 1
    while (d <= numer) {
      d <<= 1
      count += 1
    }
    count
  }

  def ubOnNumLevels(n: Long): Int = 1 + floorOfLog2OfFraction(n, 1)

  // 0 <= power <= 30
  private val powersOfThree = Array[Long](1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683,
    59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721, 129140163,
    387420489, 1162261467, 3486784401L, 10460353203L, 31381059609L,
    94143178827L, 282429536481L, 847288609443L, 2541865828329L, 7625597484987L,
    22876792454961L, 68630377364883L, 205891132094649L)

  // compute k * (2/3)^depth rounded to the nearest integer
  def depthCapacity(depth: Int, k: Int): Int = {
    assert(depth <= 30)
    val tmp = ((k.toLong << (depth + 1)) / powersOfThree(depth)).toInt
    val result = (tmp + 1) >> 1
    result
  }

  def levelCapacity(level: Int, numLevels: Int, k: Int, m: Int): Int =
    math.max(m, depthCapacity(numLevels - level - 1, k))

  def findInitialLevelsCapacity(k: Int, m: Int): Int = {
    var numLevels = 0
    while (depthCapacity(numLevels, k) >= m) numLevels += 1
    numLevels + 1
  }

  def computeTotalCapacity(numLevels: Int, k: Int, m: Int): Int = {
    var total = 0
    var h = 0
    while (h < numLevels) {
      total += levelCapacity(h, numLevels, k, m)
      h += 1
    }
    total
  }
}

object Main {
  def time(block: => Unit) = {
    val t0 = System.nanoTime()
    block
    val t1 = System.nanoTime()
    t1 - t0
  }

  def computeErrors(values: Array[Int], ranks: Array[Long], epsilon: Double): (Double, Double) = {
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
    import scala.util.{Random, Sorting}

    val k = 150
    val n = 150000
    val preReps = 50
    val reps = 500

    //    val kllEager = new RegionValueApproxCDFLongAggregator2(100, 32, false)
    val kllAgg = new RegionValueApproxCDFAggregator[Int](k, 8, 1, false)
//    val kllAgg2 = new RegionValueApproxCDFAggregator[Int](k, 8, 1, false)

    val rand = new Random()
    val data = rand.shuffle(IndexedSeq.range(0, n)).toArray
    //    val data = Array.range(0, n)
    val results: ArrayBuilder[(Array[Int], Array[Long])] = new ArrayBuilder(reps)

    time {
      var rep = 0
      while (rep < preReps) {
        var i = 0
        //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
        while (i < n) {
          kllAgg._seqOp(data(i));
          i += 1
        }
        //      agg.combOp(agg2.asInstanceOf[agg.type])
        rep += 1
      }
    }
    val kllTime = time {
      var rep = 0
      while (rep < reps) {
        kllAgg.clear()
//        kllAgg2.clear()
        var i = 0
//        while (i < n/2) { kllAgg2._seqOp(data(i)); i += 1 }
        while (i < n) { kllAgg._seqOp(data(i)); i += 1 }
//        kllAgg._combOp(kllAgg2)
        results += kllAgg.cdf
        rep += 1
      }
    }
    val kllResults = results.result()
    results.clear()

//    time {
//      var i = 0
//      //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
//      while (i < n) { kllEager._seqOp(data(i)); i += 1 }
//      //      agg.combOp(agg2.asInstanceOf[agg.type])
//    }
//    val kllEagerTime = time {
//      var rep = 0
//      while (rep < reps) {
//        kllEager.clear()
//        var i = 0
//        //      while (i < n/2) { agg2.seqOp(data(i)); i += 1 }
//        while (i < n) {
//          kllEager._seqOp(data(i)); i += 1
//        }
//        //      agg.combOp(agg2.asInstanceOf[agg.type])
//        results += kllEager.cdf
//        rep += 1
//      }
//    }
//    val kllEagResults = results.result()
//    results.clear()

//    val scalaStableSortTime = time {
//      var rep = 0
//      while (rep < reps) { data.sorted; rep += 1 }
//    }
//    val scalaSortTime = time {
//      var rep = 0
//      while (rep < reps) {
//        val data2 = Array.ofDim[Long](n)
//        var i = 0
//        while (i < n) {
//          data2(i) = data(i)
//          i += 1
//        }
//        Sorting.quickSort(data2)
//        rep += 1
//      }
//    }

    val javaSortTime = time {
      var rep = 0
      while (rep < reps) {
        val data2 = Array.ofDim[Int](n)
        var i = 0
        while (i < n) {
          data2(i) = data(i)
          i += 1
        }
        java.util.Arrays.sort(data2)
//        QuickSort.sort(data2)
        rep += 1
      }
    }

    val factor = pow(10, 9)
    println(s"KLL took ${ kllTime.toDouble / factor / reps } s")
//    println(s"Eager KLL took ${ kllEagerTime.toDouble / factor / reps } s")
    println(s"java sort took ${ javaSortTime.toDouble / factor / reps } s")
//    println(s"scala quicksort took ${ scalaSortTime.toDouble / factor / reps } s")
//    println(s"stable sort took ${ scalaStableSortTime.toDouble / factor / reps } s")

    println()

    val (kllAvgErrors, kllMaxErrors) = kllResults.map { case (values, ranks) => computeErrors(values, ranks, 0.01) }.unzip
    println(s"KLL average error = ${ kllAvgErrors.sum / reps }")
    println(s"KLL max error = ${ kllMaxErrors.sum / reps }")
    println(s"KLL memory used = ${ kllAgg.memUsage }")

//    println()
//
//    val (kllEagAvgErrors, kllEagMaxErrors) = kllEagResults.map { case (values, ranks) => computeErrors(values, ranks, 0.01) }.unzip
//    println(s"Eager KLL average error = ${ kllEagAvgErrors.sum / reps }")
//    println(s"Eager KLL max error = ${ kllEagMaxErrors.sum / reps }")
//    println(s"Eager KLL memory used = ${ kllEager.memUsage }")
  }
}
