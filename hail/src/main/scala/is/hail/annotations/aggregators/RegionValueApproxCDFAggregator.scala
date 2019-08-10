package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row

import scala.reflect.ClassTag
import scala.math.Ordering

object ApproxCDFHelper {
  implicit object IntHelper extends ApproxCDFHelper[Int] {
    def compare(x: Int, y: Int): Int = java.lang.Integer.compare(x, y)
    def lt(x: Int, y: Int): Boolean = x < y
    def sort(a: Array[Int], begin: Int, end: Int) {
      java.util.Arrays.sort(a, begin, end)
    }
    def dummyValue: Int = 0
    val hailType = TInt32()
  }
  implicit object LongHelper extends ApproxCDFHelper[Long] {
    def compare(x: Long, y: Long): Int = java.lang.Long.compare(x, y)
    def lt(x: Long, y: Long): Boolean = x < y
    def sort(a: Array[Long], begin: Int, end: Int) {
      java.util.Arrays.sort(a, begin, end)
    }
    def dummyValue: Long = 0
    val hailType = TInt64()
  }
  implicit object FloatHelper extends ApproxCDFHelper[Float] {
    def compare(x: Float, y: Float): Int = java.lang.Float.compare(x, y)
    def lt(x: Float, y: Float): Boolean = x < y
    def sort(a: Array[Float], begin: Int, end: Int) {
      java.util.Arrays.sort(a, begin, end)
    }
    def dummyValue: Float = 0
    val hailType = TFloat32()
  }
  implicit object DoubleHelper extends ApproxCDFHelper[Double] {
    def compare(x: Double, y: Double): Int = java.lang.Double.compare(x, y)
    def lt(x: Double, y: Double): Boolean = x < y
    def sort(a: Array[Double], begin: Int, end: Int) {
      java.util.Arrays.sort(a, begin, end)
    }
    def dummyValue: Double = 0
    val hailType = TFloat64()
  }
}

abstract class ApproxCDFHelper[@specialized(Int, Long, Float, Double) T] extends Serializable {
  val hailType: Type

  def dummyValue: T

  def compare(x: T, y: T): Int
  def lt(x: T, y: T): Boolean

  def min(x: T, y: T): T = if (lt(x, y)) x else y
  def max(x: T, y: T): T = if (lt(x, y)) y else x

  def sort(a: Array[T], begin: Int, end: Int): Unit

  def merge(
    left: Array[T], lStart: Int, lEnd: Int,
    right: Array[T], rStart: Int, rEnd: Int,
    out: Array[T], outStart: Int
  ): Unit = {
    assert((left ne out) || (outStart <= lStart - (rEnd - rStart)) || (outStart >= lEnd))
    assert((right ne out) || (outStart <= rStart - (lEnd - lStart)) || (outStart >= rEnd))
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

  def reverse(buf: Array[T], start: Int, end: Int): Unit = {
    var left = start
    var right = end
    while (true) {
      if (left == right) return
      right -= 1
      if (left == right) return
      val temp = buf(left)
      buf(left) = buf(right)
      buf(right) = temp
      left += 1
    }
  }

  def rotate(buf: Array[T], start: Int, mid: Int, end: Int): Int = {
    if (start == mid) return end
    if (mid == end) return start
    reverse(buf, start, mid)
    reverse(buf, mid, end)
    reverse(buf, start, end)
    start + (end - mid)
  }

  def compactBuffer(
    buf: Array[T], inStart: Int, inEnd: Int,
    out: Array[T], outStart: Int,
    skipFirst: Boolean
  ): Unit = {
    assert((buf ne out) || (outStart <= inStart) || (outStart >= inEnd))
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

  def compactBufferBackwards(
    buf: Array[T], inStart: Int, inEnd: Int,
    out: Array[T], outEnd: Int,
    skipFirst: Boolean
  ): Unit = {
    assert((buf ne out) || (outEnd <= inStart) || (outEnd >= inEnd))
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
}

object ApproxCDFCombiner {
  def apply[@specialized(Int, Long, Float, Double) T: ClassTag : Ordering : ApproxCDFHelper](
    numLevels: Int, capacity: Int, keepRatio: Option[Double], dummy: T, rand: java.util.Random
  ): ApproxCDFCombiner[T] = new ApproxCDFCombiner[T](
    Array.fill[Int](numLevels + 1)(capacity),
    Array.ofDim[T](capacity),
    Array.fill[Int](numLevels)(0),
    1,
    keepRatio.getOrElse(Double.NaN),
    rand)

  def apply[@specialized(Int, Long, Float, Double) T: ClassTag : Ordering : ApproxCDFHelper](
    numLevels: Int, capacity: Int, keepRatio: Option[Double], dummy: T
  ): ApproxCDFCombiner[T] =
    apply(numLevels, capacity, keepRatio, dummy, new java.util.Random())
}

/* Keep a collection of values, grouped into levels.
 *
 * Invariants:
 * - `items` stores all levels contiguously. Each level above 0 is
 *   always sorted in non-decreasing order.
 * - `levels` tracks the boundaries of the levels stored in `items`. It is
 *   always non-decreasing, and `levels(numLevels)` always equals `items.length`.
 *   The values in level i occupy indices from `levels(i)` (inclusive) to
 *   `levels(i+1)` (exclusive).
 * - `numLevels` is the number of levels currently held. The top level is
 *   never empty, so this is also the greatest nonempty level.
 */
class ApproxCDFCombiner[@specialized(Int, Long, Float, Double) T: ClassTag : Ordering](
  val levels: Array[Int],
  val items: Array[T],
  val compactionCounts: Array[Int],
  var numLevels: Int,
  val keepRatio: Double,
  val rand: java.util.Random
)(implicit helper: ApproxCDFHelper[T]
) extends Serializable {

  def copy(): ApproxCDFCombiner[T] =
    new ApproxCDFCombiner[T](levels.clone(), items.clone(), compactionCounts.clone(), numLevels, keepRatio, rand)

  def maxNumLevels = levels.length - 1
  def capacity = items.length
  def isFull = levels(0) == 0

  def size = levels(maxNumLevels) - levels(0)

  def levelSize(level: Int): Int = levels(level + 1) - levels(level)

  def safeLevelSize(level: Int): Int =
    if (level >= maxNumLevels) 0 else levels(level + 1) - levels(level)

  def push(t: T) {
    val bot = levels(0)

    val newBot = bot - 1
    items(newBot) = t
    levels(0) = newBot
  }

  def grow(newNumLevels: Int, newCapacity: Int, dummy: T = helper.dummyValue): ApproxCDFCombiner[T] = {
    require(newNumLevels > maxNumLevels && newCapacity > capacity)
    val newLevels = Array.ofDim[Int](newNumLevels + 1)
    val newItems = Array.ofDim[T](newCapacity)
    val newCompactionCounts = Array.fill[Int](newNumLevels)(0)
    val shift = newCapacity - capacity
    var i = 0
    while (i < maxNumLevels) {
      newLevels(i) = levels(i) + shift
      newCompactionCounts(i) = compactionCounts(i)
      i += 1
    }
    while (i <= newNumLevels) {
      newLevels(i) = newCapacity
      i += 1
    }
    System.arraycopy(items, levels(0), newItems, newLevels(0), size)

    new ApproxCDFCombiner[T](newLevels, newItems, newCompactionCounts, numLevels, keepRatio, rand)
  }

  def clear() {
    numLevels = 1
    var i = 0
    while (i < levels.length) {
      levels(i) = items.length
      i += 1
    }
  }

  /* Compact level `level`, merging the compacted results into level `level+1`,
   * keeping the 'keep' smallest and 'keep' largest values at 'level'. If
   * 'shiftLowerLevels' is true, shift lower levels up to keep items contiguous.
   *
   * Returns the new end of 'level'. If 'shiftLowerLevels', this is always
   * equal to 'levels(level + 1)`.
   */
  def compactLevel(level: Int, shiftLowerLevels: Boolean = true, dummy: T = helper.dummyValue): Int = {
    val keep = if (keepRatio.isNaN) {
      if (level == 0) 1 else 0
    } else {
      (levelSize(level) * keepRatio).toInt + 1
    }

    val levelEnd = _compactLevel(level, keep)

    val bot = levels(0)
    val sizeBelow = levelEnd - bot
    val shift = levels(level + 1) - levelEnd

    if (shiftLowerLevels) {
      if (sizeBelow > 1) {
        System.arraycopy(items, bot, items, bot + shift, sizeBelow)
      } else {
        // only needs to be done if sizeBelow == 1, but doesn't hurt otherwise
        items(levels(level + 1) - 1) = items(bot)
      }

      var lvl = 0
      while (lvl <= level) {
        levels(lvl) += shift
        lvl += 1
      }
    }

    levelEnd
  }

  // Compacts level 'level` into 'level + 1', keeping the 'keep' smallest and
  // 'keep' largest values at 'level'. Returns 'a', the new end of 'level'.
  // To return to a valid state, the interval ['levels(0)', 'a') must be shifted
  // up to end at 'levels(level + 1)`.
  def _compactLevel(level: Int, keep: Int, dummy: T = helper.dummyValue): Int = {
    assert(level <= numLevels - 1)
    if (level == numLevels - 1) numLevels += 1
    compactionCounts(level) += 1

    // before rotate:
    // |         level         | level + 1 |
    // | keep | compact | keep |           |
    // a0                      c           d

    // after rotate:
    // |        level          | level + 1 |
    // | keep | keep | compact |           |
    // a0            a    b    c           d

    // after compact:
    // |    level    |    |   level + 1    |
    // a0            a    b    c           d

    val a0 = levels(level)
    val c = levels(level + 1)
    val d = levels(level + 2)

    val adj = (c - a0) % 2

    val levelAboveSize = d - c

    if (level == 0) helper.sort(items, a0, c)
    val a = helper.rotate(items, a0 + keep + adj, c - keep, c)
    val halfSize = (c - a) / 2
    val b = a + halfSize

    if (levelAboveSize == 0) {
      helper.compactBufferBackwards(items, a, c, items, c, rand.nextBoolean())
    } else {
      helper.compactBuffer(items, a, c, items, a, rand.nextBoolean())
      helper.merge(items, a, b, items, c, d, items, b)
    }
    levels(level + 1) = b

    a
  }

  def merge(other: ApproxCDFCombiner[T], ubOnNumLevels: Int): ApproxCDFCombiner[T] = {
    val mergedLevels = Array.ofDim[Int](ubOnNumLevels + 1)
    val mergedItems = Array.ofDim[T](size + other.size)

    val selfPop = levelSize(0)
    val otherPop = other.levelSize(0)
    System.arraycopy(items, levels(0), mergedItems, 0, selfPop)
    System.arraycopy(other.items, other.levels(0), mergedItems, selfPop, otherPop)

    mergedLevels(0) = 0
    mergedLevels(1) = selfPop + otherPop

    var lvl = 1
    while (lvl < mergedLevels.length - 1) {
      val selfPop = safeLevelSize(lvl)
      val otherPop = other.safeLevelSize(lvl)
      mergedLevels(lvl + 1) = mergedLevels(lvl) + selfPop + otherPop

      if (selfPop > 0 && otherPop > 0)
        helper.merge(
          items, levels(lvl), levels(lvl + 1),
          other.items, other.levels(lvl), other.levels(lvl + 1),
          mergedItems, mergedLevels(lvl))
      else if (selfPop > 0)
        System.arraycopy(items, levels(lvl), mergedItems, mergedLevels(lvl), selfPop)
      else if (otherPop > 0)
        System.arraycopy(other.items, other.levels(lvl), mergedItems, mergedLevels(lvl), otherPop)

      lvl += 1
    }

    val mergedCompactionCounts = Array.fill[Int](ubOnNumLevels)(0)
    lvl = 0
    while (lvl < compactionCounts.length) {
      mergedCompactionCounts(lvl) += compactionCounts(lvl)
      lvl += 1
    }
    lvl = 0
    while (lvl < other.compactionCounts.length) {
      mergedCompactionCounts(lvl) += other.compactionCounts(lvl)
      lvl += 1
    }

    new ApproxCDFCombiner[T](
      mergedLevels,
      mergedItems,
      mergedCompactionCounts,
      math.max(numLevels, other.numLevels),
      keepRatio,
      rand)
  }

  def generalCompact(minCapacity: Int, levelCapacity: (Int, Int) => Int, dummy: T = helper.dummyValue) {
    var currentItemCount = levels(numLevels) - levels(0) // decreases with each compaction
    var targetItemCount = { // increases if we add levels
      var lvl = 0
      var acc = 0
      while (lvl < numLevels) {
        acc += levelCapacity(lvl, numLevels)
        lvl += 1
      }
      acc
    }

    var curLevel = 0
    var endOfCompacted = 0

    while (curLevel < numLevels) {
      val start = levels(curLevel)
      var lvlSize = levels(curLevel + 1) - start

      if (currentItemCount >= targetItemCount && lvlSize >= levelCapacity(curLevel, numLevels)) {
        // The sketch is too full AND this level is too full, so we compact it
        // Note: this can add a level and thus change the sketches capacities
        val curLevelEnd = compactLevel(curLevel, shiftLowerLevels = false)

        lvlSize = curLevelEnd - start
        currentItemCount -= (levels(curLevel + 1) - curLevelEnd)
        if (curLevel == (numLevels - 1)) {
          numLevels += 1
          targetItemCount += levelCapacity(0, numLevels)
        }
      }

      System.arraycopy(items, start, items, endOfCompacted, lvlSize)
      levels(curLevel) = endOfCompacted
      endOfCompacted = levels(curLevel) + lvlSize

      curLevel += 1
    }

    while (curLevel <= maxNumLevels) {
      levels(curLevel) = endOfCompacted
      curLevel += 1
    }
  }

  def copyFrom(other: ApproxCDFCombiner[T]) {
    assert(capacity >= other.size)
    assert(maxNumLevels >= other.numLevels)

    val freeSpaceAtBottom = items.length - other.size

    System.arraycopy(other.items, other.levels(0), items, freeSpaceAtBottom, other.size)

    val offset = freeSpaceAtBottom - other.levels(0)
    var lvl = 0
    while (lvl < other.numLevels) {
      levels(lvl) = other.levels(lvl) + offset
      compactionCounts(lvl) = other.compactionCounts(lvl)
      lvl += 1
    }
    while (lvl < levels.length) {
      levels(lvl) = items.length
      lvl += 1
    }

    numLevels = other.numLevels
  }

  def cdf: (Array[T], Array[Long]) = {
    val builder: ArrayBuilder[(Long, T)] = new ArrayBuilder(size)

    var level = 0
    while (level < numLevels) {
      val weight: Long = 1 << level
      var i = levels(level)
      while (i < levels(level + 1)) {
        builder += (weight -> items(i))
        i += 1
      }
      level += 1
    }

    val sorted = builder.result().sortBy(_._2)

    val values = new ArrayBuilder[T]
    val ranks = new ArrayBuilder[Long]
    var rank: Long = 0
    var i = 0
    ranks += 0
    while (i < sorted.length) {
      rank += sorted(i)._1
      if (i == sorted.length - 1 || sorted(i)._2 != sorted(i + 1)._2) {
        values += sorted(i)._2
        ranks += rank
      }
      i += 1
    }

    (values.result(), ranks.result())
  }
}

class RegionValueApproxCDFIntAggregator(k: Int) extends RegionValueApproxCDFAggregator[Int](k, 8, 4, false) {
  override def newInstance(): RegionValueApproxCDFIntAggregator = {
    new RegionValueApproxCDFIntAggregator(k)
  }

  override def copy(): RegionValueApproxCDFIntAggregator = {
    val newAgg = newInstance()
    newAgg.n = n
    newAgg.combiner = combiner.copy()
    newAgg
  }
}

class RegionValueApproxCDFLongAggregator(k: Int) extends RegionValueApproxCDFAggregator[Long](k, 8, 4, false) {
  override def newInstance(): RegionValueApproxCDFLongAggregator = {
    new RegionValueApproxCDFLongAggregator(k)
  }

  override def copy(): RegionValueApproxCDFLongAggregator = {
    val newAgg = newInstance()
    newAgg.n = n
    newAgg.combiner = combiner.copy()
    newAgg
  }
}

class RegionValueApproxCDFFloatAggregator(k: Int) extends RegionValueApproxCDFAggregator[Float](k, 8, 4, false) {
  override def newInstance(): RegionValueApproxCDFFloatAggregator = {
    new RegionValueApproxCDFFloatAggregator(k)
  }

  override def copy(): RegionValueApproxCDFFloatAggregator = {
    val newAgg = newInstance()
    newAgg.n = n
    newAgg.combiner = combiner.copy()
    newAgg
  }
}

class RegionValueApproxCDFDoubleAggregator(k: Int) extends RegionValueApproxCDFAggregator[Double](k, 8, 4, false) {
  override def newInstance(): RegionValueApproxCDFDoubleAggregator = {
    new RegionValueApproxCDFDoubleAggregator(k)
  }

  override def copy(): RegionValueApproxCDFDoubleAggregator = {
    val newAgg = newInstance()
    newAgg.n = n
    newAgg.combiner = combiner.copy()
    newAgg
  }
}

/* Compute an approximation to the sorted sequence of values seen.
 *
 * Let `n` be the number of non-missing values seen, and let `m` and `M` be
 * respectively the minimum and maximum values seen. The result of the
 * aggregator is an array "values" of samples, in increasing order, and an array
 * "ranks" of integers less than `n`, in increasing order, such that:
 * - ranks.length = values.length + 1
 * - ranks(0) = 0
 * - ranks(values.length) = n
 * - values(0) = m
 * - values(values.length - 1) = M
 * These represent a summary of the sorted list of values seen by the
 * aggregator. For example, values=[0,2,5,6,9] and ranks=[0,3,4,5,8,10]
 * represents the approximation [0,0,0,2,5,6,6,6,9,9], with the value
 * `values(i)` occupying indices `ranks(i)` to `ranks(i+1)` (again half-open).
 */
abstract class RegionValueApproxCDFAggregator[@specialized(Int, Long, Float, Double) T: ClassTag : Ordering](
  val k: Int, val m: Int = 8, growthRate: Int = 4, eager: Boolean = false, relError: Option[Double] = None
)(implicit helper: ApproxCDFHelper[T]
) extends RegionValueAggregator {

  /* The sketch maintains a sample of items seen, organized into levels.
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
   * To keep things simple, we require that any level contains a minimum of m
   * samples at the time of compaction, where `m` is a class parameter, m>=2,
   * controlling the minimum size of a compaction. Because of this minimum size,
   * we must (very slowly) grow the `items` buffer over time.
   *
   * To maintain the correct total weight, we only compact even numbers of
   * samples. If a level contains an odd number of samples when compacting,
   * we leave one sample at the lower level.
   *
   * Invariants:
   * - `n` is the number of items seen.
   * - `levelsCapacity` is the number of levels `items` and `levels` have room
   *   for before we need to reallocate.
   * - `numLevels` is the number of levels currently held. The top level is
   *   never empty, so this is also the greatest nonempty level.
   * - `items.length` is always at least the sum of all level capacities up to
   *   `numLevels`. Thus if `items` is full, at least one level must be full.
   *
   * [KLL] "Optimal Quantile Approximation in Streams", Karnin, Lang, and Liberty
   * https://github.com/DataSketches/sketches-core/tree/master/src/main/java/com/yahoo/sketches/kll
   */

  var n: Long = 0
  var initLevelsCapacity = QuantilesAggregator.findInitialLevelsCapacity(k, m)
  var combiner: ApproxCDFCombiner[T] = ApproxCDFCombiner[T](
      initLevelsCapacity,
      QuantilesAggregator.computeTotalCapacity(initLevelsCapacity, k, m),
      relError,
      helper.dummyValue)
  private[aggregators] var capacities: Array[Int] = QuantilesAggregator.capacities(k, m)

  def levels: Array[Int] = combiner.levels
  def items: Array[T] = combiner.items
  def numLevels = combiner.numLevels
  def levelsCapacity = combiner.maxNumLevels

  private[aggregators] def capacity: Int = combiner.capacity

  def seqOp(region: Region, x: T, missing: Boolean) {
    if (!missing) _seqOp(x)
  }

  private[aggregators] def _seqOp(x: T) {
    if (combiner.isFull) {
      if (eager)
        compactEager(helper.dummyValue)
      else
        compact(helper.dummyValue)
    }

    n += 1
    combiner.push(x)
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

  private[aggregators] def makeCdf(): (IndexedSeq[T], IndexedSeq[Long]) = {
    val (values, ranks) = combiner.cdf

    assert(ranks.last == n)

    (values, ranks)
  }

  def result(rvb: RegionValueBuilder): Unit = {
    val cdf = makeCdf()
    val res = Row(cdf._1, cdf._2, combiner.compactionCounts.toFastIndexedSeq)
    rvb.addAnnotation(QuantilesAggregator.resultType(helper.hailType), res)
  }

  def clear() {
    n = 0
    combiner.clear()
  }

  private def findFullLevel(): Int = {
    var level: Int = 0
    while (levels(level + 1) - levels(level) < levelCapacity(level)) {
      level += 1
    }
    level
  }

  private def levelCapacity(level: Int, numLevels: Int = numLevels): Int = {
    val depth = numLevels - level - 1
    if (depth < capacities.length) capacities(depth) else m
  }

  /* Compact the first over-capacity level. If that is the top level, grow the
   * sketch.
   */
  private def compact(dummy: T) {
    assert(combiner.isFull)
    val level = findFullLevel()
    if (level == numLevels - 1) growSketch()

    combiner.compactLevel(level)
  }

  /* If we are following the eager compacting strategy, level 0 must be full
   * when starting a compaction. This strategy sacrifices some accuracy, but
   * avoids having to shift up items below the compacted level.
   */
  private def compactEager(dummy: T) {
    assert(combiner.levelSize(0) >= levelCapacity(0))

    var level = 0
    var desiredFreeCapacity = 0
    var grew = false
    do {
      assert(combiner.levelSize(level) >= levelCapacity(level))

      if (level == numLevels - 1) {
        growSketch()
        assert(combiner.capacity >= computeTotalCapacity(numLevels + 1))
        grew = true
      }
      combiner.compactLevel(level)
      desiredFreeCapacity += levelCapacity(level)
      level += 1
    } while (levels(level) < desiredFreeCapacity && !grew)
  }

  private def growSketch() {
    if (combiner.numLevels == combiner.maxNumLevels)
      combiner = combiner.grow(
        combiner.maxNumLevels + growthRate,
        combiner.capacity + m * growthRate)
  }

  private def merge(other: RegionValueApproxCDFAggregator[T]) {
    val finalN = n + other.n
    val ub = QuantilesAggregator.ubOnNumLevels(finalN)

    val mergedCombiner = combiner.merge(other.combiner, ub)
    mergedCombiner.generalCompact(m, levelCapacity)

    val finalNumLevels = mergedCombiner.numLevels
    if (finalNumLevels > levelsCapacity)
      combiner = ApproxCDFCombiner[T](finalNumLevels, computeTotalCapacity(finalNumLevels), relError, helper.dummyValue)

    combiner.copyFrom(mergedCombiner)
    n = finalN
  }

  private def computeTotalCapacity(numLevels: Int): Int =
    QuantilesAggregator.computeTotalCapacity(numLevels, k, m)
}

object QuantilesAggregator {
  def resultType(eltType: Type): Type =
    TStruct("values" -> TArray(eltType), "ranks" -> TArray(TInt64()), "_compaction_counts" -> TArray(TInt32()))

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

  def capacities(k: Int, m: Int): Array[Int] = {
    val buffer: ArrayBuilder[Int] = new ArrayBuilder()
    var depth = 0
    var capacity = depthCapacity(depth, k)
    while (capacity > m) {
      buffer += capacity
      depth += 1
      capacity = depthCapacity(depth, k)
    }
    buffer.result()
  }

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
