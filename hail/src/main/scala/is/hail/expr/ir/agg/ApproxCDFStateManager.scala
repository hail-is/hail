package is.hail.expr.ir.agg

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{DoubleArrayBuilder, IntArrayBuilder, LongArrayBuilder}
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.{PCanonicalArray, PCanonicalStruct, PFloat64, PInt32}
import is.hail.utils._

object ApproxCDFHelper {
  def sort(a: Array[Double], begin: Int, end: Int): Unit = java.util.Arrays.sort(a, begin, end)

  def merge(
    left: Array[Double], lStart: Int, lEnd: Int,
    right: Array[Double], rStart: Int, rEnd: Int,
    out: Array[Double], outStart: Int
  ): Unit = {
    assert((left ne out) || (outStart <= lStart - (rEnd - rStart)) || (outStart >= lEnd))
    assert((right ne out) || (outStart <= rStart - (lEnd - lStart)) || (outStart >= rEnd))
    var i: Int = lStart
    var j: Int = rStart
    var o: Int = outStart
    while (i < lEnd && j < rEnd) {
      val c = java.lang.Double.compare(left(i), right(j))
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

  def reverse(buf: Array[Double], start: Int, end: Int): Unit = {
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

  def rotate(buf: Array[Double], start: Int, mid: Int, end: Int): Int = {
    if (start == mid) return end
    if (mid == end) return start
    reverse(buf, start, mid)
    reverse(buf, mid, end)
    reverse(buf, start, end)
    start + (end - mid)
  }

  def compactBuffer(
    buf: Array[Double], inStart: Int, inEnd: Int,
    out: Array[Double], outStart: Int,
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
    buf: Array[Double], inStart: Int, inEnd: Int,
    out: Array[Double], outEnd: Int,
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
  def apply(numLevels: Int, capacity: Int, rand: java.util.Random): ApproxCDFCombiner = new ApproxCDFCombiner(
    { val a = Array.ofDim[Int](numLevels + 1); java.util.Arrays.fill(a, capacity); a },
    Array.ofDim[Double](capacity),
    Array.ofDim[Int](numLevels),
    1,
    rand)

  def apply(numLevels: Int, capacity: Int): ApproxCDFCombiner =
    apply(numLevels, capacity, new java.util.Random())

  def deserializeFrom(ib: InputBuffer): ApproxCDFCombiner = {
    val levels = new Array[Int](ib.readInt())
    var i = 0
    while (i < levels.length) {
      levels(i) = ib.readInt()
      i += 1
    }
    val items = new Array[Double](ib.readInt())
    i = 0
    while (i < items.length) {
      items(i) = ib.readDouble()
      i += 1
    }
    val compactionCounts = new Array[Int](ib.readInt())
    i = 0
    while (i < compactionCounts.length) {
      compactionCounts(i) = ib.readInt()
      i += 1
    }
    val numLevels = ib.readInt()
    new ApproxCDFCombiner(levels, items, compactionCounts, numLevels, new java.util.Random())
  }
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
class ApproxCDFCombiner(
  val levels: Array[Int],
  val items: Array[Double],
  val compactionCounts: Array[Int],
  var numLevels: Int,
  val rand: java.util.Random
) extends Serializable {

  def serializeTo(ob: OutputBuffer): Unit = {
    var i = 0
    ob.writeInt(levels.length)
    while (i < levels.length) {
      ob.writeInt(levels(i))
      i += 1
    }

    ob.writeInt(items.length)
    i = 0
    while (i < items.length) {
      ob.writeDouble(items(i))
      i += 1
    }

    ob.writeInt(compactionCounts.length)
    i = 0
    while (i < compactionCounts.length) {
      ob.writeInt(compactionCounts(i))
      i += 1
    }

    ob.writeInt(numLevels)
  }

  def copy(): ApproxCDFCombiner =
    new ApproxCDFCombiner(levels.clone(), items.clone(), compactionCounts.clone(), numLevels, rand)

  def maxNumLevels = levels.length - 1

  def capacity = items.length

  def n: Int = {
    var n = 0
    var i = 0
    while (i < numLevels) {
      n += (levels(i + 1) - levels(i)) << i
      i += 1
    }
    n
  }

  def isFull = levels(0) == 0

  def size = levels(maxNumLevels) - levels(0)

  def levelSize(level: Int): Int = levels(level + 1) - levels(level)

  def safeLevelSize(level: Int): Int =
    if (level >= maxNumLevels) 0 else levels(level + 1) - levels(level)

  def push(t: Double) {
    val bot = levels(0)

    val newBot = bot - 1
    items(newBot) = t
    levels(0) = newBot
  }

  def grow(newNumLevels: Int, newCapacity: Int): ApproxCDFCombiner = {
    require(newNumLevels > maxNumLevels && newCapacity > capacity)
    val newLevels = Array.ofDim[Int](newNumLevels + 1)
    val newItems = Array.ofDim[Double](newCapacity)
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

    new ApproxCDFCombiner(newLevels, newItems, newCompactionCounts, numLevels, rand)
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
  def compactLevel(level: Int, shiftLowerLevels: Boolean = true): Int = {
    val keep = if (level == 0) 1 else 0

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
  def _compactLevel(level: Int, keep: Int, dummy: Double = 0): Int = {
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

    if (level == 0) java.util.Arrays.sort(items, a0, c)
    val a = ApproxCDFHelper.rotate(items, a0 + keep + adj, c - keep, c)
    val halfSize = (c - a) / 2
    val b = a + halfSize

    if (levelAboveSize == 0) {
      ApproxCDFHelper.compactBufferBackwards(items, a, c, items, c, rand.nextBoolean())
    } else {
      ApproxCDFHelper.compactBuffer(items, a, c, items, a, rand.nextBoolean())
      ApproxCDFHelper.merge(items, a, b, items, c, d, items, b)
    }
    levels(level + 1) = b

    a
  }

  def merge(other: ApproxCDFCombiner, ubOnNumLevels: Int): ApproxCDFCombiner = {
    val mergedLevels = Array.ofDim[Int](ubOnNumLevels + 1)
    val mergedItems = Array.ofDim[Double](size + other.size)

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
        ApproxCDFHelper.merge(
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
    while (compactionCounts(lvl) > 0) {
      mergedCompactionCounts(lvl) += compactionCounts(lvl)
      lvl += 1
    }
    lvl = 0
    while (other.compactionCounts(lvl) > 0) {
      mergedCompactionCounts(lvl) += other.compactionCounts(lvl)
      lvl += 1
    }

    new ApproxCDFCombiner(
      mergedLevels,
      mergedItems,
      mergedCompactionCounts,
      math.max(numLevels, other.numLevels),
      rand)
  }

  def generalCompact(minCapacity: Int, levelCapacity: (Int, Int) => Int) {
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

  def copyFrom(other: ApproxCDFCombiner) {
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

  def computeCDF(): (Array[Double], Array[Long]) = {
    val builder: BoxedArrayBuilder[(Long, Double)] = new BoxedArrayBuilder(size)

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

    val values = new DoubleArrayBuilder(16)
    val ranks = new LongArrayBuilder(16)
    var rank: Long = 0
    var i = 0
    ranks.add(0)
    while (i < sorted.length) {
      rank += sorted(i)._1
      if (i == sorted.length - 1 || sorted(i)._2 != sorted(i + 1)._2) {
        values.add(sorted(i)._2)
        ranks.add(rank)
      }
      i += 1
    }

    (values.result(), ranks.result())
  }
}

object ApproxCDFStateManager {
  val defaultM: Int = 8

  def apply(k: Int): ApproxCDFStateManager = {
    val m: Int = defaultM
    val initLevelsCapacity: Int = QuantilesAggregator.findInitialLevelsCapacity(k, m)
    val combiner: ApproxCDFCombiner = ApproxCDFCombiner(
      initLevelsCapacity,
      QuantilesAggregator.computeTotalCapacity(initLevelsCapacity, k, m))
    new ApproxCDFStateManager(k, combiner)
  }

  def deserializeFrom(k: Int, ib: InputBuffer): ApproxCDFStateManager = {
    val a = ApproxCDFStateManager(k)
    a.combiner = ApproxCDFCombiner.deserializeFrom(ib)
    a
  }

  def fromData(k: Int, levels: Array[Int], items: Array[Double], compactionCounts: Array[Int]): ApproxCDFStateManager = {
    val combiner: ApproxCDFCombiner = new ApproxCDFCombiner(
      levels, items, compactionCounts, levels.length - 1, new java.util.Random)
    new ApproxCDFStateManager(k, combiner)
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
class ApproxCDFStateManager(val k: Int, var combiner: ApproxCDFCombiner) {
  val m: Int = ApproxCDFStateManager.defaultM
  private val growthRate: Int = 4
  private val eager: Boolean = false
  private val capacities: Array[Int] = QuantilesAggregator.capacities(k, m)

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

  def levels: Array[Int] = combiner.levels

  def items: Array[Double] = combiner.items

  def compactionCounts: Array[Int] = combiner.compactionCounts

  def numLevels = combiner.numLevels

  def levelsCapacity = combiner.maxNumLevels

  def n: Int = combiner.n

  private[agg] def capacity: Int = combiner.capacity

  def seqOp(x: Double): Unit = {
    if (combiner.isFull) {
      if (eager)
        compactEager()
      else
        compact()
    }

    combiner.push(x)
  }

  def combOp(other: ApproxCDFStateManager) {
    assert(m == other.m)
    if (other.numLevels == 1) {
      var i = other.levels(0)
      while (i < other.levels(1)) {
        seqOp(other.items(i))
        i += 1
      }
    } else {
      merge(other)
    }
  }

  def result(rvb: RegionValueBuilder): Unit = {
    val counts = combiner.compactionCounts
    rvb.startBaseStruct()

    val numItems = levels(numLevels) - levels(0)
    val offset = levels(0)

    rvb.startArray(numLevels + 1)
    var i = 0
    while (i <= numLevels) {
      rvb.addInt(levels(i) - offset)
      i += 1
    }
    rvb.endArray()

    rvb.startArray(numItems)
    i = levels(0)
    while (i < levels(numLevels)) {
      rvb.addDouble(items(i))
      i += 1
    }
    rvb.endArray()

    rvb.startArray(numLevels)
    i = 0
    while (i < numLevels) {
      rvb.addInt(counts(i))
      i += 1
    }
    rvb.endArray()

    rvb.endBaseStruct()
  }

  def rvResult(r: Region): Long = {
    val rvb = new RegionValueBuilder(HailStateManager(Map.empty), r)
    rvb.start(QuantilesAggregator.resultPType)
    result(rvb)
    rvb.end()
  }

  def clear() {
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
  private def compact() {
    assert(combiner.isFull)
    val level = findFullLevel()
    if (level == numLevels - 1) growSketch()

    combiner.compactLevel(level)
  }

  /* If we are following the eager compacting strategy, level 0 must be full
   * when starting a compaction. This strategy sacrifices some accuracy, but
   * avoids having to shift up items below the compacted level.
   */
  private def compactEager() {
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

  private def merge(other: ApproxCDFStateManager) {
    val finalN = n + other.n
    val ub = QuantilesAggregator.ubOnNumLevels(finalN)

    val mergedCombiner = combiner.merge(other.combiner, ub)
    mergedCombiner.generalCompact(m, levelCapacity)

    val finalNumLevels = mergedCombiner.numLevels
    if (finalNumLevels > levelsCapacity)
      combiner = ApproxCDFCombiner(finalNumLevels, computeTotalCapacity(finalNumLevels))

    combiner.copyFrom(mergedCombiner)
  }

  private def computeTotalCapacity(numLevels: Int): Int =
    QuantilesAggregator.computeTotalCapacity(numLevels, k, m)

  def serializeTo(ob: OutputBuffer): Unit = {
    combiner.serializeTo(ob)
  }
}

object QuantilesAggregator {
  val resultPType: PCanonicalStruct =
    PCanonicalStruct(required = false,
      "levels" -> PCanonicalArray(PInt32(true), required = true),
      "items" -> PCanonicalArray(PFloat64(true), required = true),
      "_compaction_counts" -> PCanonicalArray(PInt32(true), required = true))

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
    val buffer: IntArrayBuilder = new IntArrayBuilder(8)
    var depth = 0
    var capacity = depthCapacity(depth, k)
    while (capacity > m) {
      buffer.add(capacity)
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
