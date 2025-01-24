package is.hail.rvd

import is.hail.annotations.{Region, RegionValue, SafeRow, WritableRegionValue}
import is.hail.backend.HailStateManager
import is.hail.types.virtual.Type
import is.hail.utils._

import net.sourceforge.jdistlib.rng.MersenneTwister

case class RVDPartitionInfo(
  partitionIndex: Int,
  size: Long,
  min: Any,
  max: Any,
  // min, max: RegionValue[kType]
  samples: Array[Any],
  sortedness: Int,
  contextStr: String,
) {
  val interval = Interval(min, max, true, true)

  def pretty(t: Type): String =
    s"partitionIndex=$partitionIndex,size=$size,min=$min,max=$max,samples=${samples.mkString(",")},sortedness=$sortedness"
}

object RVDPartitionInfo {
  final val UNSORTED = 0
  final val TSORTED = 1
  final val KSORTED = 2

  def apply(
    sm: HailStateManager,
    typ: RVDType,
    partitionKey: Int,
    sampleSize: Int,
    partitionIndex: Int,
    it: Iterator[Long],
    seed: Long,
    producerContext: RVDContext,
  ): RVDPartitionInfo = {
    using(RVDContext.default(producerContext.r.pool)) { localctx =>
      val kPType = typ.kType
      val pkOrd = typ.copy(key = typ.key.take(partitionKey)).kOrd(sm)
      val minF = WritableRegionValue(sm, kPType, localctx.freshRegion())
      val maxF = WritableRegionValue(sm, kPType, localctx.freshRegion())
      val prevF = WritableRegionValue(sm, kPType, localctx.freshRegion())

      assert(it.hasNext)
      val f0 = it.next()

      minF.set(f0, deepCopy = true)
      maxF.set(f0, deepCopy = true)
      prevF.set(f0, deepCopy = true)

      var sortedness = KSORTED
      var contextStr = ""

      val rng = new MersenneTwister(seed)
      val samples = new Array[WritableRegionValue](sampleSize)

      var i: Long = 0

      if (sampleSize > 0) {
        samples(0) = WritableRegionValue(sm, kPType, f0, localctx.freshRegion())
        i += 1
      }

      producerContext.region.clear()
      while (it.hasNext) {
        val f = it.next()

        val kOrd = typ.kOrd(sm)
        if (sortedness > UNSORTED && kOrd.lt(f, prevF.value.offset)) {
          if (pkOrd.lt(f, prevF.value.offset)) {
            val curr = Region.pretty(typ.kType, f)
            val prev = prevF.pretty
            log.info(s"unsorted: $curr, $prev")
            contextStr = s"CURRENT=$curr, PREV=$prev"
            sortedness = UNSORTED
          } else if (sortedness > TSORTED) {
            val curr = Region.pretty(typ.kType, f)
            val prev = prevF.pretty
            log.info(s"partition-key-sorted: $curr, $prev")
            contextStr = s"CURRENT=$curr, PREV=$prev"
            sortedness = sortedness.min(TSORTED)
          }
        }

        if (kOrd.lt(f, minF.value.offset))
          minF.set(f, deepCopy = true)
        if (kOrd.gt(f, maxF.value.offset))
          maxF.set(f, deepCopy = true)

        prevF.set(f, deepCopy = true)

        if (i < sampleSize)
          samples(i.toInt) = WritableRegionValue(sm, kPType, f, localctx.freshRegion())
        else {
          val j: Long = if (i > 0) rng.nextLong(i) else 0
          if (j < sampleSize)
            samples(j.toInt).set(f, deepCopy = true)
        }

        producerContext.region.clear()
        i += 1
      }

      val safe: RegionValue => Any = SafeRow(kPType, _)

      RVDPartitionInfo(
        partitionIndex,
        i,
        safe(minF.value),
        safe(maxF.value),
        Array.tabulate[Any](math.min(i, sampleSize).toInt)(i => safe(samples(i).value)),
        sortedness,
        contextStr,
      )
    }
  }
}
