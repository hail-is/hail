package is.hail.rvd

import is.hail.annotations.{RegionValue, SafeRow, WritableRegionValue}
import is.hail.expr.types.virtual.Type
import is.hail.utils._

case class RVDPartitionInfo(
  partitionIndex: Int,
  size: Int,
  min: Any,
  max: Any,
  // min, max: RegionValue[kType]
  samples: Array[Any],
  sortedness: Int,
  contextStr: String
) {
  val interval = Interval(min, max, true, true)

  def pretty(t: Type): String = {
    s"partitionIndex=$partitionIndex,size=$size,min=$min,max=$max,samples=${samples.mkString(",")},sortedness=$sortedness"
  }
}

object RVDPartitionInfo {
  final val UNSORTED = 0
  final val TSORTED = 1
  final val KSORTED = 2

  def apply(
    typ: RVDType,
    partitionKey: Int,
    sampleSize: Int,
    partitionIndex: Int,
    it: Iterator[RegionValue],
    seed: Int,
    producerContext: RVDContext
  ): RVDPartitionInfo = {
    using(RVDContext.default) { localctx =>
      val kPType = typ.kType
      val pkOrd = typ.copy(key = typ.key.take(partitionKey)).kOrd
      val minF = WritableRegionValue(kPType, localctx.freshRegion)
      val maxF = WritableRegionValue(kPType, localctx.freshRegion)
      val prevF = WritableRegionValue(kPType, localctx.freshRegion)

      assert(it.hasNext)
      val f0 = it.next()

      minF.set(f0)
      maxF.set(f0)
      prevF.set(f0)

      var sortedness = KSORTED
      var contextStr = ""

      val rng = new java.util.Random(seed)
      val samples = new Array[WritableRegionValue](sampleSize)

      var i = 0

      if (sampleSize > 0) {
        samples(0) = WritableRegionValue(kPType, f0, localctx.freshRegion)
        i += 1
      }

      producerContext.region.clear()
      while (it.hasNext) {
        val f = it.next()

        if (sortedness > UNSORTED && typ.kOrd.lt(f, prevF.value)) {
          if (pkOrd.lt(f, prevF.value)) {
            val curr = f.pretty(typ.kType)
            val prev = prevF.pretty
            log.info(s"unsorted: $curr, $prev")
            contextStr = s"CURRENT=$curr, PREV=$prev"
            sortedness = UNSORTED
          } else if (sortedness > TSORTED) {
            val curr = f.pretty(typ.kType)
            val prev = prevF.pretty
            log.info(s"partition-key-sorted: $curr, $prev")
            contextStr = s"CURRENT=$curr, PREV=$prev"
            sortedness = sortedness.min(TSORTED)
          }
        }

        if (typ.kOrd.lt(f, minF.value))
          minF.set(f)
        if (typ.kOrd.gt(f, maxF.value))
          maxF.set(f)

        prevF.set(f)

        if (i < sampleSize)
          samples(i) = WritableRegionValue(kPType, f, localctx.freshRegion)
        else {
          val j = if (i > 0) rng.nextInt(i) else 0
          if (j < sampleSize)
            samples(j).set(f)
        }

        producerContext.region.clear()
        i += 1
      }

      val safe: RegionValue => Any = SafeRow(kPType, _)

      RVDPartitionInfo(partitionIndex, i,
        safe(minF.value), safe(maxF.value),
        Array.tabulate[Any](math.min(i, sampleSize))(i => safe(samples(i).value)),
        sortedness,
        contextStr)
    }
  }
}
