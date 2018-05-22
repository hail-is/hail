package is.hail.rvd

import is.hail.utils._
import is.hail.annotations.{RegionValue, SafeRow, WritableRegionValue}
import is.hail.expr.types.Type

case class OrderedRVPartitionInfo(
  partitionIndex: Int,
  size: Int,
  min: Any,
  max: Any,
  // min, max: RegionValue[pkType]
  samples: Array[Any],
  sortedness: Int) {
  def pretty(t: Type): String = {
    s"partitionIndex=$partitionIndex,size=$size,min=$min,max=$max,samples=${samples.mkString(",")},sortedness=$sortedness"
  }
}

object OrderedRVPartitionInfo {
  final val UNSORTED = 0
  final val TSORTED = 1
  final val KSORTED = 2

  def apply(
    typ: OrderedRVDType,
    sampleSize: Int,
    partitionIndex: Int,
    it: Iterator[RegionValue],
    seed: Int,
    producerContext: RVDContext
  ): OrderedRVPartitionInfo = {
    using(RVDContext.default) { localctx =>
      val minF = WritableRegionValue(typ.pkType, localctx.freshRegion)
      val maxF = WritableRegionValue(typ.pkType, localctx.freshRegion)
      val prevF = WritableRegionValue(typ.kType, localctx.freshRegion)

      assert(it.hasNext)
      val f0 = it.next()

      minF.setSelect(typ.kType, typ.pkKFieldIdx, f0)
      maxF.setSelect(typ.kType, typ.pkKFieldIdx, f0)
      prevF.set(f0)

      var sortedness = KSORTED

      val rng = new java.util.Random(seed)
      val samples = new Array[WritableRegionValue](sampleSize)

      var i = 0

      if (sampleSize > 0) {
        samples(0) = WritableRegionValue(typ.pkType, f0, localctx.freshRegion)
        i += 1
      }

      producerContext.region.clear()
      while (it.hasNext) {
        val f = it.next()

        if (typ.kOrd.compare(f, prevF.value) < 0) {
          if (typ.pkInKOrd.compare(f, prevF.value) < 0)
            sortedness = UNSORTED
          else
            sortedness = sortedness.min(TSORTED)
        }

        if (typ.pkKOrd.compare(minF.value, f) > 0)
          minF.setSelect(typ.kType, typ.pkKFieldIdx, f)
        if (typ.pkKOrd.compare(maxF.value, f) < 0)
          maxF.setSelect(typ.kType, typ.pkKFieldIdx, f)

        prevF.set(f)

        if (i < sampleSize)
          samples(i) = WritableRegionValue(typ.pkType, f, localctx.freshRegion)
        else {
          val j = if (i > 0) rng.nextInt(i) else 0
          if (j < sampleSize)
            samples(j).set(f)
        }

        producerContext.region.clear()
        i += 1
      }

      val safe: RegionValue => Any = SafeRow(typ.pkType, _)

      OrderedRVPartitionInfo(partitionIndex, i,
        safe(minF.value), safe(maxF.value),
        Array.tabulate[Any](math.min(i, sampleSize))(i => safe(samples(i).value)),
        sortedness)
    }
  }
}
