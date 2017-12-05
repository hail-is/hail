package is.hail.rvd

import is.hail.annotations.{RegionValue, WritableRegionValue}
import is.hail.expr.Type

case class OrderedRVPartitionInfo(
  partitionIndex: Int,
  size: Int,
  min: RegionValue,
  max: RegionValue,
  // min, max: RegionValue[pkType]
  samples: Array[RegionValue],
  sortedness: Int) {
  def pretty(t: Type): String = {
    s"partitionIndex=$partitionIndex,size=$size,min=${min.pretty(t)},max=${max.pretty(t)},samples=${samples.map(_.pretty(t)).mkString(",")},sortedness=$sortedness"
  }
}

object OrderedRVPartitionInfo {
  final val UNSORTED = 0
  final val TSORTED = 1
  final val KSORTED = 2

  def apply(typ: OrderedRVType, sampleSize: Int, partitionIndex: Int, it: Iterator[RegionValue], seed: Int): OrderedRVPartitionInfo = {
    val minF = WritableRegionValue(typ.pkType)
    val maxF = WritableRegionValue(typ.pkType)
    val prevF = WritableRegionValue(typ.kType)

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
      samples(0) = WritableRegionValue(typ.pkType, f0)
      i += 1
    }

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
        samples(i) = WritableRegionValue(typ.pkType, f)
      else {
        val j = rng.nextInt(i)
        if (j < sampleSize)
          samples(j).set(f)
      }

      i += 1
    }

    OrderedRVPartitionInfo(partitionIndex, i,
      minF.value, maxF.value,
      Array.tabulate[RegionValue](math.min(i, sampleSize))(i => samples(i).value),
      sortedness)
  }
}