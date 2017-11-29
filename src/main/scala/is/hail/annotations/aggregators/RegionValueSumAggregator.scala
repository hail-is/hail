package is.hail.annotations.aggregators

import is.hail.expr._
import is.hail.annotations._
import is.hail.expr.RegionValueAggregator

object RegionValueSumAggregator {
  def apply(t: Type): RegionValueAggregator = t match {
    case _: TBoolean => new RegionValueSumBooleanAggregator()
    case _: TInt32 => new RegionValueSumIntAggregator()
    case _: TInt64 => new RegionValueSumLongAggregator()
    case _: TFloat32 => new RegionValueSumFloatAggregator()
    case _: TFloat64 => new RegionValueSumDoubleAggregator()
    case _ => throw new IllegalArgumentException(s"Cannot sum over values of type $t")
  }
}

class RegionValueSumBooleanAggregator extends RegionValueAggregator {
  private var sum: Boolean = false

  def typ: TBoolean = TBoolean(true)

  def seqOp(region: MemoryBuffer, off: Long, missing: Boolean) {
    if (!missing)
      sum |= region.loadBoolean(off)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum |= agg2.asInstanceOf[RegionValueSumBooleanAggregator].sum
  }

  def result(region: MemoryBuffer): Long =
    region.appendByte(if (sum) 1.toByte else 0.toByte)

  def copy(): RegionValueSumBooleanAggregator = new RegionValueSumBooleanAggregator()
}

class RegionValueSumIntAggregator extends RegionValueAggregator {
  private var sum: Int = 0

  def typ: TInt32 = TInt32(true)

  def seqOp(region: MemoryBuffer, off: Long, missing: Boolean) {
    if (!missing)
      sum += region.loadInt(off)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumIntAggregator].sum
  }

  def result(region: MemoryBuffer): Long =
    region.appendInt(sum)

  def copy(): RegionValueSumIntAggregator = new RegionValueSumIntAggregator()
}

class RegionValueSumLongAggregator extends RegionValueAggregator {
  private var sum: Long = 0L

  def typ: TInt64 = TInt64(true)

  def seqOp(region: MemoryBuffer, off: Long, missing: Boolean) {
    if (!missing)
      sum += region.loadLong(off)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumLongAggregator].sum
  }

  def result(region: MemoryBuffer): Long =
    region.appendLong(sum)

  def copy(): RegionValueSumLongAggregator = new RegionValueSumLongAggregator()
}

class RegionValueSumFloatAggregator extends RegionValueAggregator {
  private var sum: Float = 0.0f

  def typ: TFloat32 = TFloat32(true)

  def seqOp(region: MemoryBuffer, off: Long, missing: Boolean) {
    if (!missing)
      sum += region.loadFloat(off)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumFloatAggregator].sum
  }

  def result(region: MemoryBuffer): Long =
    region.appendFloat(sum)

  def copy(): RegionValueSumFloatAggregator = new RegionValueSumFloatAggregator()
}

class RegionValueSumDoubleAggregator extends RegionValueAggregator {
  private var sum: Double = 0.0

  def typ: TFloat64 = TFloat64(true)

  def seqOp(region: MemoryBuffer, off: Long, missing: Boolean) {
    if (!missing)
      sum += region.loadDouble(off)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumDoubleAggregator].sum
  }

  def result(region: MemoryBuffer): Long =
    region.appendDouble(sum)

  def copy(): RegionValueSumDoubleAggregator = new RegionValueSumDoubleAggregator()
}
