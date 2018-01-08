package is.hail.annotations.aggregators

import is.hail.asm4s._
import is.hail.expr._
import is.hail.expr.ir.defaultValue
import is.hail.annotations._
import is.hail.expr.RegionValueAggregator

import scala.reflect.ClassTag

object RegionValueSumAggregator {
  private def seqOp[Agg <: RegionValueAggregator : ClassTag : TypeInfo, T : ClassTag](t: Type): (Code[RegionValueAggregator], Code[_], Code[Boolean]) => Code[Unit] =
  { (rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]) =>
    mv.mux(
      Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](defaultValue(t)), true),
      Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](v), false)) }

  def apply(t: Type): ((Code[RegionValueAggregator], Code[_], Code[Boolean]) => Code[Unit], RegionValueAggregator) = {
    t match {
      case _: TBoolean =>
        (seqOp[RegionValueSumBooleanAggregator, Boolean](t),  new RegionValueSumBooleanAggregator())
      case _: TInt32 =>
        (seqOp[RegionValueSumIntAggregator, Int](t),  new RegionValueSumIntAggregator())
      case _: TInt64 =>
        (seqOp[RegionValueSumLongAggregator, Long](t),  new RegionValueSumLongAggregator())
      case _: TFloat32 =>
        (seqOp[RegionValueSumFloatAggregator, Float](t),  new RegionValueSumFloatAggregator())
      case _: TFloat64 =>
        (seqOp[RegionValueSumDoubleAggregator, Double](t),  new RegionValueSumDoubleAggregator())
      case _ => throw new IllegalArgumentException(s"Cannot sum over values of type ${t}")
    }
  }
}

class RegionValueSumBooleanAggregator extends RegionValueAggregator {
  private var sum: Boolean = false

  def seqOp(b: Boolean, missing: Boolean) {
    if (!missing)
      sum |= b
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadBoolean(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum |= agg2.asInstanceOf[RegionValueSumBooleanAggregator].sum
  }

  def result(region: Region): Long =
    region.appendByte(if (sum) 1.toByte else 0.toByte)

  def copy(): RegionValueSumBooleanAggregator = new RegionValueSumBooleanAggregator()
}

class RegionValueSumIntAggregator extends RegionValueAggregator {
  private var sum: Int = 0

  def seqOp(i: Int, missing: Boolean) {
    if (!missing)
      sum += i
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadInt(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumIntAggregator].sum
  }

  def result(region: Region): Long =
    region.appendInt(sum)

  def copy(): RegionValueSumIntAggregator = new RegionValueSumIntAggregator()
}

class RegionValueSumLongAggregator extends RegionValueAggregator {
  private var sum: Long = 0L

  def seqOp(l: Long, missing: Boolean) {
    if (!missing)
      sum += l
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadLong(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumLongAggregator].sum
  }

  def result(region: Region): Long =
    region.appendLong(sum)

  def copy(): RegionValueSumLongAggregator = new RegionValueSumLongAggregator()
}

class RegionValueSumFloatAggregator extends RegionValueAggregator {
  private var sum: Float = 0.0f

  def seqOp(f: Float, missing: Boolean) {
    if (!missing)
      sum += f
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadFloat(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumFloatAggregator].sum
  }

  def result(region: Region): Long =
    region.appendFloat(sum)

  def copy(): RegionValueSumFloatAggregator = new RegionValueSumFloatAggregator()
}

class RegionValueSumDoubleAggregator extends RegionValueAggregator {
  private var sum: Double = 0.0

  def seqOp(d: Double, missing: Boolean) {
    println(s"$d $missing")
    if (!missing)
      sum += d
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadDouble(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumDoubleAggregator].sum
  }

  def result(region: Region): Long =
    region.appendDouble(sum)

  def copy(): RegionValueSumDoubleAggregator = new RegionValueSumDoubleAggregator()
}
