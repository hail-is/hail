package is.hail.annotations.aggregators
import is.hail.annotations.RegionValueBuilder
import is.hail.annotations._
import is.hail.expr.types.{TArray, TFloat64, TTuple}
import is.hail.stats.DownsampleCombiner

object RegionValueDownsampleAggregator {
  val typ = TArray(TTuple(Array(TFloat64(), TFloat64())))
}

class RegionValueDownsampleAggregator(nDivisions: Int) extends RegionValueAggregator {
  require(nDivisions > 0)

  var combiner: DownsampleCombiner = new DownsampleCombiner(nDivisions)

  def seqOp(r: Region, offset: Long, missing: Boolean) = {
    if (!missing) {
      val row = UnsafeRow.read(TTuple(Array(TFloat64(), TFloat64())), r, offset).asInstanceOf[UnsafeRow]
      combiner.merge(row.getDouble(0), row.getDouble(1))
    }
  }
  
  def combOp(agg2: RegionValueAggregator) = combiner.merge(agg2.asInstanceOf[RegionValueDownsampleAggregator].combiner)

  def result(rvb: RegionValueBuilder) {
    rvb.startArray(combiner.binsToCoords.size)
    combiner.binsToCoords.foreach { case (_, (x, y)) =>
      rvb.startTuple()
      rvb.addDouble(x)
      rvb.addDouble(y)
      rvb.endTuple()
    }
    rvb.endArray()
  }

  def newInstance(): RegionValueAggregator = new RegionValueDownsampleAggregator(nDivisions)

  def copy(): RegionValueDownsampleAggregator = {
    val rva = new RegionValueDownsampleAggregator(nDivisions)
    rva.combiner = combiner.copy()
    rva
  }

  def clear() = combiner.clear()
}
