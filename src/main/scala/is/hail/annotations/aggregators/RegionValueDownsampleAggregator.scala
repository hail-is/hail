package is.hail.annotations.aggregators
import is.hail.annotations.RegionValueBuilder
import is.hail.annotations._
import is.hail.expr.types.{TArray, TFloat64, TTuple, TString}
import is.hail.stats.DownsampleCombiner

object RegionValueDownsampleAggregator {
  val typ = TArray(TTuple(Array(TFloat64(), TFloat64(), TArray(TString()))))
}

class RegionValueDownsampleAggregator(nDivisions: Int) extends RegionValueAggregator {
  require(nDivisions > 0)

  var combiner: DownsampleCombiner = new DownsampleCombiner(nDivisions)

  def seqOp(region: Region, x: Double, xm: Boolean, y: Double, ym: Boolean, l: Long, lm: Boolean): Unit = {
    if (!xm && !ym) {
      val label = if (lm) null else SafeRow.read(TArray(TString()), region, l).asInstanceOf[IndexedSeq[String]]
      combiner.merge(x, y, label)
    }
  }
  
  def combOp(agg2: RegionValueAggregator) = combiner.merge(agg2.asInstanceOf[RegionValueDownsampleAggregator].combiner)

  def result(rvb: RegionValueBuilder) {
    if (combiner.binToPoint == null) {
      combiner.binToPoint = Map.empty
    }
    rvb.startArray(combiner.binToPoint.size)
    combiner.binToPoint.foreach { case (_, (x, y, a)) =>
      rvb.startTuple()
      rvb.addDouble(x)
      rvb.addDouble(y)
      rvb.addAnnotation(TArray(TString()), a)
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
