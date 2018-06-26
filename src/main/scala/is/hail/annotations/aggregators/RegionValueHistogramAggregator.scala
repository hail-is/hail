package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.stats.HistogramCombiner
import is.hail.utils._

object RegionValueHistogramAggregator {
  val typ: TStruct = HistogramCombiner.schema
}

class RegionValueHistogramAggregator(start: Double, end: Double, bins: Int) extends RegionValueAggregator {
  if (bins <= 0)
    fatal(s"""method `hist' expects `bins' argument to be > 0, but got $bins""")

  private val binSize = (end - start) / bins
  if (binSize <= 0)
    fatal(
      s"""invalid bin size from given arguments (start = $start, end = $end, bins = $bins)
           |  Method requires positive bin size [(end - start) / bins], but got ${ binSize.formatted("%.2f") }
                  """.stripMargin)

  private val indices = Array.tabulate(bins + 1)(i => start + i * binSize)

  private var combiner = new HistogramCombiner(indices)

  def seqOp(region: Region, x: Double, missing: Boolean) {
    if (!missing)
      combiner.merge(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    combiner.merge(agg2.asInstanceOf[RegionValueHistogramAggregator].combiner)
  }

  def result(rvb: RegionValueBuilder) {
    rvb.startStruct()
    rvb.startArray(combiner.indices.length)
    combiner.indices.foreach(rvb.addDouble _)
    rvb.endArray()
    rvb.startArray(combiner.frequency.length)
    combiner.frequency.foreach(rvb.addLong _)
    rvb.endArray()
    rvb.addLong(combiner.nSmaller)
    rvb.addLong(combiner.nLarger)
    rvb.endStruct()
  }

  def newInstance() = new RegionValueHistogramAggregator(start, end, bins)

  override def copy(): RegionValueHistogramAggregator = {
    val rva = new RegionValueHistogramAggregator(start, end, bins)
    rva.combiner = combiner.copy()
    rva
  }

  def clear() {
    combiner.clear()
  }
}
