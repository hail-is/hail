package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.stats.HistogramCombiner
import is.hail.utils._

object RegionValueHistogramAggregator {
  val typ = HistogramCombiner.schema

  def stagedNew(start: Code[Double], mstart: Code[Boolean], end: Code[Double], mend: Code[Boolean], bins: Code[Int], mbins: Code[Boolean]): Code[RegionValueHistogramAggregator] =
    (mbins | mstart | mend).mux(
      Code._throw(Code.newInstance[RuntimeException, String]("Histogram aggregator cannot take NA arguments")),
      Code.newInstance[RegionValueHistogramAggregator, Double, Double, Int](start, end, bins))
}

class RegionValueHistogramAggregator(start: Double, end: Double, bins: Int) extends RegionValueAggregator {
  import RegionValueHistogramAggregator._

  if (bins <= 0)
    fatal(s"""method `hist' expects `bins' argument to be > 0, but got $bins""")

  private val binSize = (end - start) / bins
  if (binSize <= 0)
    fatal(
      s"""invalid bin size from given arguments (start = $start, end = $end, bins = $bins)
           |  Method requires positive bin size [(end - start) / bins], but got ${ binSize.formatted("%.2f") }
                  """.stripMargin)

  private val indices = Array.tabulate(bins + 1)(i => start + i * binSize)

  private val combiner = new HistogramCombiner(indices)

  def seqOp(x: Double, missing: Boolean) {
    if (!missing)
      combiner.merge(x)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadDouble(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    combiner.merge(agg2.asInstanceOf[RegionValueHistogramAggregator].combiner)
  }

  private val rvb = new RegionValueBuilder()

  def result(region: Region): Long = {
    rvb.set(region)
    rvb.start(typ)
    rvb.startArray(combiner.indices.length)
    combiner.indices.foreach(rvb.addDouble _)
    rvb.endArray()
    rvb.startArray(combiner.frequency.length)
    combiner.frequency.foreach(rvb.addLong _)
    rvb.endArray()
    rvb.addLong(combiner.nLess)
    rvb.addLong(combiner.nGreater)
    rvb.end()
  }

  def copy() = new RegionValueHistogramAggregator(start, end, bins)
}
