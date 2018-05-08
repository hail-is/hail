package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.types.{TArray, TString, TStruct}
import is.hail.stats.{CallStats, CallStatsCombiner}
import is.hail.utils.ArrayBuilder
import is.hail.variant.Call

object RegionValueCallStatsAggregator {
  val allelesTyp = TArray(TString())
  val typ: TStruct = CallStats.schema
}

class RegionValueCallStatsAggregator extends RegionValueAggregator {
  private var combiner: CallStatsCombiner = _
  private var alleles: Array[String] = _

  def initOp(region: Region, offset: Long, missing: Boolean): Unit = {
    if (!missing) {
      val l = RegionValueCallStatsAggregator.allelesTyp.loadLength(region, offset)
      alleles = new Array[String](l)
      var i = 0
      while (i < l) {
        assert(RegionValueCallStatsAggregator.allelesTyp.isElementDefined(region, offset, i))
        alleles(i) = TString.loadString(region, RegionValueCallStatsAggregator.allelesTyp.loadElement(region, offset, i))
        i += 1
      }
      combiner = new CallStatsCombiner(alleles)
    }
  }

  def seqOp(x: Call, missing: Boolean): Unit = {
    if (combiner != null && !missing)
      combiner.merge(x)
  }

  def combOp(agg2: RegionValueAggregator): Unit = {
    val other = agg2.asInstanceOf[RegionValueCallStatsAggregator]
    if (other.combiner != null) {
      if (combiner == null)
        combiner = new CallStatsCombiner(other.combiner.alleles)
    }
    combiner.merge(other.combiner)
  }

  def result(rvb: RegionValueBuilder): Unit = {
    if (combiner != null) {
      val r = combiner.result()
      rvb.startStruct()
      rvb.startArray(r.alleleCount.length)
      r.alleleCount.foreach(rvb.addInt _)
      rvb.endArray()
      r.alleleFrequency match {
        case Some(af) =>
          rvb.startArray(af.length)
          af.foreach(rvb.addDouble _)
          rvb.endArray()
        case None =>
          rvb.setMissing()
      }
      rvb.addInt(r.alleleNumber)
      rvb.endStruct()
    } else
      rvb.setMissing()
  }

  def copy() = new RegionValueCallStatsAggregator()

  def clear() {
    combiner = null
    alleles = null
  }
}
