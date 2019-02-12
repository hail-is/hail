package is.hail.annotations.aggregators

import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeRow}
import is.hail.expr.types.virtual.Type

class RegionValuePrevNonnullAnnotationAggregator(t: Type) extends RegionValueAggregator {
  var last: Annotation = null

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    if (!missing)
      last = SafeRow.read(t.physicalType, region, offset)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValuePrevNonnullAnnotationAggregator]
    if (that.last != null)
      last = that.last
  }

  def result(rvb: RegionValueBuilder) {
    if (last != null)
      rvb.addAnnotation(t, last)
    else
      rvb.setMissing()
  }

  def newInstance(): RegionValuePrevNonnullAnnotationAggregator = new RegionValuePrevNonnullAnnotationAggregator(t)

  def copy(): RegionValuePrevNonnullAnnotationAggregator = {
    val rva = new RegionValuePrevNonnullAnnotationAggregator(t)
    rva.last = last
    rva
  }

  def clear() {
    last = null
  }
}
