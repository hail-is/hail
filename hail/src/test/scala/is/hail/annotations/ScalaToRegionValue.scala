package is.hail.annotations

import is.hail.expr.types._
import is.hail.expr.types.virtual.Type

object ScalaToRegionValue {
  def apply(region: Region, t: Type, a: Annotation): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(t.physicalType)
    rvb.addAnnotation(t, a)
    rvb.end()
  }
}
