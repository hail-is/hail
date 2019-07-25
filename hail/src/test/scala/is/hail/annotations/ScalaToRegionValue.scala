package is.hail.annotations

import is.hail.expr.types.physical.PType

object ScalaToRegionValue {
  def apply(region: Region, t: PType, a: Annotation): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.addAnnotation(t.virtualType, a)
    rvb.end()
  }
}
