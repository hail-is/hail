package is.hail.annotations

import is.hail.expr.types.Type

object JoinedRegionValue {
  def apply(): JoinedRegionValue = new JoinedRegionValue(null, null)

  def apply(left: RegionValue, right: RegionValue): JoinedRegionValue = new JoinedRegionValue(left, right)
}
