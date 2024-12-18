package is.hail.annotations

object JoinedRegionValue {
  def apply(): JoinedRegionValue = new JoinedRegionValue(null, null)

  def apply(left: RegionValue, right: RegionValue): JoinedRegionValue =
    new JoinedRegionValue(left, right)
}
