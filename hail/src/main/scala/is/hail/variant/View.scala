package is.hail.variant

import is.hail.annotations._

trait View {
  final def setRegion(rv: RegionValue) {
    setRegion(rv.region, rv.offset)
  }

  // FIXME: Remove region?
  def setRegion(region: Region, offset: Long)
}
