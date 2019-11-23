package is.hail.variant

import is.hail.annotations._

trait View {
  final def setRegion(rv: RegionValue) {
    setRegion(rv.offset)
  }

  def setRegion(offset: Long)
}
