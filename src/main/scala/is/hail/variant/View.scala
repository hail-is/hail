package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

trait View {
  def setRegion(rv: RegionValue) {
    setRegion(rv.region, rv.offset)
  }

  def setRegion(region: MemoryBuffer, offset: Long)
}

