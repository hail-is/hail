package is.hail.annotations

import is.hail.expr.Type

object RegionValue {
  def apply(): RegionValue = new RegionValue(null, 0)

  def apply(region: Region): RegionValue = new RegionValue(region, 0)

  def apply(region: Region, offset: Long) = new RegionValue(region, offset)
}

final class RegionValue(var region: Region,
  var offset: Long) extends Serializable {
  def set(newRegion: Region, newOffset: Long) {
    region = newRegion
    offset = newOffset
  }

  def setRegion(newRegion: Region) {
    region = newRegion
  }

  def setOffset(newOffset: Long) {
    offset = newOffset
  }

  def pretty(t: Type): String = region.pretty(t, offset)

  def copy(): RegionValue = RegionValue(region.copy(), offset)
}