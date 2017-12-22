package is.hail.annotations

import is.hail.expr.typ.Type

object JoinedRegionValue {
  def apply(): JoinedRegionValue = new JoinedRegionValue(null, null)

  def apply(left: RegionValue, right: RegionValue): JoinedRegionValue = new JoinedRegionValue(left, right)
}

final class JoinedRegionValue(var rvLeft: RegionValue, var rvRight: RegionValue) extends Serializable {
  def set(left: RegionValue, right: RegionValue) {
    rvLeft = left
    rvRight = right
  }

  def pretty(lTyp: Type, rTyp: Type): String = rvLeft.pretty(lTyp) + "," + rvRight.pretty(rTyp)
}