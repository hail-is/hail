package is.hail.annotations

import is.hail.expr.types.Type

object JoinedRegionValue {
  def apply(): JoinedRegionValue = new JoinedRegionValue(null, null)

  def apply(left: RegionValue, right: RegionValue): JoinedRegionValue = new JoinedRegionValue(left, right)
  implicit class RichJoinedRegionValue(jrv: JoinedRegionValue) {
    def pretty(lTyp: Type, rTyp: Type): String = jrv._1.pretty(lTyp) + "," + jrv._2.pretty(rTyp)
  }
}


