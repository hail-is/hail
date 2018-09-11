package is.hail.utils.richUtils

import is.hail.annotations.{JoinedRegionValue, RegionValue}
import is.hail.expr.types.Type

class RichJoinedRegionValue(jrv: JoinedRegionValue) {
  def pretty(lTyp: Type, rTyp: Type): String = jrv._1.pretty(lTyp) + "," + jrv._2.pretty(rTyp)
  def rvLeft: RegionValue = jrv._1
  def rvRight: RegionValue = jrv._2
}
