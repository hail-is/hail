package is.hail.sparkextras.implicits

import is.hail.annotations.{JoinedRegionValue, RegionValue}
import is.hail.types.physical.PType

class RichJoinedRegionValue(jrv: JoinedRegionValue) {
  def pretty(lTyp: PType, rTyp: PType): String = jrv._1.pretty(lTyp) + "," + jrv._2.pretty(rTyp)
  def rvLeft: RegionValue = jrv._1
  def rvRight: RegionValue = jrv._2
}
