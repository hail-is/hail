package is.hail.annotations

import is.hail.expr.types._

object WritableRegionValue {
  def apply(t: Type, initial: RegionValue): WritableRegionValue =
    WritableRegionValue(t, initial.region, initial.offset)

  def apply(t: Type, initialRegion: Region, initialOffset: Long): WritableRegionValue = {
    val wrv = WritableRegionValue(t)
    wrv.set(initialRegion, initialOffset)
    wrv
  }

  def apply(t: Type): WritableRegionValue = {
    val region = Region()
    new WritableRegionValue(t, region, new RegionValueBuilder(region), RegionValue(region, 0))
  }
}

class WritableRegionValue(val t: Type,
  val region: Region,
  rvb: RegionValueBuilder,
  val value: RegionValue) {

  def offset: Long = value.offset

  def setSelect(fromT: TStruct, toFromFieldIdx: Array[Int], fromRV: RegionValue) {
    (t: @unchecked) match {
      case t: TStruct =>
        region.clear()
        rvb.start(t)
        rvb.startStruct()
        var i = 0
        while (i < t.size) {
          rvb.addField(fromT, fromRV, toFromFieldIdx(i))
          i += 1
        }
        rvb.endStruct()
        value.setOffset(rvb.end())
    }
  }

  def set(rv: RegionValue): Unit = set(rv.region, rv.offset)

  def set(fromRegion: Region, fromOffset: Long) {
    region.clear()
    rvb.start(t)
    rvb.addRegionValue(t, fromRegion, fromOffset)
    value.setOffset(rvb.end())
  }

  def pretty: String = value.pretty(t)
}
