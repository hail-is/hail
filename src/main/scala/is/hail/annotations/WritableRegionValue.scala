package is.hail.annotations

import scala.collection.generic.Growable
import scala.collection.mutable.ArrayBuffer

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
    new WritableRegionValue(t)
  }
}

class WritableRegionValue private (val t: Type) {
  val region = Region()
  val value = RegionValue(region, 0)
  private val rvb: RegionValueBuilder = new RegionValueBuilder(region)

  def offset: Long = value.offset

  def setSelect(fromT: TStruct, fromFieldIdx: Array[Int], fromRV: RegionValue) {
    (t: @unchecked) match {
      case t: TStruct =>
        region.clear()
        rvb.start(t)
        rvb.startStruct()
        var i = 0
        while (i < t.size) {
          rvb.addField(fromT, fromRV, fromFieldIdx(i))
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

class RegionValueArrayBuffer(val t: Type)
  extends Iterable[RegionValue] with Growable[RegionValue] {

  val region = Region()
  val value = RegionValue(region, 0)

  private val rvb = new RegionValueBuilder(region)
  val idx = ArrayBuffer.empty[Long]

  def length = idx.length

  def +=(rv: RegionValue): this.type = {
    this.append(rv.region, rv.offset)
  }

  def append(fromRegion: Region, fromOffset: Long): this.type = {
    rvb.start(t)
    rvb.addRegionValue(t, fromRegion, fromOffset)
    idx += rvb.end()
    this
  }

  def appendSelect(
    fromT: TStruct,
    fromFieldIdx: Array[Int],
    fromRV: RegionValue): this.type = {

    (t: @unchecked) match {
      case t: TStruct =>
        rvb.start(t)
        rvb.selectRegionValue(fromT, fromFieldIdx, fromRV)
        idx += rvb.end()
    }
    this
  }

  def clear() {
    region.clear()
    idx.clear()
    rvb.clear()
  }

  private var itIdx = 0
  private val it = new Iterator[RegionValue] {
    def next(): RegionValue = {
      value.setOffset(idx(itIdx))
      itIdx += 1
      value
    }
    def hasNext: Boolean = itIdx < idx.size
  }

  def iterator: Iterator[RegionValue] = {
    itIdx = 0
    it
  }
}
