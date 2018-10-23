package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}

import scala.collection.generic.Growable
import scala.collection.mutable.ArrayBuffer
import is.hail.expr.types._
import is.hail.expr.types.physical.{PBaseStruct, PStruct, PType}
import is.hail.rvd.RVDContext
import sun.reflect.generics.reflectiveObjects.NotImplementedException

object WritableRegionValue {
  def apply(t: PType, initial: RegionValue, region: Region): WritableRegionValue =
    WritableRegionValue(t, initial.region, initial.offset, region)

  def apply(t: PType, initialRegion: Region, initialOffset: Long, targetRegion: Region): WritableRegionValue = {
    val wrv = WritableRegionValue(t, targetRegion)
    wrv.set(initialRegion, initialOffset)
    wrv
  }

  def apply(t: PType, region: Region): WritableRegionValue = {
    new WritableRegionValue(t, region)
  }
}

class WritableRegionValue private (
  val t: PType,
  val region: Region
) extends UnKryoSerializable {
  val value = RegionValue(region, 0)
  private val rvb: RegionValueBuilder = new RegionValueBuilder(region)

  def offset: Long = value.offset

  def setSelect(fromT: PStruct, fromFieldIdx: Array[Int], fromRV: RegionValue) {
    (t: @unchecked) match {
      case t: PStruct =>
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

  def pretty: String = value.pretty(t.virtualType)

  private def writeObject(s: ObjectOutputStream): Unit = {
    throw new NotImplementedException()
  }

  private def readObject(s: ObjectInputStream): Unit = {
    throw new NotImplementedException()
  }
}

class RegionValueArrayBuffer(val t: PType, region: Region)
  extends Iterable[RegionValue] with Growable[RegionValue] {

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
    fromT: PStruct,
    fromFieldIdx: Array[Int],
    fromRV: RegionValue): this.type = {

    (t: @unchecked) match {
      case t: PStruct =>
        rvb.start(t)
        rvb.selectRegionValue(fromT, fromFieldIdx, fromRV)
        idx += rvb.end()
    }
    this
  }

  def clear() {
    region.clear()
    idx.clear()
    rvb.clear() // remove
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
