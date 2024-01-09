package is.hail.annotations

import is.hail.backend.HailStateManager
import is.hail.rvd.RVDContext
import is.hail.types.physical.{PStruct, PType}

import java.io.{ObjectInputStream, ObjectOutputStream}
import scala.collection.generic.Growable
import scala.collection.mutable.{ArrayBuffer, PriorityQueue}

import sun.reflect.generics.reflectiveObjects.NotImplementedException

object WritableRegionValue {
  def apply(sm: HailStateManager, t: PType, initial: RegionValue, region: Region)
    : WritableRegionValue =
    WritableRegionValue(sm, t, initial.region, initial.offset, region)

  def apply(sm: HailStateManager, t: PType, initialOffset: Long, targetRegion: Region)
    : WritableRegionValue = {
    val wrv = WritableRegionValue(sm, t, targetRegion)
    wrv.set(initialOffset, deepCopy = true)
    wrv
  }

  def apply(
    sm: HailStateManager,
    t: PType,
    initialRegion: Region,
    initialOffset: Long,
    targetRegion: Region,
  ): WritableRegionValue = {
    val wrv = WritableRegionValue(sm, t, targetRegion)
    wrv.set(initialRegion, initialOffset)
    wrv
  }

  def apply(sm: HailStateManager, t: PType, region: Region): WritableRegionValue =
    new WritableRegionValue(t, region, sm)
}

class WritableRegionValue private (
  val t: PType,
  val region: Region,
  sm: HailStateManager,
) extends UnKryoSerializable {
  val value = RegionValue(region, 0)
  private val rvb: RegionValueBuilder = new RegionValueBuilder(sm, region)

  def offset: Long = value.offset

  def setSelect(fromT: PStruct, fromFieldIdx: Array[Int], fromRV: RegionValue) {
    setSelect(fromT, fromFieldIdx, fromRV.region, fromRV.offset)
  }

  def setSelect(fromT: PStruct, fromFieldIdx: Array[Int], fromRegion: Region, fromOffset: Long) {
    setSelect(fromT, fromFieldIdx, fromOffset, region.ne(fromRegion))
  }

  def setSelect(fromT: PStruct, fromFieldIdx: Array[Int], fromOffset: Long, deepCopy: Boolean) {
    (t: @unchecked) match {
      case t: PStruct =>
        region.clear()
        rvb.start(t)
        rvb.startStruct()
        var i = 0
        while (i < t.size) {
          rvb.addField(fromT, fromOffset, fromFieldIdx(i), deepCopy)
          i += 1
        }
        rvb.endStruct()
        value.setOffset(rvb.end())
    }
  }

  def set(rv: RegionValue): Unit = set(rv.region, rv.offset)

  def set(fromRegion: Region, fromOffset: Long) {
    set(fromOffset, region.ne(fromRegion))
  }

  def set(fromOffset: Long, deepCopy: Boolean) {
    region.clear()
    rvb.start(t)
    rvb.addRegionValue(t, fromOffset, deepCopy)
    value.setOffset(rvb.end())
  }

  def pretty: String = value.pretty(t)

  private def writeObject(s: ObjectOutputStream): Unit =
    throw new NotImplementedException()

  private def readObject(s: ObjectInputStream): Unit =
    throw new NotImplementedException()
}

class RegionValuePriorityQueue(
  sm: HailStateManager,
  val t: PType,
  ctx: RVDContext,
  ord: Ordering[RegionValue],
) extends Iterable[RegionValue] {
  private val queue = new PriorityQueue[RegionValue]()(ord)
  private val rvb = new RegionValueBuilder(sm)

  override def nonEmpty: Boolean = queue.nonEmpty

  def empty: Boolean = queue.nonEmpty

  override def head: RegionValue = queue.head

  def enqueue(rv: RegionValue) {
    val region = ctx.freshRegion()
    rvb.set(region)
    rvb.start(t)
    rvb.addRegionValue(t, rv)
    queue.enqueue(RegionValue(region, rvb.end()))
  }

  def +=(rv: RegionValue): this.type = {
    enqueue(rv)
    this
  }

  def dequeue() {
    val popped = queue.dequeue()
    popped.region.close()
  }

  def iterator: Iterator[RegionValue] = queue.iterator
}

class RegionValueArrayBuffer(val t: PType, region: Region, sm: HailStateManager)
    extends Iterable[RegionValue] with Growable[RegionValue] {

  val value = RegionValue(region, 0)

  private val rvb = new RegionValueBuilder(sm, region)
  val idx = ArrayBuffer.empty[Long]

  def length = idx.length

  def +=(rv: RegionValue): this.type =
    this.append(rv.region, rv.offset)

  def append(fromRegion: Region, fromOffset: Long): this.type = {
    rvb.start(t)
    rvb.addRegionValue(t, fromRegion, fromOffset)
    idx += rvb.end()
    this
  }

  def appendSelect(
    fromT: PStruct,
    fromFieldIdx: Array[Int],
    fromRV: RegionValue,
  ): this.type = {

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
