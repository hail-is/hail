package is.hail.annotations

import is.hail.rvd.OrderedRVDType
import is.hail.utils._

case class OrderedRVIterator(t: OrderedRVDType, iterator: Iterator[RegionValue]) {

  def staircase: StagingIterator[FlipbookIterator[RegionValue]] =
    iterator.toEphemeralIterator.staircased(t.kRowOrdView)

  def cogroup(other: OrderedRVIterator):
      FlipbookIterator[Muple[FlipbookIterator[RegionValue], FlipbookIterator[RegionValue]]] =
    this.iterator.toEphemeralIterator.cogroup(
      other.iterator.toEphemeralIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      this.t.kComp(other.t).compare
    )

  def zipJoin(other: OrderedRVIterator): FlipbookIterator[JoinedRegionValue] =
    iterator.toEphemeralIterator.orderedZipJoin(
      other.iterator.toEphemeralIterator,
      leftDefault = null,
      rightDefault = null,
      OrderedRVDType.selectUnsafeOrdering(
        t.rowType, t.kRowFieldIdx, other.t.rowType, other.t.kRowFieldIdx)
        .compare
    )

  def innerJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toEphemeralIterator.innerJoinDistinct(
      other.iterator.toEphemeralIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      this.t.kComp(other.t).compare
    )

  def leftJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toEphemeralIterator.leftJoinDistinct(
      other.iterator.toEphemeralIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      this.t.kComp(other.t).compare
    )

  def innerJoin(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    iterator.toEphemeralIterator.innerJoin(
      other.iterator.toEphemeralIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      new RegionValueArrayBuffer(other.t.rowType),
      this.t.kComp(other.t).compare
    )
  }

  def leftJoin(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    iterator.toEphemeralIterator.leftJoin(
      other.iterator.toEphemeralIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      new RegionValueArrayBuffer(other.t.rowType),
      this.t.kComp(other.t).compare
    )
  }

  def rightJoin(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    iterator.toEphemeralIterator.rightJoin(
      other.iterator.toEphemeralIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      new RegionValueArrayBuffer(other.t.rowType),
      this.t.kComp(other.t).compare
    )
  }

  def outerJoin(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    iterator.toEphemeralIterator.outerJoin(
      other.iterator.toEphemeralIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      new RegionValueArrayBuffer(other.t.rowType),
      this.t.kComp(other.t).compare
    )
  }
}
