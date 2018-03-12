package is.hail.annotations

import is.hail.rvd.OrderedRVDType
import is.hail.utils._

case class OrderedRVIterator(t: OrderedRVDType, iterator: Iterator[RegionValue]) {

  def restrictToPKInterval(interval: Interval): Iterator[RegionValue] = {
    val ur = new UnsafeRow(t.rowType, null, 0)
    val pk = new KeyedRow(ur, t.pkRowFieldIdx)
    iterator.filter( rv => {
      ur.set(rv)
      interval.contains(t.pkType.ordering, pk)
    } )
  }

  def staircase: StagingIterator[FlipbookIterator[RegionValue]] =
    iterator.toFlipbookIterator.staircased(t.kRowOrdView)

  def cogroup(other: OrderedRVIterator):
      FlipbookIterator[Muple[FlipbookIterator[RegionValue], FlipbookIterator[RegionValue]]] =
    this.iterator.toFlipbookIterator.cogroup(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      this.t.kComp(other.t).compare
    )

  def zipJoin(other: OrderedRVIterator): FlipbookIterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.orderedZipJoin(
      other.iterator.toFlipbookIterator,
      leftDefault = null,
      rightDefault = null,
      OrderedRVDType.selectUnsafeOrdering(
        t.rowType, t.kRowFieldIdx, other.t.rowType, other.t.kRowFieldIdx)
        .compare
    )

  def innerJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.innerJoinDistinct(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      this.t.kComp(other.t).compare
    )

  def leftJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.leftJoinDistinct(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      this.t.kComp(other.t).compare
    )

  def innerJoin(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.innerJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      new RegionValueArrayBuffer(other.t.rowType),
      this.t.kComp(other.t).compare
    )
  }

  def leftJoin(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.leftJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      new RegionValueArrayBuffer(other.t.rowType),
      this.t.kComp(other.t).compare
    )
  }

  def rightJoin(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.rightJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      new RegionValueArrayBuffer(other.t.rowType),
      this.t.kComp(other.t).compare
    )
  }

  def outerJoin(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.outerJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView,
      other.t.kRowOrdView,
      null,
      null,
      new RegionValueArrayBuffer(other.t.rowType),
      this.t.kComp(other.t).compare
    )
  }
}
