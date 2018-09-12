package is.hail.annotations

import is.hail.rvd.{OrderedRVDType, RVDContext}
import is.hail.utils._

import scala.collection.generic.Growable

case class OrderedRVIterator(
  t: OrderedRVDType,
  iterator: Iterator[RegionValue],
  ctx: RVDContext
) {

  def staircase: StagingIterator[FlipbookIterator[RegionValue]] =
    iterator.toFlipbookIterator.staircased(t.kRowOrdView(ctx.freshRegion))

  def cogroup(other: OrderedRVIterator):
      FlipbookIterator[Muple[FlipbookIterator[RegionValue], FlipbookIterator[RegionValue]]] =
    this.iterator.toFlipbookIterator.cogroup(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(ctx.freshRegion),
      other.t.kRowOrdView(ctx.freshRegion),
      this.t.kComp(other.t).compare
    )

  def zipJoin(other: OrderedRVIterator): FlipbookIterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.orderedZipJoin(
      other.iterator.toFlipbookIterator,
      leftDefault = null,
      rightDefault = null,
      OrderedRVDType.selectUnsafeOrdering(
        t.rowType, t.kFieldIdx, other.t.rowType, other.t.kFieldIdx)
        .compare
    )

  def innerJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.innerJoinDistinct(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(ctx.freshRegion),
      other.t.kRowOrdView(ctx.freshRegion),
      null,
      null,
      this.t.joinComp(other.t).compare
    )

  def leftJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.leftJoinDistinct(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(ctx.freshRegion),
      other.t.kRowOrdView(ctx.freshRegion),
      null,
      null,
      this.t.joinComp(other.t).compare
    )

  def innerJoin(
    other: OrderedRVIterator,
    rightBuffer: Iterable[RegionValue] with Growable[RegionValue]
  ): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.innerJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(ctx.freshRegion),
      other.t.kRowOrdView(ctx.freshRegion),
      null,
      null,
      rightBuffer,
      this.t.joinComp(other.t).compare
    )
  }

  def leftJoin(
    other: OrderedRVIterator,
    rightBuffer: Iterable[RegionValue] with Growable[RegionValue]
  ): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.leftJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(ctx.freshRegion),
      other.t.kRowOrdView(ctx.freshRegion),
      null,
      null,
      rightBuffer,
      this.t.joinComp(other.t).compare
    )
  }

  def rightJoin(
    other: OrderedRVIterator,
    rightBuffer: Iterable[RegionValue] with Growable[RegionValue]
  ): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.rightJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(ctx.freshRegion),
      other.t.kRowOrdView(ctx.freshRegion),
      null,
      null,
      rightBuffer,
      this.t.joinComp(other.t).compare
    )
  }

  def outerJoin(
    other: OrderedRVIterator,
    rightBuffer: Iterable[RegionValue] with Growable[RegionValue]
  ): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.outerJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(ctx.freshRegion),
      other.t.kRowOrdView(ctx.freshRegion),
      null,
      null,
      rightBuffer,
      this.t.joinComp(other.t).compare
    )
  }

  def merge(other: OrderedRVIterator): Iterator[RegionValue] = {
    iterator.toFlipbookIterator.merge(
      other.iterator.toFlipbookIterator,
      this.t.kComp(other.t).compare
    )
  }
}
