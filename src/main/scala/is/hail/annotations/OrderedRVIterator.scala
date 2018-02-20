package is.hail.annotations

import is.hail.rvd.OrderedRVDType
import is.hail.utils._

case class OrderedRVIterator(t: OrderedRVDType, iterator: Iterator[RegionValue]) {

  def staircase: StagingIterator[EphemeralIterator[RegionValue]] =
    iterator.toEphemeralIterator.staircased(t.mutableEquiv)

  def cogroup(other: OrderedRVIterator):
      EphemeralIterator[Muple[EphemeralIterator[RegionValue], EphemeralIterator[RegionValue]]] =
    this.staircase.toEphemeralIterator.orderedZipJoin(
      other.staircase,
      leftDefault = EphemeralIterator.empty,
      rightDefault = EphemeralIterator.empty,
      (l, r) => t.kComp(other.t).compare(l.head, r.head)
    )


  def zipJoin(other: OrderedRVIterator): EphemeralIterator[JoinedRegionValue] =
    iterator.toEphemeralIterator.orderedZipJoin(
      other.iterator.toEphemeralIterator,
      leftDefault = null,
      rightDefault = null,
      OrderedRVDType.selectUnsafeOrdering(
        t.rowType, t.kRowFieldIdx, other.t.rowType, other.t.kRowFieldIdx)
        .compare
    )

  def innerJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    val muple = Muple[RegionValue, RegionValue](null, null)
    for (Muple(l, r) <- this.cogroup(other) if (l.hasNext && r.hasNext))
    yield {
      muple.set(l.head, r.head)
      muple
    }
  }

  def leftJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] = {
    val muple = Muple[RegionValue, RegionValue](null, null)
    for { Muple(l, r) <- this.cogroup(other)
          lrv <- l
    } yield {
      muple.set(lrv, if (r.hasNext) r.head else null)
      muple
    }

  }
}
