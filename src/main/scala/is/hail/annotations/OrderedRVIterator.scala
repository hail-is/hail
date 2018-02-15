package is.hail.annotations

import is.hail.rvd.OrderedRVType
import is.hail.utils._

case class OrderedRVIterator(t: OrderedRVType, iterator: Iterator[RegionValue]) {

  def staircase: StagingIterator[EphemeralIterator[RegionValue]] =
    iterator.toEphemeralIterator.staircased(t.mutableEquiv)

  def cogroup(other: OrderedRVIterator):
      EphemeralIterator[Muple[EphemeralIterator[RegionValue], EphemeralIterator[RegionValue]]] =
    new OrderedZipJoinIterator[EphemeralIterator[RegionValue],
                               EphemeralIterator[RegionValue]](
      left = this.staircase,
      leftDefault = EphemeralIterator.empty,
      right = other.staircase,
      rightDefault = EphemeralIterator.empty,
      (l, r) => t.kComp(other.t).compare(l.head, r.head)
    )


  def zipJoin(other: OrderedRVIterator): EphemeralIterator[JoinedRegionValue] =
    new OrderedZipJoinIterator[RegionValue, RegionValue](
      left = iterator.toStagingIterator,
      leftDefault = null,
      right = other.iterator.toStagingIterator,
      rightDefault = null,
      OrderedRVType.selectUnsafeOrdering(
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
