package is.hail.annotations

import is.hail.rvd.OrderedRVType
import is.hail.utils.{Muple, OrderedZipJoinIterator, StaircaseIterator}

case class OrderedRVIterator(t: OrderedRVType, iterator: BufferedIterator[RegionValue])
  extends BufferedIterator[RegionValue] {

  def hasNext: Boolean = iterator.hasNext
  def head: RegionValue = iterator.head
  def next(): RegionValue = iterator.next()

  def staircase: StaircaseIterator[RegionValue] =
    new StaircaseIterator(iterator, t.mutableEquiv)

  def cogroup(other: OrderedRVIterator):
      Iterator[Muple[BufferedIterator[RegionValue], BufferedIterator[RegionValue]]] =
  {
    new OrderedZipJoinIterator[BufferedIterator[RegionValue],
                               BufferedIterator[RegionValue]](
      left = this.staircase.buffered,
      leftDefault = Iterator.empty.buffered,
      right = other.staircase.buffered,
      rightDefault = Iterator.empty.buffered,
      (l, r) => t.kComp(other.t).compare(l.head, r.head)
    )
  }

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
