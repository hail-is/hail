package is.hail.annotations

import is.hail.rvd.OrderedRVType
import is.hail.sparkextras.StaircaseIterator
import is.hail.utils.{Muple, OrderedZipJoinIterator}

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
    val leftStair = this.staircase
    val rightStair = other.staircase
    val ord = OrderedRVType.selectUnsafeOrdering(
      this.t.rowType,
      this.t.kRowFieldIdx,
      other.t.rowType,
      other.t.kRowFieldIdx)
    new OrderedZipJoinIterator[BufferedIterator[RegionValue],
                               BufferedIterator[RegionValue]](
      left = this.staircase.buffered,
      leftDefault = Iterator.empty.buffered,
      right = other.staircase.buffered,
      rightDefault = Iterator.empty.buffered,
      (l, r) => ord.compare(l.head, r.head)
    )
  }
}
