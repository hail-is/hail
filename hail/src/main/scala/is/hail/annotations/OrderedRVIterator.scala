package is.hail.annotations

import is.hail.types.physical.PInterval
import is.hail.rvd.{RVDContext, RVDType}
import is.hail.utils._

import scala.collection.generic.Growable
import scala.collection.mutable

object OrderedRVIterator {
  def multiZipJoin(
    its: IndexedSeq[OrderedRVIterator]
  ): Iterator[BoxedArrayBuilder[(RegionValue, Int)]] = {
    require(its.length > 0)
    val first = its(0)
    val flipbooks = its.map(_.iterator.toFlipbookIterator)
    FlipbookIterator.multiZipJoin(
      flipbooks.toArray,
      first.t.joinComp(first.t).compare
    )
  }
}

case class OrderedRVIterator(
  t: RVDType,
  iterator: Iterator[RegionValue],
  ctx: RVDContext
) {

  def staircase: StagingIterator[FlipbookIterator[RegionValue]] =
    iterator.toFlipbookIterator.staircased(t.kRowOrdView(ctx.freshRegion()))

  def leftIntervalJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.leftJoinDistinct(
      other.iterator.toFlipbookIterator,
      null,
      null,
      this.t.intervalJoinComp(other.t).compare
    )

  def leftIntervalJoin(other: OrderedRVIterator): Iterator[Muple[RegionValue, Iterable[RegionValue]]] = {
    val left = iterator.toFlipbookIterator
    val right = other.iterator.toFlipbookIterator
    val rightEndpointOrdering: Ordering[RegionValue] = RVDType.selectUnsafeOrdering(
      other.t.rowType,
      other.t.kFieldIdx,
      other.t.rowType,
      other.t.kFieldIdx,
      Array(other.t.kType.types(0).asInstanceOf[PInterval].endPrimaryUnsafeOrdering()),
      missingEqual = true
    ).toRVOrdering.reverse
    val mixedOrd: (RegionValue, RegionValue) => Int = this.t.intervalJoinComp(other.t).compare

    val sm = new StateMachine[Muple[RegionValue, Iterable[RegionValue]]] {
      val buffer = new RegionValuePriorityQueue(other.t.rowType, ctx, rightEndpointOrdering)

      val value: Muple[RegionValue, Iterable[RegionValue]] = Muple(null, buffer)

      var isValid: Boolean = true

      def setValue() {
        if (left.isValid) {
          while (buffer.nonEmpty && mixedOrd(left.value, buffer.head) > 0)
            buffer.dequeue()
          while (right.isValid && mixedOrd(left.value, right.value) >= 0) {
            if (mixedOrd(left.value, right.value) == 0) {
              buffer.enqueue(right.value)
            }
            right.advance()
          }
          value.set(left.value, buffer)
        } else {
          isValid = false
        }
      }

      def advance() {
        left.advance()
        setValue()
      }

      setValue()
    }

    FlipbookIterator(sm)
  }

  def merge(other: OrderedRVIterator): Iterator[RegionValue] = {
    iterator.toFlipbookIterator.merge(
      other.iterator.toFlipbookIterator,
      this.t.kComp(other.t).compare
    )
  }

  def localKeySort(
    newKey: IndexedSeq[String]
  ): Iterator[RegionValue] = {
    require(newKey startsWith t.key)
    require(newKey.forall(t.rowType.fieldNames.contains))

    val consumerRegion = ctx.region

    new Iterator[RegionValue] {
      private val bit = iterator.buffered

      private val q = new mutable.PriorityQueue[RegionValue]()(
        t.copy(key = newKey).kInRowOrd.toRVOrdering.reverse)

      private val rvb = new RegionValueBuilder(consumerRegion)
      private val rv = RegionValue()

      def hasNext: Boolean = bit.hasNext || q.nonEmpty

      def next(): RegionValue = {
        if (q.isEmpty) {
          do {
            val rv = bit.next()
            val r = ctx.freshRegion()
            rvb.set(r)
            rvb.start(t.rowType)
            rvb.addRegionValue(t.rowType, rv)
            q.enqueue(RegionValue(rvb.region, rvb.end()))
          } while (bit.hasNext && t.kInRowOrd.compare(q.head, bit.head) == 0)
        }

        rvb.set(consumerRegion)
        rvb.start(t.rowType)
        val fromQueue = q.dequeue()
        rvb.addRegionValue(t.rowType, fromQueue)
        ctx.closeChild(fromQueue.region)
        rv.set(consumerRegion, rvb.end())
        rv
      }
    }
  }
}
