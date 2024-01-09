package is.hail.annotations

import is.hail.backend.HailStateManager
import is.hail.rvd.{RVDContext, RVDType}
import is.hail.types.physical.PInterval
import is.hail.utils._

import scala.collection.generic.Growable
import scala.collection.mutable

object OrderedRVIterator {
  def multiZipJoin(
    sm: HailStateManager,
    its: IndexedSeq[OrderedRVIterator],
  ): Iterator[BoxedArrayBuilder[(RegionValue, Int)]] = {
    require(its.length > 0)
    val first = its(0)
    val flipbooks = its.map(_.iterator.toFlipbookIterator)
    FlipbookIterator.multiZipJoin(
      flipbooks.toArray,
      first.t.joinComp(sm, first.t).compare,
    )
  }
}

case class OrderedRVIterator(
  t: RVDType,
  iterator: Iterator[RegionValue],
  ctx: RVDContext,
  sm: HailStateManager,
) {

  def staircase: StagingIterator[FlipbookIterator[RegionValue]] =
    iterator.toFlipbookIterator.staircased(t.kRowOrdView(sm, ctx.freshRegion()))

  def cogroup(other: OrderedRVIterator)
    : FlipbookIterator[Muple[FlipbookIterator[RegionValue], FlipbookIterator[RegionValue]]] =
    this.iterator.toFlipbookIterator.cogroup(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(sm, ctx.freshRegion()),
      other.t.kRowOrdView(sm, ctx.freshRegion()),
      this.t.kComp(sm, other.t).compare,
    )

  def leftJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.leftJoinDistinct(
      other.iterator.toFlipbookIterator,
      null,
      null,
      this.t.joinComp(sm, other.t).compare,
    )

  def leftIntervalJoinDistinct(other: OrderedRVIterator): Iterator[JoinedRegionValue] =
    iterator.toFlipbookIterator.leftJoinDistinct(
      other.iterator.toFlipbookIterator,
      null,
      null,
      this.t.intervalJoinComp(sm, other.t).compare,
    )

  def leftIntervalJoin(other: OrderedRVIterator)
    : Iterator[Muple[RegionValue, Iterable[RegionValue]]] = {
    val left = iterator.toFlipbookIterator
    val right = other.iterator.toFlipbookIterator
    val rightEndpointOrdering: Ordering[RegionValue] = RVDType.selectUnsafeOrdering(
      other.t.rowType,
      other.t.kFieldIdx,
      other.t.rowType,
      other.t.kFieldIdx,
      Array(other.t.kType.types(0).asInstanceOf[PInterval].endPrimaryUnsafeOrdering(sm)),
      missingEqual = true,
    ).toRVOrdering.reverse
    val mixedOrd: (RegionValue, RegionValue) => Int = this.t.intervalJoinComp(sm, other.t).compare

    val stateMachine = new StateMachine[Muple[RegionValue, Iterable[RegionValue]]] {
      val buffer = new RegionValuePriorityQueue(sm, other.t.rowType, ctx, rightEndpointOrdering)

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

    FlipbookIterator(stateMachine)
  }

  def innerJoin(
    other: OrderedRVIterator,
    rightBuffer: Iterable[RegionValue] with Growable[RegionValue],
  ): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.innerJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(sm, ctx.freshRegion()),
      other.t.kRowOrdView(sm, ctx.freshRegion()),
      null,
      null,
      rightBuffer,
      this.t.joinComp(sm, other.t).compare,
    )
  }

  def leftJoin(
    other: OrderedRVIterator,
    rightBuffer: Iterable[RegionValue] with Growable[RegionValue],
  ): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.leftJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(sm, ctx.freshRegion()),
      other.t.kRowOrdView(sm, ctx.freshRegion()),
      null,
      null,
      rightBuffer,
      this.t.joinComp(sm, other.t).compare,
    )
  }

  def rightJoin(
    other: OrderedRVIterator,
    rightBuffer: Iterable[RegionValue] with Growable[RegionValue],
  ): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.rightJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(sm, ctx.freshRegion()),
      other.t.kRowOrdView(sm, ctx.freshRegion()),
      null,
      null,
      rightBuffer,
      this.t.joinComp(sm, other.t).compare,
    )
  }

  def outerJoin(
    other: OrderedRVIterator,
    rightBuffer: Iterable[RegionValue] with Growable[RegionValue],
  ): Iterator[JoinedRegionValue] = {
    iterator.toFlipbookIterator.outerJoin(
      other.iterator.toFlipbookIterator,
      this.t.kRowOrdView(sm, ctx.freshRegion()),
      other.t.kRowOrdView(sm, ctx.freshRegion()),
      null,
      null,
      rightBuffer,
      this.t.joinComp(sm, other.t).compare,
    )
  }

  def merge(other: OrderedRVIterator): Iterator[RegionValue] =
    iterator.toFlipbookIterator.merge(
      other.iterator.toFlipbookIterator,
      this.t.kComp(sm, other.t).compare,
    )

  def localKeySort(
    newKey: IndexedSeq[String]
  ): Iterator[RegionValue] = {
    require(newKey startsWith t.key)
    require(newKey.forall(t.rowType.fieldNames.contains))

    val consumerRegion = ctx.region

    new Iterator[RegionValue] {
      private val bit = iterator.buffered

      private val q = new mutable.PriorityQueue[RegionValue]()(
        t.copy(key = newKey).kInRowOrd(sm).toRVOrdering.reverse
      )

      private val rvb = new RegionValueBuilder(sm, consumerRegion)
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
          } while (bit.hasNext && t.kInRowOrd(sm).compare(q.head, bit.head) == 0)
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
