package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable.ArrayBuffer

class OrderedRVDPartitioner(
  val partitionKey: Option[Int],
  val kType: TStruct,
  // rangeBounds: Array[Interval[kType]]
  // rangeBounds is interval containing all keys within a partition
  val rangeBounds: IndexedSeq[Interval]
) {
  def this(
    partitionKey: Array[String],
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ) = this(Some(partitionKey.length), kType, rangeBounds)

  require(rangeBounds.forall { case Interval(l, r, _, _) =>
    kType.relaxedTypeCheck(l) && kType.relaxedTypeCheck(r)
  })
  require(OrderedRVDPartitioner.isValid(partitionKey, kType, rangeBounds))

  val numPartitions: Int = rangeBounds.length

  val pkType = partitionKey.map(pk => TStruct(kType.fields.take(pk)))
  val rangeBoundsType = TArray(TInterval(kType))

  val rangeTree: IntervalTree[Int] = IntervalTree.fromSorted(kType.ordering,
    Array.tabulate[(Interval, Int)](numPartitions) { i =>
      (rangeBounds(i), i)
    })

  def coarsenedRangeBounds(newKeyLen: Int): IndexedSeq[Interval] =
    rangeBounds.map(_.coarsen(newKeyLen))

  def coarsen(newKeyLen: Int): OrderedRVDPartitioner =
    new OrderedRVDPartitioner(
      partitionKey.flatMap(pk => if (newKeyLen < pk) None else Some(pk)),
      kType.truncate(newKeyLen),
      coarsenedRangeBounds(newKeyLen)
    )

  def range: Option[Interval] = rangeTree.root.map(_.range)

  def contains(index: Int, key: Any): Boolean = {
    require(kType.isComparableAt(key))
    rangeBounds(index).contains(kType.ordering, key)
  }

  // Return the sequence of partition IDs overlapping the given interval of
  // keys.
  def getPartitionRange(query: Interval): Seq[Int] = {
    require(kType.isComparableAt(query.start) && kType.isComparableAt(query.end))
    if (!rangeTree.probablyOverlaps(kType.ordering, query))
      FastSeq.empty[Int]
    else
      rangeTree.queryOverlappingValues(kType.ordering, query)
  }

  // Get greatest partition ID whose lower bound is less than 'key'. Returns -1
  // if all partitions are above 'key'.
  def getSafePartitionUpperBound(key: Any): Int = {
    require(rangeBounds.nonEmpty)
    require(kType.isComparableAt(key))

    val range = getPartitionRange(Interval(Row(), key, true, true))
    range.lastOption.getOrElse(-1)
  }

  def getSafePartitionKeyRange(key: Any): Range =
    Range.inclusive(getSafePartitionLowerBound(key), getSafePartitionUpperBound(key))

  // Get least partition ID whose upper bound is greater than 'key'. Returns
  // numPartitions if all partitions are below 'key'.
  def getSafePartitionLowerBound(key: Any): Int = {
    require(rangeBounds.nonEmpty)
    require(kType.isComparableAt(key))

    val range = getPartitionRange(Interval(key, Row(), true, true))
    range.headOption.getOrElse(numPartitions)
  }

  def copy(
    partitionKey: Option[Int] = partitionKey,
    kType: TStruct = kType,
    rangeBounds: IndexedSeq[Interval] = rangeBounds
  ): OrderedRVDPartitioner =
    new OrderedRVDPartitioner(partitionKey, kType, rangeBounds)

  def enlargeToRange(newRange: Interval): OrderedRVDPartitioner =
    enlargeToRange(Some(newRange))

  def enlargeToRange(newRange: Option[Interval]): OrderedRVDPartitioner = {
    require(newRange.forall{i => kType.relaxedTypeCheck(i.start) && kType.relaxedTypeCheck(i.end)})

    if (newRange.isEmpty)
      return this
    if (range.isEmpty)
      return copy(rangeBounds = FastIndexedSeq(newRange.get))
    val pord: IntervalEndpointOrdering = kType.ordering.intervalEndpointOrdering
    val newLeft = pord.min(range.get.left, newRange.get.left).asInstanceOf[IntervalEndpoint]
    val newRight = pord.max(range.get.right, newRange.get.right).asInstanceOf[IntervalEndpoint]
    val newRangeBounds =
      rangeBounds match {
        case IndexedSeq(x) => FastIndexedSeq(Interval(newLeft, newRight))
        case _ =>
          rangeBounds.head.extendLeft(newLeft)  +:
            rangeBounds.tail.init :+
            rangeBounds.last.extendRight(newRight)
      }
    copy(rangeBounds = newRangeBounds)
  }

  def coalesceRangeBounds(newPartEnd: Array[Int]): OrderedRVDPartitioner = {
    val newRangeBounds = (-1 +: newPartEnd.init).zip(newPartEnd).map { case (s, e) =>
      rangeBounds(s+1).hull(kType.ordering, rangeBounds(e))
    }
    copy(rangeBounds = newRangeBounds)
  }

  @transient
  @volatile var partitionerBc: Broadcast[OrderedRVDPartitioner] = _

  def broadcast(sc: SparkContext): Broadcast[OrderedRVDPartitioner] = {
    if (partitionerBc == null) {
      synchronized {
        if (partitionerBc == null)
          partitionerBc = sc.broadcast(this)
      }
    }
    partitionerBc
  }

  def sparkPartitioner(sc: SparkContext): Partitioner = {
    val selfBc = broadcast(sc)

    new Partitioner {
      def numPartitions: Int = selfBc.value.numPartitions

      def getPartition(key: Any): Int = {
        val range = selfBc.value.getSafePartitionKeyRange(key)
        assert(range.size == 1)
        range.start
      }
    }
  }
}

object OrderedRVDPartitioner {
  def empty(typ: OrderedRVDType): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(typ.partitionKey, typ.kType, Array.empty[Interval])
  }

  // Factory for OrderedRVDPartitioner with weaker preconditions on 'rangeBounds'.
  def fixup(
    partitionKey: Array[String],
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ): OrderedRVDPartitioner = {
    if (rangeBounds.isEmpty)
      return new OrderedRVDPartitioner(partitionKey, kType, rangeBounds)

    val kord = kType.ordering
    val (pkType, _) = kType.select(partitionKey)
    val pkord = pkType.ordering

    require(rangeBounds.forall { case Interval(l, r, _, _) =>
      kType.relaxedTypeCheck(l) && kType.relaxedTypeCheck(r)
    })
    require {
      val lefts = rangeBounds.map(_.left)
      lefts.zip(lefts.tail).forall { case (l1, l2) => kord.intervalEndpointOrdering.lteq(l1, l2) }
    }
    require {
      val rights = rangeBounds.map(_.right)
      rights.zip(rights.tail).forall { case (r1, r2) => kord.intervalEndpointOrdering.lteq(r1, r2) }
    }

    val coarsenedBounds = rangeBounds.map(_.coarsen(partitionKey.length))
    val newBounds = new ArrayBuilder[Interval]()
    var i = 0
    var cursor = rangeBounds.head.left
    while (i != rangeBounds.length) {
      val left = cursor
      val (right, nextLeft) =
        if (i == rangeBounds.length - 1)
          (rangeBounds(i).right, null)
        else if (coarsenedBounds(i).isBelow(pkord, coarsenedBounds(i + 1)))
          (rangeBounds(i).right, rangeBounds(i + 1).left)
        else
          (coarsenedBounds(i).right, coarsenedBounds(i).right)
      Interval.orNone(kord, left, right) match {
        case None =>
        case Some(interval) =>
          newBounds += interval
          cursor = nextLeft
      }
      i += 1
    }

    new OrderedRVDPartitioner(partitionKey, kType, newBounds.result())
  }

  def generate(
    partitionKey: Array[String],
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ): OrderedRVDPartitioner = {
    require(rangeBounds.forall { case Interval(l, r, _, _) =>
      kType.relaxedTypeCheck(l) && kType.relaxedTypeCheck(r)
    })

    if (rangeBounds.isEmpty)
      return new OrderedRVDPartitioner(partitionKey, kType, rangeBounds)

    val kord = kType.ordering
    val eord = kord.intervalEndpointOrdering.toOrdering.asInstanceOf[Ordering[IntervalEndpoint]]

    val chunked = union(partitionKey, kType, rangeBounds)
    val cutPoints = rangeBounds
      .map(_.right.coarsenRight(partitionKey.length))
      .sorted(eord)

    var i = 0
    val newBounds = chunked.flatMap { chunk =>
      val first = cutPoints.indexWhere(eord.gt(_, chunk.left), i)
      val last = cutPoints.indexWhere(eord.gt(_, chunk.right), first)
      val cuts = cutPoints.slice(first, last)
      i = last
      for {
        (l, r) <- (chunk.left +: cuts) zip (cuts :+ chunk.right)
        interval <- Interval.orNone(kord, l, r)
      } yield interval
    }

    new OrderedRVDPartitioner(partitionKey, kType, newBounds)
  }

  private def union(
    partitionKey: Array[String],
    kType: TStruct,
    intervals: IndexedSeq[Interval]
  ): IndexedSeq[Interval] = {
    val kord = kType.ordering
    val eord = kord.intervalEndpointOrdering
    val iord = Interval.ordering(kord, startPrimary = true)
    if (intervals.isEmpty)
      intervals
    else {
      val unpruned = intervals.sorted(iord.toOrdering.asInstanceOf[Ordering[Interval]])
      val pk = partitionKey.length
      val ab = new ArrayBuilder[Interval](intervals.length)
      var tmp = unpruned(0)
      for (i <- unpruned.tail) {
        if (eord.gteq(tmp.right.coarsenRight(pk), i.left.coarsenLeft(pk)))
          tmp = tmp.hull(kord, i)
        else {
          ab += tmp
          tmp = i
        }
      }
      ab += tmp
      ab.result()
    }
  }

  // takes npartitions + 1 points and returns npartitions intervals: [a,b], (b,c], (c,d], ... (i, j]
  def makeRangeBoundIntervals(rangeBounds: Array[Any]): Array[Interval] = {
    var includesStart = true
    rangeBounds.zip(rangeBounds.tail).map { case (s, e) =>
      val i = Interval(s, e, includesStart, true)
      includesStart = false
      i
    }
  }

  def isValid(
    partitionKey: Option[Int],
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ): Boolean = {
    rangeBounds.isEmpty ||
      (rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
      left.isBelow(kType.ordering, right) ||
        (left.end.asInstanceOf[Row].size == kType.size &&
          right.start.asInstanceOf[Row].size == kType.size &&
          kType.ordering.equiv(left.end, right.start))
    } && partitionKey.forall { pk =>
      val pkBounds = rangeBounds.map(_.coarsen(pk))
      pkBounds.zip(pkBounds.tail).forall { case (left: Interval, right: Interval) =>
        left.isBelow(kType.ordering, right)
      }
    })
  }
}
