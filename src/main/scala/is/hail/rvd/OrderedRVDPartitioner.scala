package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.broadcast.Broadcast

class OrderedRVDPartitioner(
  val kType: TStruct,
  // rangeBounds: Array[Interval[kType]]
  // rangeBounds is interval containing all keys within a partition
  val rangeBounds: IndexedSeq[Interval],
  allowedOverlap: Int
) {
  def this(
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ) = this(kType, rangeBounds, kType.size)

  def this(
    partitionKey: Array[String],
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ) = this(kType, rangeBounds, math.max(partitionKey.length - 1, 0))

  def this(
    partitionKey: Option[Int],
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ) = this(kType, rangeBounds, partitionKey.map(_ - 1).getOrElse(kType.size))

  require(rangeBounds.forall { case Interval(l, r, _, _) =>
    kType.relaxedTypeCheck(l) && kType.relaxedTypeCheck(r)
  })
  require(allowedOverlap >= 0 && allowedOverlap <= kType.size)
  require(OrderedRVDPartitioner.isValid(kType, rangeBounds, allowedOverlap))

  def satisfiesAllowedOverlap(testAllowedOverlap: Int): Boolean =
    OrderedRVDPartitioner.isValid(kType, rangeBounds, testAllowedOverlap)

  val numPartitions: Int = rangeBounds.length

  val rangeBoundsType = TArray(TInterval(kType))

  val rangeTree: IntervalTree[Int] = IntervalTree.fromSorted(kType.ordering,
    Array.tabulate[(Interval, Int)](numPartitions) { i =>
      (rangeBounds(i), i)
    })

  override def equals(other: Any): Boolean = other match {
    case that: OrderedRVDPartitioner =>
      this.kType == that.kType && this.rangeBounds == that.rangeBounds
    case _ => false
  }

  def coarsenedRangeBounds(newKeyLen: Int): IndexedSeq[Interval] =
    rangeBounds.map(_.coarsen(newKeyLen))

  def coarsen(newKeyLen: Int): OrderedRVDPartitioner =
    new OrderedRVDPartitioner(
      kType.truncate(newKeyLen),
      coarsenedRangeBounds(newKeyLen)
    )

  // Adjusts 'rangeBounds' so that 'satisfiesAllowedOverlap(kType.size - 1)'
  // holds, then changes key type to 'newKType'. If 'newKType' is 'kType', still
  // adjusts 'rangeBounds'.
  def extendKey(newKType: TStruct): OrderedRVDPartitioner = {
    require(kType isPrefixOf newKType)
    OrderedRVDPartitioner.generate(newKType.fieldNames, newKType, rangeBounds)
  }

  def subdivide(
    cutPoints: IndexedSeq[IntervalEndpoint],
    allowedOverlap: Int = kType.size
  ): OrderedRVDPartitioner = {
    require(cutPoints.forall { case IntervalEndpoint(row, _) =>
      kType.relaxedTypeCheck(row)
    })
    require(allowedOverlap >= 0 && allowedOverlap <= kType.size)
    require(satisfiesAllowedOverlap(allowedOverlap))

    val kord = kType.ordering
    val eord = kord.intervalEndpointOrdering.toOrdering.asInstanceOf[Ordering[IntervalEndpoint]]
    val sorted = cutPoints.map(_.coarsenRight(allowedOverlap + 1)).sorted(eord)

    var i = 0
    def firstPast(threshold: IntervalEndpoint, start: Int): Int = {
      val iw = sorted.indexWhere(eord.gt(_, threshold), start)
      if (iw == -1) sorted.length else iw
    }
    val newBounds = rangeBounds.flatMap { interval =>
      val first = firstPast(interval.left, i)
      val last = firstPast(interval.right, first)
      val cuts = sorted.slice(first, last)
      i = last
      for {
        (l, r) <- (interval.left +: cuts) zip (cuts :+ interval.right)
        interval <- Interval.orNone(kord, l, r)
      } yield interval
    }

    new OrderedRVDPartitioner(kType, newBounds, allowedOverlap)
  }

  def range: Option[Interval] = rangeTree.root.map(_.range)

  def contains(index: Int, key: Any): Boolean = {
    require(kType.isComparableAt(key))
    rangeBounds(index).contains(kType.ordering, key)
  }

  // Return the sequence of partition IDs overlapping the given interval of
  // keys.
  def getPartitionRange(query: Interval): Seq[Int] = {
    require(kType.isComparableAt(query.start) && kType.isComparableAt(query.end))
    if (!rangeTree.overlaps(kType.ordering, query))
      FastSeq.empty[Int]
    else
      rangeTree.queryOverlappingValues(kType.ordering, query)
  }

  // Returns the least partition which is not completely below 'key', i.e. the
  // least partition whose upper bound is greater than 'key'. Returns
  // numPartitions if all partitions are below 'key'. The range of partitions
  // which can contain 'key' is always [lowerBound, upperBound). lowerBound =
  // upperBound if and only if 'key' is not contained in the partitioner, in
  // which case i = lowerBound is the partition index at which a new partition
  // containing 'key' would be inserted (becoming the new partition i, between
  // the old partition i-1 and the old partition i).
  def getSafePartitionLowerBound(key: Any): Int = {
    require(rangeBounds.nonEmpty)
    require(kType.isComparableAt(key))

    val range = getPartitionRange(Interval(key, Row(), true, true))
    range.headOption.getOrElse(numPartitions)
  }

  // Returns the least partition which is completely above 'key', i.e. the least
  // partition whose upper bound is greater than 'key'. Returns numPartitions
  // if no partition is above 'key'. The range of partitions which can contain
  // 'key' is always [lowerBound, upperBound).
  def getSafePartitionUpperBound(key: Any): Int = {
    require(rangeBounds.nonEmpty)
    require(kType.isComparableAt(key))

    val range = getPartitionRange(Interval(Row(), key, true, true))
    range.lastOption.getOrElse(-1) + 1
  }

  def getSafePartitionKeyRange(key: Any): Range =
    Range(getSafePartitionLowerBound(key), getSafePartitionUpperBound(key))

  def copy(
    kType: TStruct = kType,
    rangeBounds: IndexedSeq[Interval] = rangeBounds
  ): OrderedRVDPartitioner =
    new OrderedRVDPartitioner(kType, rangeBounds)

  def enlargeToRange(newRange: Interval): OrderedRVDPartitioner =
    enlargeToRange(Some(newRange))

  def enlargeToRange(newRange: Option[Interval]): OrderedRVDPartitioner = {
    require(newRange.forall(i => kType.relaxedTypeCheck(i.start) && kType.relaxedTypeCheck(i.end)))

    if (newRange.isEmpty)
      return this
    if (range.isEmpty)
      return copy(rangeBounds = FastIndexedSeq(newRange.get))
    val pord = kType.ordering.intervalEndpointOrdering
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

  def coalesceRangeBounds(newPartEnd: IndexedSeq[Int]): OrderedRVDPartitioner = {
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
    new OrderedRVDPartitioner(typ.kType, Array.empty[Interval])
  }

  def unkeyed(numPartitions: Int): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(
      TStruct.empty(),
      Array.fill(numPartitions)(Interval(Row(), Row(), true, true)),
      0)
  }

  def generate(
    partitionKey: IndexedSeq[String],
    kType: TStruct,
    intervals: IndexedSeq[Interval]
  ): OrderedRVDPartitioner = {
    require(intervals.forall { case Interval(l, r, _, _) =>
      kType.relaxedTypeCheck(l) && kType.relaxedTypeCheck(r)
    })

    val allowedOverlap = math.max(partitionKey.length - 1, 0)
    union(kType, intervals, allowedOverlap).subdivide(intervals.map(_.right), allowedOverlap)
  }

  private def union(
    kType: TStruct,
    intervals: IndexedSeq[Interval],
    allowedOverlap: Int
  ): OrderedRVDPartitioner = {
    val kord = kType.ordering
    val eord = kord.intervalEndpointOrdering
    val iord = Interval.ordering(kord, startPrimary = true)
    val pk = allowedOverlap + 1
    val rangeBounds: IndexedSeq[Interval] =
      if (intervals.isEmpty)
        intervals
      else {
        val unpruned = intervals.sorted(iord.toOrdering.asInstanceOf[Ordering[Interval]])
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

    new OrderedRVDPartitioner(kType, rangeBounds, allowedOverlap)
  }

  def fromKeySamples(
    typ: OrderedRVDType,
    min: Any,
    max: Any,
    keys: IndexedSeq[Any],
    nPartitions: Int,
    partitionKey: Int
  ): OrderedRVDPartitioner = {
    require(nPartitions > 0)
    require(typ.kType.relaxedTypeCheck(min))
    require(typ.kType.relaxedTypeCheck(max))
    require(keys.forall(typ.kType.relaxedTypeCheck))

    val sortedKeys = keys.sorted(typ.kType.ordering.toOrdering)
    val step = (sortedKeys.length - 1).toDouble / nPartitions
    val partitionEdges = Array.tabulate(nPartitions - 1) { i =>
      IntervalEndpoint(sortedKeys(((i + 1) * step).toInt), 1)
    }.toFastIndexedSeq

    val interval = Interval(min, max, true, true)
    new OrderedRVDPartitioner(
      typ.kType,
      FastIndexedSeq(interval)
    ).subdivide(partitionEdges, math.max(partitionKey - 1, 0))
  }

  def isValid(kType: TStruct, rangeBounds: IndexedSeq[Interval]): Boolean =
    isValid(kType, rangeBounds, kType.size)

  def isValid(
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval],
    allowedOverlap: Int
  ): Boolean = {
    rangeBounds.isEmpty ||
      rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
        kType.ordering.intervalEndpointOrdering.lteqWithOverlap(allowedOverlap)(left.right, right.left)
      }
  }
}
