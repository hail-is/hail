package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.broadcast.Broadcast

class OrderedRVDPartitioner(
  val partitionKey: Array[String], val kType: TStruct,
  // rangeBounds: Array[Interval[pkType]]
  // rangeBounds is interval containing all partition keys within a partition
  val rangeBounds: IndexedSeq[Interval]) {
  val numPartitions: Int = rangeBounds.length

  val (pkType, _) = kType.select(partitionKey)
  val pkIntervalType = TInterval(pkType)
  val rangeBoundsType = TArray(pkIntervalType)

  require(rangeBounds.forall { case Interval(l, r, _, _) =>
    pkType.isComparableAt(l) && pkType.isComparableAt(r)
  })

  require(rangeBounds.isEmpty || (rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
    !left.mayOverlap(pkType.ordering, right) && pkType.ordering.lteq(left.start, right.start)
  } && rangeBounds.forall { i: Interval =>
    pkType.ordering.lteq(i.start, i.end) && !i.definitelyEmpty(pkType.ordering)
  }))

  require(rangeBounds.isEmpty || rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
    pkType.ordering.equiv(left.end, right.start) && (left.includesEnd || right.includesStart) } )

  val rangeTree: IntervalTree[Int] = IntervalTree.fromSorted(pkType.ordering,
    Array.tabulate[(Interval, Int)](numPartitions) { i =>
      (rangeBounds(i), i)
    })

  def coarsenedPKRangeTree(newPK: Int): IntervalTree[Int] = {
    val (newPKType, getNewPK) = pkType.select(partitionKey.take(newPK))
    IntervalTree.fromSorted(
      newPKType.ordering,
      Array.tabulate[(Interval, Int)](numPartitions) { i =>
        (Interval(
          getNewPK(rangeBounds(i).start.asInstanceOf[Row]),
          getNewPK(rangeBounds(i).end.asInstanceOf[Row]),
          includesStart = true,
          includesEnd = true),
        i)
      }
    )
  }

  def range: Option[Interval] = rangeTree.root.map(_.range)

  /**
    * Find the partition containing the given Row.
    *
    * If pkType is a prefix of the type of row, the prefix of row is used to
    * find the partition.
    *
    * If row falls outside the bounds of the partitioner, return the min or max
    * partition.
    */
  def getPartitionPK(row: Any): Int = {
    require(rangeBounds.nonEmpty)
    val part = rangeTree.queryValues(pkType.ordering, row)
    part match {
      case Array() =>
        if (range.get.isAbovePosition(pkType.ordering, row))
          0
        else {
          assert(range.get.isBelowPosition(pkType.ordering, row))
          numPartitions - 1
        }

      case Array(x) => x
    }
  }

  // Return the sequence of partition IDs overlapping the given interval of
  // partition keys.
  def getPartitionRange(query: Any): Seq[Int] = {
    query match {
      case row: Row =>
        rangeTree.queryValues(pkType.ordering, row)
      case interval: Interval =>
        if (!rangeTree.probablyOverlaps(pkType.ordering, interval))
          FastSeq.empty[Int]
        else {
          val startRange = getPartitionRange(interval.start)
          val start = if (startRange.nonEmpty)
            startRange.min
          else
            0
          val endRange =  getPartitionRange(interval.end)
          val end = if (endRange.nonEmpty)
            endRange.max
          else
            numPartitions - 1
          start to end
        }
    }
  }

  // return the partition containing key
  // if outside bounds, return min or max depending on location
  // key: RegionValue[kType]
  def getPartition(key: Any): Int =
    getPartitionPK(new UnsafeRow(kType, key.asInstanceOf[RegionValue]))

  def getSafePartition(key: Any): Int =
    getPartitionPK(key)

  def withKType(newPartitionKey: Array[String], newKType: TStruct): OrderedRVDPartitioner = {
    val newPart = new OrderedRVDPartitioner(newPartitionKey, newKType, rangeBounds)
    assert(newPart.pkType.types.sameElements(pkType.types))
    newPart
  }

  def copy(numPartitions: Int = numPartitions, partitionKey: Array[String] = partitionKey,
    kType: TStruct = kType, rangeBounds: IndexedSeq[Interval] = rangeBounds): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(partitionKey, kType, rangeBounds)
  }

  def enlargeToRange(newRange: Interval): OrderedRVDPartitioner =
    enlargeToRange(Some(newRange))

  // FIXME Make work if newRange has different point type than pkType
  def enlargeToRange(newRange: Option[Interval]): OrderedRVDPartitioner = {
    if (newRange.isEmpty)
      return this
    if (range.isEmpty)
      return copy(rangeBounds = FastIndexedSeq(newRange.get))
    val newStart = pkType.ordering.min(range.get.start, newRange.get.start)
    val newEnd = pkType.ordering.max(range.get.end, newRange.get.end)
    val newRangeBounds =
      rangeBounds match {
        case IndexedSeq(x) => FastIndexedSeq(Interval(newStart, newEnd, true, true))
        case IndexedSeq(x1, x2) =>
          FastIndexedSeq(x1.copy(start = newStart, includesStart = true),
            x2.copy(end = newEnd, includesEnd = true))
        case _ =>
          rangeBounds.head.copy(start = newStart, includesStart = true)  +:
            rangeBounds.tail.init :+
            rangeBounds.last.copy(end = newEnd, includesEnd = true)
      }
    copy(rangeBounds = newRangeBounds)
  }

  def coalesceRangeBounds(newPartEnd: Array[Int]): OrderedRVDPartitioner = {
    val newRangeBounds = (-1 +: newPartEnd.init).zip(newPartEnd).map { case (s, e) =>
        val i1 = rangeBounds(s + 1)
        val i2 = rangeBounds(e)
        Interval(i1.start, i2.end, i1.includesStart, i2.includesEnd)
      }
    copy(numPartitions = newPartEnd.length, rangeBounds = newRangeBounds)
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

      def getPartition(key: Any): Int = selfBc.value.getSafePartition(key)
    }
  }
}

object OrderedRVDPartitioner {
  def empty(typ: OrderedRVDType): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(typ.partitionKey, typ.kType, Array.empty[Interval])
  }

  // takes npartitions + 1 points and returns npartitions intervals: [a,b], (b,c], (c,d], ... (i, j]
  def makeRangeBoundIntervals(pType: Type, rangeBounds: Array[Any]): Array[Interval] = {
    var includesStart = true
    rangeBounds.zip(rangeBounds.tail).map { case (s, e) =>
      val i = Interval(s, e, includesStart, true)
      includesStart = false
      i
    }
  }
}
