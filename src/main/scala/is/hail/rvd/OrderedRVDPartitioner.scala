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
    kType.relaxedTypeCheck(l) && kType.relaxedTypeCheck(r)
  })

  require(rangeBounds.isEmpty || rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
    left.isBelow(pkType.ordering, right)
  })

  val rangeTree: IntervalTree[Int] = IntervalTree.fromSorted(kType.ordering,
    Array.tabulate[(Interval, Int)](numPartitions) { i =>
      (rangeBounds(i), i)
    })

  def coarsenedPKRangeBounds(newPK: Int): IndexedSeq[Interval] = {
    rangeBounds.map{ i =>
      val start = i.start.asInstanceOf[Row]
      val startLen = start.size
      val end = i.end.asInstanceOf[Row]
      val endLen = end.size
      Interval(
        if (startLen > newPK)
          Row.fromSeq(start.toSeq.take(newPK))
        else
          start,
        if (endLen > newPK)
          Row.fromSeq(end.toSeq.take(newPK))
        else
          end,
        includesStart = startLen > newPK || i.includesStart,
        includesEnd = endLen > newPK || i.includesEnd
      )}
  }

  def range: Option[Interval] = rangeTree.root.map(_.range)

  def contains(index: Int, key: Any): Boolean = {
    require(kType.isComparableAt(key))
    rangeBounds(index).contains(kType.ordering, key)
  }

  // Return the sequence of partition IDs overlapping the given interval of
  // partition keys.
  def getPartitionRange(query: Interval): Seq[Int] = {
    require(kType.isComparableAt(query.start) && kType.isComparableAt(query.end))
    if (!rangeTree.probablyOverlaps(kType.ordering, query))
      FastSeq.empty[Int]
    else
      rangeTree.queryOverlappingValues(kType.ordering, query)
  }

  // Get greatest partition ID whose lower bound is less than 'key'. Returns -1
  // if all partitions are above 'key'.
  def getSafePartitionLowerBound(key: Any): Int = {
    require(rangeBounds.nonEmpty)
    require(kType.isComparableAt(key))

    val range = getPartitionRange(Interval(Row(), key, true, true))
    range.lastOption.getOrElse(-1)
  }

  // Get least partition ID whose upper bound is greater than 'key'. Returns
  // numPartitions if all partitions are below 'key'.
  def getSafePartitionUpperBound(key: Any): Int = {
    require(rangeBounds.nonEmpty)
    require(kType.isComparableAt(key))

    val range = getPartitionRange(Interval(key, Row(), true, true))
    range.headOption.getOrElse(numPartitions)
  }

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

      def getPartition(key: Any): Int = selfBc.value.getSafePartitionLowerBound(key)
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

  // takes range bounds from n partitioners and splits them such that each
  // resulting partition only comes from one partition per original partitioner.
  def mergePartitioners(p1: OrderedRVDPartitioner, p2: OrderedRVDPartitioner): OrderedRVDPartitioner = {
    require(p1.kType == p2.kType)
    if (p1.range.isEmpty)
      return p2
    if (p2.range.isEmpty)
      return p1

    val ord = p1.kType.ordering
    def cmp(p1: (Any, Boolean), p2: (Any, Boolean)): Int = {
      val c = ord.compare(p1._1, p2._1)
      if (c != 0)
        c
      else if (p1._2 == p2._2)
        0
      else if (p1._2)
        -1
      else 1
    }

    val bounds1 = p1.rangeBounds.map { i => i.start -> i.includesStart } :+ p1.rangeBounds.last.end -> !p1.rangeBounds.last.includesEnd
    val bounds2 = p2.rangeBounds.map { i => i.start -> i.includesStart } :+ p2.rangeBounds.last.end -> !p2.rangeBounds.last.includesEnd

    val boundsAB = new ArrayBuilder[(Any, Boolean)]()

    var i = 0
    var j = 0
    while (i < bounds1.length && j < bounds2.length) {
      val newBound = if (cmp(bounds1(i), bounds2(j)) <= 0)
        bounds1(i)
      else
        bounds2(j)

      while (i < bounds1.length && cmp(newBound, bounds1(i)) == 0) {
        i += 1
      }
      while (j < bounds2.length && cmp(newBound, bounds2(j)) == 0) {
        j += 1
      }
      boundsAB += newBound
    }

    while (i < bounds1.length) {
      boundsAB += bounds1(i)
      i += 1
    }

    while (j < bounds2.length) {
      boundsAB += bounds2(j)
      j += 1
    }

    val bounds = boundsAB.result()

    val newBounds = bounds.zip(bounds.tail).map { case ((a1, i1), (a2, i2)) =>
      Interval(a1, a2, i1, !i2)
    }
    p1.copy(numPartitions = newBounds.length, rangeBounds = newBounds)
  }
}
