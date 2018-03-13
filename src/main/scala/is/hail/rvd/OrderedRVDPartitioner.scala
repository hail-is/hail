package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.broadcast.Broadcast

class OrderedRVDPartitioner(
  val partitionKey: Array[String], val kType: TStruct,
  // rangeBounds: Array[Interval[pkType]]
  // rangeBounds is interval containing all partition keys within a partition
  val rangeBounds: UnsafeIndexedSeq) extends Partitioner {
  val numPartitions: Int = rangeBounds.length

  val (pkType, _) = kType.select(partitionKey)
  val pkIntervalType = TInterval(pkType)
  val rangeBoundsType = TArray(pkIntervalType)

  assert(rangeBoundsType.typeCheck(rangeBounds))

  require(rangeBounds.isEmpty || (rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
    !left.mayOverlap(pkType.ordering, right) && pkType.ordering.lteq(left.start, right.start)
  } && rangeBounds.forall { case i: Interval =>
    pkType.ordering.lteq(i.start, i.end) && !i.definitelyEmpty(pkType.ordering)
  }))

  require(rangeBounds.isEmpty || rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
    pkType.ordering.equiv(left.end, right.start) && (left.includesEnd || right.includesStart) } )

  val rangeTree: IntervalTree[Int] = IntervalTree.fromSorted(pkType.ordering,
    Array.tabulate[(Interval, Int)](numPartitions) { i =>
      (rangeBounds(i).asInstanceOf[Interval], i)
    })

  val pkKFieldIdx: Array[Int] = partitionKey.map(n => kType.fieldIdx(n))

  def region: Region = rangeBounds.region

  def loadElement(i: Int): Long = rangeBoundsType.loadElement(region, rangeBounds.aoff, rangeBounds.length, i)

  def loadStart(i: Int): Long = pkIntervalType.loadStart(region, loadElement(i))

  def loadEnd(i: Int): Long = pkIntervalType.loadStart(region, loadElement(i))

  def range: Interval = rangeTree.root.get.range

  // if outside bounds, return min or max depending on location
  // pk: Annotation[pkType]
  def getPartitionPK(
    pk: Any,
    resolveAmbiguity: Int = OrderedRVDPartitioner.UNAMBIGUOUS
  ): Int = {
    val part = rangeTree.queryValues(pkType.ordering, pk)
    part match {
      case Array() =>
        if (range.isAbovePosition(pkType.ordering, pk))
          0
        else {
          assert(range.isBelowPosition(pkType.ordering, pk))
          numPartitions - 1
        }

      case Array(x) => x

      case parts => {
        assert(resolveAmbiguity != OrderedRVDPartitioner.UNAMBIGUOUS)
        if (resolveAmbiguity == OrderedRVDPartitioner.SMALLEST)
          parts.min
        else
          parts.max
      }
    }
  }

  // Return the sequence of partition IDs overlapping the given interval of
  // partition keys.
  def getPartitionRangePK(pkInterval: Interval): Seq[Int] = {
    if (!rangeTree.probablyOverlaps(pkType.ordering, pkInterval))
      Seq.empty[Int]
    else {
      val start = getPartitionPK(pkInterval.start, OrderedRVDPartitioner.SMALLEST)
      val end = getPartitionPK(pkInterval.end, OrderedRVDPartitioner.LARGEST)
      start to end
    }
  }

  // return the partition containing key
  // if outside bounds, return min or max depending on location
  // key: RegionValue[kType]
  def getPartition(key: Any): Int = {
    val keyrv = key.asInstanceOf[RegionValue]
    val kUR = new UnsafeRow(kType, keyrv)
    val pkUR = new KeyedRow(kUR, pkKFieldIdx)

    val part = rangeTree.queryValues(pkType.ordering, pkUR)

    part match {
      case Array() =>
        if (range.isAbovePosition(pkType.ordering, pkUR))
          0
        else {
          assert(range.isBelowPosition(pkType.ordering, pkUR))
          numPartitions - 1
        }
      case Array(x) => x
    }
  }

  def withKType(newPartitionKey: Array[String], newKType: TStruct): OrderedRVDPartitioner = {
    val (newPKType, _) = newKType.select(newPartitionKey)
    val newRangeBounds = new UnsafeIndexedSeq(TArray(TInterval(newPKType)), rangeBounds.region, rangeBounds.aoff)
    val newPart = new OrderedRVDPartitioner(newPartitionKey, newKType, newRangeBounds)
    assert(newPart.pkType.types.sameElements(pkType.types))
    newPart
  }

  def copy(numPartitions: Int = numPartitions, partitionKey: Array[String] = partitionKey,
    kType: TStruct = kType, rangeBounds: UnsafeIndexedSeq = rangeBounds): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(partitionKey, kType, rangeBounds)
  }

  def enlargeToRange(newRange: Interval): OrderedRVDPartitioner = {
    val newStart = pkType.ordering.min(range.start, newRange.start)
    val newEnd = pkType.ordering.max(range.end, newRange.end)
    val newRangeBounds = rangeBounds.toArray
    newRangeBounds(0) = newRangeBounds(0).asInstanceOf[Interval]
      .copy(start = newStart, includeStart = true)
    newRangeBounds(newRangeBounds.length - 1) = newRangeBounds(newRangeBounds.length - 1)
      .asInstanceOf[Interval].copy(end = newEnd, includeEnd = true)
    copy(rangeBounds = UnsafeIndexedSeq(rangeBoundsType, newRangeBounds))
  }

  def coalesceRangeBounds(newPartEnd: Array[Int]): OrderedRVDPartitioner = {
    val newRangeBounds = UnsafeIndexedSeq(
      rangeBoundsType,
      (-1 +: newPartEnd.init).zip(newPartEnd).map { case (s, e) =>
        val i1 = rangeBounds(s + 1).asInstanceOf[Interval]
        val i2 = rangeBounds(e).asInstanceOf[Interval]
        Interval(i1.start, i2.end, i1.includesStart, i2.includesEnd)
      })
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

      def getPartition(key: Any): Int = selfBc.value.getPartition(key)
    }
  }
}

object OrderedRVDPartitioner {
  val UNAMBIGUOUS: Int = 0
  val SMALLEST: Int = -1
  val LARGEST: Int = 1

  def empty(typ: OrderedRVDType): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(typ.partitionKey, typ.kType, UnsafeIndexedSeq.empty(TArray(TInterval(typ.pkType))))
  }

  // takes npartitions + 1 points and returns npartitions intervals: [a,b], (b,c], (c,d], ... (i, j]
  def makeRangeBoundIntervals(pType: Type, rangeBounds: Array[RegionValue]): UnsafeIndexedSeq = {
    val uisRangeBounds = UnsafeIndexedSeq(TArray(pType), rangeBounds)
    var includesStart = true
    val rangeBoundIntervals = uisRangeBounds.zip(uisRangeBounds.tail).map { case (s, e) =>
        val i = Interval(s, e, includesStart, true)
        includesStart = false
        i
    }
    UnsafeIndexedSeq(TArray(TInterval(pType)), rangeBoundIntervals)
  }
}
