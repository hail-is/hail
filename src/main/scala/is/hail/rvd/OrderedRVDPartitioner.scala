package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.Partitioner

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

  require(rangeBounds.isEmpty || rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
    !left.overlaps(pkType.ordering, right) && pkType.ordering.compare(left.start, right.start) <= 0
  })

  require(rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
    pkType.ordering.equiv(left.end, right.start)})

  val rangeTree: IntervalTree[Int] = IntervalTree.fromSorted(pkType.ordering,
    Array.tabulate[(BaseInterval, Int)](numPartitions) { i =>
      (rangeBounds(i).asInstanceOf[Interval], i)
    })

  val pkKFieldIdx: Array[Int] = partitionKey.map(n => kType.fieldIdx(n))
  val pkKOrd: ExtendedOrdering = OrderedRVDType.selectExtendedOrdering(pkType, (0 until pkType.size).toArray, kType, pkKFieldIdx)

  val ordering: Ordering[Annotation] = pkType.ordering.toOrdering

  def region: Region = rangeBounds.region

  def loadElement(i: Int): Long = rangeBoundsType.loadElement(region, rangeBounds.aoff, rangeBounds.length, i)

  def loadStart(i: Int): Long = pkIntervalType.loadStart(region, loadElement(i))

  def loadEnd(i: Int): Long = pkIntervalType.loadStart(region, loadElement(i))

  // pk: Annotation[pkType]
  def getPartitionPK(pk: Any): Int = {
    assert(pkType.typeCheck(pk))
    val part = rangeTree.queryValues(pkType.ordering, pk)
    part match {
      case Array(x) => x
    }
  }

  // return the partition containing key
  // if outside bounds, throw error
  // key: RegionValue[kType]
  def getPartition(key: Any): Int = {
    val keyrv = key.asInstanceOf[RegionValue]
    val keyUR = new UnsafeRow(kType, keyrv)

    val part = rangeTree.queryValues(pkKOrd, keyUR)

    part match {
      case Array(x) => x
    }
  }

  def withKType(newPartitionKey: Array[String], newKType: TStruct): OrderedRVDPartitioner = {
    val newPart = new OrderedRVDPartitioner(newPartitionKey, newKType, rangeBounds)
    assert(newPart.pkType == pkType)
    newPart
  }

  def copy(numPartitions: Int = numPartitions, partitionKey: Array[String] = partitionKey,
    kType: TStruct = kType, rangeBounds: UnsafeIndexedSeq = rangeBounds): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(partitionKey, kType, rangeBounds)
  }

  def coalesceRangeBounds(newPartEnd: Array[Int]): OrderedRVDPartitioner = {
    val newRangeBounds = UnsafeIndexedSeq(
      rangeBoundsType,
      (-1 +: newPartEnd.init).zip(newPartEnd).map { case (s, e) =>
        val i1 = rangeBounds(s + 1).asInstanceOf[Interval]
        val i2 = rangeBounds(e).asInstanceOf[Interval]
        Interval(i1.start, i2.end, i1.includeStart, i2.includeEnd)
      })
    copy(numPartitions = newPartEnd.length, rangeBounds = newRangeBounds)
  }
}

object OrderedRVDPartitioner {
  def empty(typ: OrderedRVDType): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(typ.partitionKey, typ.kType, UnsafeIndexedSeq.empty(TArray(TInterval(typ.pkType))))
  }

  // takes npartitions + 1 points and returns npartitions intervals: [a,b], (b,c], (c,d], ... (i, j]
  def makeRangeBoundIntervals(pType: Type, rangeBounds: Array[RegionValue]): UnsafeIndexedSeq =
    makeRangeBoundIntervals(UnsafeIndexedSeq(TArray(pType), rangeBounds))

  def makeRangeBoundIntervals(rangeBounds: UnsafeIndexedSeq): UnsafeIndexedSeq = {
    val pType = rangeBounds.t.elementType
    val newT = TArray(TInterval(pType))

    val region = rangeBounds.region
    val rvb = new RegionValueBuilder(region)
    rvb.start(newT)
    rvb.startArray(rangeBounds.length - 1)
    var i = 0
    while (i < rangeBounds.length - 1) {
      rvb.startStruct()
      rvb.addAnnotation(pType, rangeBounds(i))
      rvb.addAnnotation(pType, rangeBounds(i + 1))
      rvb.addBoolean(if (i == 0) true else false)
      rvb.addBoolean(true)
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    new UnsafeIndexedSeq(newT, region, rvb.end())
  }
}
