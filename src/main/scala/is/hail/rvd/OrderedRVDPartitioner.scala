package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.Partitioner

class OrderedRVDPartitioner(
  val numPartitions: Int,
  val partitionKey: Array[String], val kType: TStruct,
  // rangeBounds is partition max, sorted ascending
  // rangeBounds: Array[pkType]
  val rangeBounds: UnsafeIndexedSeq) extends Partitioner {
  require((numPartitions == 0 && rangeBounds.isEmpty) || numPartitions == rangeBounds.length + 1,
    s"nPartitions = $numPartitions, ranges = ${ rangeBounds.length }")

  val (pkType, _) = kType.select(partitionKey)

  val pkKFieldIdx: Array[Int] = partitionKey.map(n => kType.fieldIdx(n))
  val pkKOrd: UnsafeOrdering = OrderedRVDType.selectUnsafeOrdering(pkType, (0 until pkType.size).toArray, kType, pkKFieldIdx)

  val rangeBoundsType = TArray(pkType)
  assert(rangeBoundsType.typeCheck(rangeBounds))

  val ordering: Ordering[Annotation] = pkType.ordering.toOrdering
  require(rangeBounds.isEmpty || rangeBounds.zip(rangeBounds.tail).forall { case (left, right) => ordering.compare(left, right) < 0 })

  def region: Region = rangeBounds.region

  def loadElement(i: Int): Long = rangeBoundsType.loadElement(rangeBounds.region, rangeBounds.aoff, rangeBounds.length, i)

  // return the smallest partition for which key <= max
  // pk: Annotation[pkType]
  def getPartitionPK(pk: Any): Int = {
    assert(pkType.typeCheck(pk))

    val part = BinarySearch.binarySearch(numPartitions,
      // key.compare(elem)
      i =>
        if (i == numPartitions - 1)
          -1 // key.compare(inf)
        else
          ordering.compare(pk, rangeBounds(i)))
    part
  }

  // return the smallest partition for which key <= max
  // key: RegionValue[kType]
  def getPartition(key: Any): Int = {
    val keyrv = key.asInstanceOf[RegionValue]

    val part = BinarySearch.binarySearch(numPartitions,
      // key.compare(elem)
      i =>
        if (i == numPartitions - 1)
          -1 // key.compare(inf)
        else
          -pkKOrd.compare(rangeBounds.region, loadElement(i), keyrv))
    part
  }

  def withKType(newPartitionKey: Array[String], newKType: TStruct): OrderedRVDPartitioner = {
    val newPart = new OrderedRVDPartitioner(numPartitions, newPartitionKey, newKType, rangeBounds)
    assert(newPart.pkType == pkType)
    newPart
  }
}

object OrderedRVDPartitioner {
  def empty(typ: OrderedRVDType): OrderedRVDPartitioner = {
    new OrderedRVDPartitioner(0, typ.partitionKey, typ.kType, UnsafeIndexedSeq.empty(TArray(typ.pkType)))
  }
}
