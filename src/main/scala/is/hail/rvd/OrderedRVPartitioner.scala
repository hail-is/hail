package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.expr.{JSONAnnotationImpex, Parser}
import is.hail.utils._
import org.apache.spark.{Partitioner, SparkContext}
import org.json4s.JsonAST._

class OrderedRVPartitioner(
  val numPartitions: Int,
  val partitionKey: Array[String], val kType: TStruct,
  // rangeBounds is partition max, sorted ascending
  // rangeBounds: Array[pkType]
  val rangeBounds: UnsafeIndexedSeq) extends Partitioner {
  require((numPartitions == 0 && rangeBounds.isEmpty) || numPartitions == rangeBounds.length + 1,
    s"nPartitions = $numPartitions, ranges = ${ rangeBounds.length }")

  val (pkType, _) = kType.select(partitionKey)

  val pkKFieldIdx: Array[Int] = partitionKey.map(n => kType.fieldIdx(n))
  val pkKOrd: UnsafeOrdering = OrderedRVType.selectUnsafeOrdering(pkType, (0 until pkType.size).toArray, kType, pkKFieldIdx)

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

  def toJSON: JValue =
    JObject(List(
      "numPartitions" -> JInt(numPartitions),
      "partitionKey" -> JArray(partitionKey.map(n => JString(n)).toList),
      "kType" -> JString(kType.toPrettyString(compact = true)),
      "rangeBounds" -> JSONAnnotationImpex.exportAnnotation(rangeBounds, rangeBoundsType)))

  def withKType(newPartitionKey: Array[String], newKType: TStruct): OrderedRVPartitioner = {
    val newPart = new OrderedRVPartitioner(numPartitions, newPartitionKey, newKType, rangeBounds)
    assert(newPart.pkType == pkType)
    newPart
  }
}

object OrderedRVPartitioner {
  def empty(typ: OrderedRVType): OrderedRVPartitioner = {
    new OrderedRVPartitioner(0, typ.partitionKey, typ.kType, UnsafeIndexedSeq.empty(TArray(typ.pkType)))
  }

  def apply(sc: SparkContext, jv: JValue): OrderedRVPartitioner = {
    case class Extract(numPartitions: Int,
      partitionKey: Array[String],
      kType: String,
      rangeBounds: JValue)
    val ex = jv.extract[Extract]

    val partitionKey = ex.partitionKey
    val kType = Parser.parseType(ex.kType).asInstanceOf[TStruct]
    val (pkType, _) = kType.select(partitionKey)

    val rangeBoundsType = TArray(pkType)
    new OrderedRVPartitioner(ex.numPartitions,
      ex.partitionKey,
      kType,
      UnsafeIndexedSeq(
        rangeBoundsType,
        JSONAnnotationImpex.importAnnotation(ex.rangeBounds, rangeBoundsType).asInstanceOf[IndexedSeq[Annotation]]))
  }
}
