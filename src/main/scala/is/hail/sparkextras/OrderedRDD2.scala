package is.hail.sparkextras

import is.hail.annotations._
import is.hail.rvd.{OrderedRVD, OrderedRVDPartitioner, OrderedRVDType}
import is.hail.utils._
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

class OrderedDependency(left: OrderedRVD, right: OrderedRVD) extends NarrowDependency[RegionValue](right.rdd) {
  override def getParents(partitionId: Int): Seq[Int] =
    OrderedDependency.getDependencies(left.partitioner, right.partitioner)(partitionId)
}

object OrderedDependency {
  def getDependencies(p1: OrderedRVDPartitioner, p2: OrderedRVDPartitioner)(partitionId: Int): Seq[Int] = {
    val partBounds = p1.rangeBounds(partitionId).asInstanceOf[Interval]

    if (!p2.rangeTree.probablyOverlaps(p2.pkType.ordering, partBounds))
      Seq.empty[Int]
    else {
      val start = p2.getPartitionPK(partBounds.start)
      val end = p2.getPartitionPK(partBounds.end)
      start to end
    }
  }
}

case class OrderedJoinDistinctRDD2Partition(index: Int, leftPartition: Partition, rightPartitions: Array[Partition]) extends Partition

class OrderedJoinDistinctRDD2(left: OrderedRVD, right: OrderedRVD, joinType: String)
  extends RDD[JoinedRegionValue](left.sparkContext,
    Seq[Dependency[_]](new OneToOneDependency(left.rdd),
      new OrderedDependency(left, right))) {
  assert(joinType == "left" || joinType == "inner")
  override val partitioner: Option[Partitioner] = Some(left.partitioner)

  def getPartitions: Array[Partition] = {
    Array.tabulate[Partition](left.getNumPartitions)(i =>
      OrderedJoinDistinctRDD2Partition(i,
        left.partitions(i),
        OrderedDependency.getDependencies(left.partitioner, right.partitioner)(i)
          .map(right.partitions)
          .toArray))
  }

  override def getPreferredLocations(split: Partition): Seq[String] = left.rdd.preferredLocations(split)

  override def compute(split: Partition, context: TaskContext): Iterator[JoinedRegionValue] = {
    val partition = split.asInstanceOf[OrderedJoinDistinctRDD2Partition]

    val leftIt = left.rdd.iterator(partition.leftPartition, context)
    val rightIt = partition.rightPartitions.iterator.flatMap { p =>
      right.rdd.iterator(p, context)
    }

    joinType match {
      case "inner" => new OrderedInnerJoinDistinctIterator(left.typ, right.typ, leftIt, rightIt)
      case "left" => new OrderedLeftJoinDistinctIterator(left.typ, right.typ, leftIt, rightIt)
      case _ => fatal(s"Unknown join type `$joinType'. Choose from `inner' or `left'.")
    }
  }
}


class GenericOrderedZipJoinIterator[T, U](
  left: BufferedIterator[T], right: BufferedIterator[U],
  leftDefault: T, rightDefault: U, ord: (T, U) => Int)
    extends Iterator[Muple[T, U]] {

  val muple = Muple(leftDefault, rightDefault)

  def hasNext: Boolean = left.hasNext || right.hasNext

  def next(): Muple[T, U] = {
    val c = {
      if (left.hasNext) {
        if (right.hasNext)
          ord(left.head, right.head)
        else
          -1
      } else if (right.hasNext)
          1
      else {
        assert(!hasNext)
        throw new NoSuchElementException("next on empty iterator")
      }
    }
    if (c == 0) {
      muple._1 = left.next()
      muple._2 = right.next()
    } else if (c < 0) {
      muple._1 = left.next()
      muple._2 = rightDefault
    } else {
      // c > 0
      muple._1 = leftDefault
      muple._2 = right.next()
    }
    muple
  }
}

class OrderedZipJoinIterator(
  leftTyp: OrderedRVType, rightTyp: OrderedRVType,
  lit: Iterator[RegionValue], rit: Iterator[RegionValue])
    extends GenericOrderedZipJoinIterator[RegionValue, RegionValue](
  lit.buffered, rit.buffered, null, null,
  OrderedRVType.selectUnsafeOrdering(
    leftTyp.rowType, leftTyp.kRowFieldIdx, rightTyp.rowType, rightTyp.kRowFieldIdx).compare
)

case class OrderedZipJoinRDDPartition(index: Int, leftPartition: Partition, rightPartitions: Array[Partition]) extends Partition

class OrderedZipJoinRDD(left: OrderedRVD, right: OrderedRVD)
  extends RDD[JoinedRegionValue](left.sparkContext,
    Seq[Dependency[_]](new OneToOneDependency(left.rdd),
      new OrderedDependency(left, right))) {

  assert(left.partitioner.pkType.ordering.lteq(left.partitioner.minBound, right.partitioner.minBound) &&
    left.partitioner.pkType.ordering.gteq(left.partitioner.maxBound, right.partitioner.maxBound))

  private val leftPartitionForRightRow = new OrderedRVDPartitioner(
    right.typ.partitionKey,
    right.typ.rowType,
    left.partitioner.rangeBounds)

  override val partitioner: Option[Partitioner] = Some(left.partitioner)

  def getPartitions: Array[Partition] = {
    Array.tabulate[Partition](left.getNumPartitions)(i =>
      OrderedZipJoinRDDPartition(i,
        left.partitions(i),
        OrderedDependency.getDependencies(left.partitioner, right.partitioner)(i)
          .map(right.partitions)
          .toArray))
  }

  override def getPreferredLocations(split: Partition): Seq[String] = left.rdd.preferredLocations(split)

  override def compute(split: Partition, context: TaskContext): Iterator[JoinedRegionValue] = {
    val partition = split.asInstanceOf[OrderedZipJoinRDDPartition]
    val index = partition.index

    val leftIt = left.rdd.iterator(partition.leftPartition, context)
    val rightIt = partition.rightPartitions.iterator.flatMap { p =>
      right.rdd.iterator(p, context)
    }
      .filter { rrv => leftPartitionForRightRow.getPartition(rrv) == index }

    new OrderedZipJoinIterator(left.typ, right.typ, leftIt, rightIt)
  }
}
