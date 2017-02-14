package is.hail.sparkextras

import org.apache.spark._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

case class OrderedOverlapPartition[PK](index: Int, parent: Partition, rightDeps: Seq[Partition],
  leftCutoff: Option[PK], rightCutoff: Option[PK]) extends Partition

object OrderedOuterJoinRDD {
  def apply[PK, K, V1, V2](left: OrderedRDD[PK, K, V1], right: OrderedRDD[PK, K, V2])
    (implicit vct1: ClassTag[V1], vct2: ClassTag[V2]): RDD[(K, (Option[V1], Option[V2]))] = {
    import left.kOk
    import left.kOk._

    val nLeftPartitions = left.partitions.length
    val nRightPartitions = right.partitions.length

    val rdd: RDD[(K, (Option[V1], Option[V2]))] = if (nLeftPartitions == 0 && nRightPartitions == 0)
      OrderedRDD.empty[PK, K, (Option[V1], Option[V2])](left.sparkContext)
    else if (nLeftPartitions == 0)
      right.mapValues(v => (None, Some(v)))
    else if (nRightPartitions == 0)
      left.mapValues(v => (Some(v), None))
    else if (nRightPartitions > nLeftPartitions) {
      val join = new OrderedOuterJoinRDD(right, left)
      join.mapValues { case (v1, v2) => (v2, v1) }
    } else {
      assert(nLeftPartitions >= nRightPartitions)
      new OrderedOuterJoinRDD(left, right)
    }
    assert(rdd.partitions.length == math.max(nLeftPartitions, nRightPartitions))
    rdd
  }
}

class OrderedOuterJoinRDD[PK, K, V1, V2] private(left: OrderedRDD[PK, K, V1], right: OrderedRDD[PK, K, V2])
  extends RDD[(K, (Option[V1], Option[V2]))](left.sparkContext, Seq(new OneToOneDependency(left),
    new OrderedDependency(left.orderedPartitioner, right.orderedPartitioner, right)): Seq[Dependency[_]]) {
  require(left.partitions.length > 0 && right.partitions.length > 0)

  override val partitioner = Some(left.orderedPartitioner)

  import left.kOk.{kOrd, pkOrd, project}

  import Ordering.Implicits._


  def getPartitions: Array[Partition] = {
    val r = OrderedDependency.getDependencies(left.orderedPartitioner, right.orderedPartitioner)(_)
    val rangeBounds = left.orderedPartitioner.rangeBounds
    left.partitions.map { p =>
      val i = p.index
      val (rightStart, rightEnd) = r(i)
      OrderedOverlapPartition(i, p, (rightStart to rightEnd).map(right.partitions),
        if (i == 0) None else Some(rangeBounds(i - 1)), if (i == rangeBounds.length) None else Some(rangeBounds(i)))
    }
  }

  override def getPreferredLocations(split: Partition): Seq[String] = left.preferredLocations(split)

  override def compute(split: Partition, context: TaskContext): Iterator[(K, (Option[V1], Option[V2]))] = {
    val oop = split.asInstanceOf[OrderedOverlapPartition[PK]]
    val leftCutoff = oop.leftCutoff
    val rightCutoff = oop.rightCutoff
    val leftIt = left.iterator(oop.parent, context).buffered
    val rightIt = oop.rightDeps
      .iterator
      .flatMap(p => right.iterator(p, context))
      .dropWhile { case (k, _) => leftCutoff.exists(_ >= project(k)) }
      .buffered

    new Iterator[(K, (Option[V1], Option[V2]))] {

      def hasNext: Boolean = leftIt.hasNext ||
        (rightIt.hasNext && rightCutoff.forall(pk => project(rightIt.head._1) <= pk))

      def next(): (K, (Option[V1], Option[V2])) = {
        if (rightIt.hasNext) {
          if (leftIt.hasNext) {
            val leftK = leftIt.head._1
            val rightK = rightIt.head._1
            if (leftK == rightK) {
              val (_, v1) = leftIt.next()
              val (_, v2) = rightIt.next()
              (leftK, (Some(v1), Some(v2)))
            } else if (leftK < rightK) {
              val (_, v1) = leftIt.next()
              (leftK, (Some(v1), None))
            } else {
              val (_, v2) = rightIt.next()
              (rightK, (None, Some(v2)))
            }
          } else {
            val (k, v2) = rightIt.next()
            (k, (None, Some(v2)))
          }
        } else {
          val (k, v1) = leftIt.next()
          (k, (Some(v1), None))
        }
      }
    }
  }
}