package org.broadinstitute.hail.sparkextras

import org.apache.spark._
import org.apache.spark.rdd.RDD

case class OrderedOverlapPartition[PK](index: Int, parent: Partition, rightDeps: Seq[Partition],
  leftCutoff: Option[PK], rightCutoff: Option[PK]) extends Partition

class OrderedOuterJoinRDD[PK, K, V1, V2](left: OrderedRDD[PK, K, V1], right: OrderedRDD[PK, K, V2])
  extends RDD[(K, (Option[V1], Option[V2]))](left.sparkContext, Seq(new OneToOneDependency(left),
    new OrderedDependency(left.orderedPartitioner, right.orderedPartitioner, right)): Seq[Dependency[_]]) {

  override val partitioner = Some(left.orderedPartitioner)

  import left.kOk.kOrd
  import left.kOk.pkOrd
  import left.kOk.project

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
        if (leftIt.isEmpty) {
          val (k, v2) = rightIt.next()
          (k, (None, Some(v2)))
        } else if (rightIt.isEmpty) {
          val (k, v1) = leftIt.next()
          (k, (Some(v1), None))
        } else {
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
        }
      }
    }
  }
}