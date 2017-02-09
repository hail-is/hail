package is.hail.utils.richUtils

import is.hail.sparkextras.{OrderedKey, OrderedPartitioner, OrderedRDD}
import is.hail.utils._
import is.hail.variant.Variant
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

case class RichVariantPairRDD[T](rdd: RDD[(Variant, T)]) extends AnyVal {

  def smartShuffleAndSort[PK](partitioner: OrderedPartitioner[PK, Variant], maxShift: Int)
    (implicit kOk: OrderedKey[PK, Variant], ct: ClassTag[T]): OrderedRDD[PK, Variant, T] = {

    val partitionerBc = rdd.sparkContext.broadcast(partitioner)

    // stay is sorted within partitions by a local sort
    val stay = rdd.mapPartitionsWithIndex(
      { case (i, it) =>
        LocalVariantSortIterator(
          it.filter { case (k, _) => partitionerBc.value.getPartition(k) == i },
          maxShift)
      }, preservesPartitioning = true)

    // move is sorted within partitions by shuffle
    val move = rdd.mapPartitionsWithIndex { case (i, it) =>
      it.filter { case (k, _) => partitionerBc.value.getPartition(k) != i }
    }.orderedRepartitionBy(partitioner)

    OrderedRDD.partitionedSortedUnion(stay, move, partitioner)
  }
}
