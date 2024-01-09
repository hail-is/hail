package is.hail.sparkextras

import scala.reflect.ClassTag

import org.apache.spark.{OneToOneDependency, Partition, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD

object MultiWayZipPartitionsRDD {
  def apply[T: ClassTag, V: ClassTag](
    rdds: IndexedSeq[RDD[T]]
  )(
    f: (Array[Iterator[T]]) => Iterator[V]
  ): MultiWayZipPartitionsRDD[T, V] =
    new MultiWayZipPartitionsRDD(rdds.head.sparkContext, rdds, f)
}

private case class MultiWayZipPartition(val index: Int, val partitions: IndexedSeq[Partition])
    extends Partition

class MultiWayZipPartitionsRDD[T: ClassTag, V: ClassTag](
  sc: SparkContext,
  var rdds: IndexedSeq[RDD[T]],
  var f: (Array[Iterator[T]]) => Iterator[V],
) extends RDD[V](sc, rdds.map(x => new OneToOneDependency(x))) {
  require(rdds.length > 0)
  private val numParts = rdds(0).partitions.length
  require(rdds.forall(rdd => rdd.partitions.length == numParts))

  override val partitioner = None

  override def getPartitions: Array[Partition] =
    Array.tabulate[Partition](numParts) { i =>
      MultiWayZipPartition(i, rdds.map(rdd => rdd.partitions(i)))
    }

  override def compute(s: Partition, tc: TaskContext) = {
    val partitions = s.asInstanceOf[MultiWayZipPartition].partitions
    val arr = Array.tabulate(rdds.length)(i => rdds(i).iterator(partitions(i), tc))
    f(arr)
  }

  override def clearDependencies() {
    super.clearDependencies
    rdds = null
    f = null
  }
}
