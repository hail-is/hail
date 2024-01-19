package is.hail.sparkextras

import scala.annotation.meta.param
import scala.reflect.ClassTag

import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD

case class MapPartitionsWithValueRDDPartition[V](
  parentPartition: Partition,
  value: V,
) extends Partition {
  def index: Int = parentPartition.index
}

class MapPartitionsWithValueRDD[T: ClassTag, U: ClassTag, V](
  var prev: RDD[T],
  @(transient @param) values: Array[V],
  f: (Int, V, Iterator[T]) => Iterator[U],
  preservesPartitioning: Boolean,
) extends RDD[U](prev) {

  @transient override val partitioner =
    if (preservesPartitioning) firstParent[T].partitioner else None

  override def getPartitions: Array[Partition] =
    firstParent[T].partitions.map(p => MapPartitionsWithValueRDDPartition(p, values(p.index)))

  override def compute(split: Partition, context: TaskContext): Iterator[U] = {
    val p = split.asInstanceOf[MapPartitionsWithValueRDDPartition[V]]
    f(split.index, p.value, firstParent[T].iterator(p.parentPartition, context))
  }

  override def clearDependencies(): Unit = {
    super.clearDependencies()
    prev = null
  }
}
