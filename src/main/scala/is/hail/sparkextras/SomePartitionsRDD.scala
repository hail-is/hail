package is.hail.sparkextras

import org.apache.spark.rdd.RDD
import org.apache.spark._
import is.hail.utils._

import scala.reflect.ClassTag

class SomePartitionsRDD[T](prev: RDD[T], val keep: Array[Int])(implicit tct: ClassTag[T])
  extends RDD[T](prev.sparkContext, Nil) {
  require(keep.nonEmpty && keep.isIncreasing && keep.head >= 0 && keep.last < prev.getNumPartitions)
  
  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(prev) {
      def getParents(partitionId: Int): Seq[Int] = Seq(keep(partitionId))
    })

  def compute(split: Partition, context: TaskContext): Iterator[T] = {
    prev.iterator(prev.partitions(split.asInstanceOf[SomePartitionsRDDPartition].prevIndex), context)
  }

  protected def getPartitions: Array[Partition] =
    Array.tabulate(keep.length) (i => new SomePartitionsRDDPartition(i, keep(i)))
  
  @transient override val partitioner: Option[Partitioner] = prev.partitioner
}

class SomePartitionsRDDPartition(val index: Int, val prevIndex: Int) extends Partition