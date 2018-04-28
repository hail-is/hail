package is.hail.sparkextras

import org.apache.spark._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

case class GeneralRDDPartition(
  partitionInputs: Array[(Int, Partition)],
  index: Int
) extends Partition

class GeneralRDD[T: ClassTag](
  sc: SparkContext,
  rdds: Array[RDD[T]],
  inputs: Array[Array[(Int, Int)]]
) extends RDD[Array[Iterator[T]]](sc, Nil) {

  override def getPartitions: Array[Partition] = {
    // Do not call getPartitions here!
    val parentPartitions = rdds.map(_.partitions)

    inputs.map { _.map { case (rddIndex, partitionIndex) =>
      (rddIndex, parentPartitions(rddIndex)(partitionIndex)) }
    } .zipWithIndex
      .map { case (x, i) => new GeneralRDDPartition(x, i) }
  }

  override def compute(
    split: Partition,
    context: TaskContext
  ): Iterator[Array[Iterator[T]]] = {
    // Do not call partitions or getPartitions here!
    val gp = split.asInstanceOf[GeneralRDDPartition]
    Iterator.single(
      gp.partitionInputs.map { case (rddIndex, partition) =>
        rdds(rddIndex).iterator(partition, context) } )
  }

  override def getDependencies: Seq[Dependency[_]] = {
    inputs
      .zipWithIndex
      .flatMap { case (input, i) =>
        input.map { case (rddIndex, partitionIndex) => (rddIndex, partitionIndex, i) } }
      .groupBy(_._1)
      .map { case (rddIndex, x) =>
        new NarrowDependency[T](rdds(rddIndex)) {
          override def getParents(partitionId: Int): Seq[Int] =
            x.filter(partitionId == _._3).map(_._2)
        }
      }.toSeq
  }
}
