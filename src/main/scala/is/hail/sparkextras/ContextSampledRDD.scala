package is.hail.sparkextras

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.util.random.RandomSampler

import java.util.Random

import scala.reflect.ClassTag

class SeededPartition(
  val prev: Partition,
  val seed: Long
) extends Partition with Serializable {
  override val index: Int = prev.index
}

// FIXME: this doesn't feel like it should be necessary, can't we do this by
// zipPartitions with a seed and then mapPartitions?
class ContextSampledRDD[C, T: ClassTag, U: ClassTag](
  prev: RDD[C => Iterator[T]],
  sampler: RandomSampler[T, U],
  @transient private val seed: Long
) extends RDD[C => Iterator[U]](prev) {

  @transient override val partitioner = prev.partitioner

  override def getPartitions: Array[Partition] = {
    val random = new Random(seed)
    prev.partitions.map(x => new SeededPartition(x, random.nextLong()))
  }

  override def getPreferredLocations(split: Partition): Seq[String] =
    prev.preferredLocations(split.asInstanceOf[SeededPartition].prev)

  override def compute(
    p: Partition,
    context: TaskContext
  ): Iterator[C => Iterator[U]] = {
    val sp = p.asInstanceOf[SeededPartition]
    val s = sampler.clone
    s.setSeed(sp.seed)

    Iterator.single {
      (c: C) => s.sample(prev.iterator(sp.prev, context).flatMap(_(c))) }
  }
}
