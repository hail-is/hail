package is.hail.sparkextras

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import org.apache.spark.{Dependency, Partition, RangeDependency, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD

private[hail] class OriginUnionPartition(
  val index: Int,
  val originIdx: Int,
  val originPart: Partition,
) extends Partition

class OriginUnionRDD[T: ClassTag, S: ClassTag](
  sc: SparkContext,
  var rdds: IndexedSeq[RDD[T]],
  f: (Int, Int, Iterator[T]) => Iterator[S],
) extends RDD[S](sc, Nil) {
  override def getPartitions: Array[Partition] = {
    val arr = new Array[Partition](rdds.map(_.partitions.length).sum)
    var i = 0
    for {
      (rdd, rddIdx) <- rdds.zipWithIndex
      part <- rdd.partitions
    } {
      arr(i) = new OriginUnionPartition(i, rddIdx, part)
      i += 1
    }
    arr
  }

  override def getDependencies: Seq[Dependency[_]] = {
    val deps = new ArrayBuffer[Dependency[_]]
    var i = 0
    for (rdd <- rdds) {
      deps += new RangeDependency(rdd, 0, i, rdd.partitions.length)
      i += rdd.partitions.length
    }
    deps
  }

  override def compute(s: Partition, tc: TaskContext): Iterator[S] = {
    val p = s.asInstanceOf[OriginUnionPartition]
    f(p.originIdx, p.originPart.index, parent[T](p.originIdx).iterator(p.originPart, tc))
  }

  override def clearDependencies(): Unit = {
    super.clearDependencies()
    rdds = null
  }
}
