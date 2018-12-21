package is.hail.sparkextras

import org.apache.spark.{Dependency, Partition, RangeDependency, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

private[hail] case class OriginUnionPartition[T: ClassTag](
  val index: Int,
  val originIdx: Int,
  val originPart: Partition
) extends Partition {
}

class OriginUnionRDD[T: ClassTag](
  sc: SparkContext,
  var rdds: IndexedSeq[RDD[T]]
) extends RDD[T](sc, Nil) {
  override def getPartitions: Array[Partition] = {
    val arr = new Array[Partition](rdds.map(_.partitions.length).sum)
    var i = 0
    for ((rdd, rddIdx) <- rdds.zipWithIndex; part <- rdd.partitions) {
      arr(i) = OriginUnionPartition(i, rddIdx, part)
      i += 1
    }
    arr
  }

  override def getDependencies(): Seq[Dependency[_]] = {
    val deps = new ArrayBuffer[Dependency[_]]
    var i = 0
    for (rdd <- rdds) {
      deps += new RangeDependency(rdd, 0, i, rdd.partitions.length)
      i += rdd.partitions.length
    }
    deps
  }

  override def compute(s: Partition, tc: TaskContext) = {
    val p = s.asInstanceOf[OriginUnionPartition[T]]
    parent[T](p.originIdx).iterator(p.originPart, tc)
  }

  override def clearDependencies() {
    super.clearDependencies()
    rdds = null
  }
}
