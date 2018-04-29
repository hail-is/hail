package is.hail.utils.richUtils

import java.io._

import is.hail.rvd.RVDContext
import org.apache.spark.TaskContext
import is.hail.utils._
import is.hail.sparkextras._

import scala.reflect.ClassTag

class RichContextRDD[T: ClassTag](crdd: ContextRDD[RVDContext, T]) {
  def writePartitions(path: String,
    write: (RVDContext, Iterator[T], OutputStream) => Long): (Array[String], Array[Long]) = {
    val sc = crdd.sparkContext
    val hadoopConf = sc.hadoopConfiguration

    hadoopConf.mkDir(path + "/parts")

    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))

    val nPartitions = crdd.getNumPartitions

    val d = digitsNeeded(nPartitions)

    val (partFiles, partitionCounts) = crdd.cmapPartitionsWithIndex { case (i, ctx, it) =>
      val f = partFile(d, i, TaskContext.get)
      val filename = path + "/parts/" + f
      val os = sHadoopConfBc.value.value.unsafeWriter(filename)
      val out = Iterator.single(f -> write(ctx, it, os))
      ctx.region.clear()
      out
    }
      .collect()
      .unzip

    val itemCount = partitionCounts.sum
    assert(nPartitions == partitionCounts.length)

    info(s"wrote $itemCount ${ plural(itemCount, "item") } " +
      s"in ${ nPartitions } ${ plural(nPartitions, "partition") }")

    (partFiles, partitionCounts)
  }
}
