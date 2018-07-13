package is.hail.utils.richUtils

import java.io._

import is.hail.rvd.RVDContext
import org.apache.spark.TaskContext
import is.hail.utils._
import is.hail.sparkextras._

import scala.reflect.ClassTag

class RichContextRDD[T: ClassTag](crdd: ContextRDD[RVDContext, T]) {
  def writePartitions(path: String,
    stageLocally: Boolean,
    write: (RVDContext, Iterator[T], OutputStream) => Long): (Array[String], Array[Long]) = {
    val sc = crdd.sparkContext
    val hadoopConf = sc.hadoopConfiguration

    hadoopConf.mkDir(path + "/parts")

    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))

    val nPartitions = crdd.getNumPartitions

    val d = digitsNeeded(nPartitions)

    val (partFiles, partitionCounts) = crdd.cmapPartitionsWithIndex { case (i, ctx, it) =>
      val f = partFile(d, i, TaskContext.get)
      val finalFilename = path + "/parts/" + f
      val filename =
        if (stageLocally) {
          val context = TaskContext.get
          val partPath = hadoopConf.getTemporaryFile("file:///tmp")
          context.addTaskCompletionListener { context =>
            hadoopConf.delete(partPath, recursive = false)
          }
          partPath
        } else
          finalFilename
      val os = sHadoopConfBc.value.value.unsafeWriter(filename)
      val count = write(ctx, it, os)
      if (stageLocally)
        hadoopConf.copy(filename, finalFilename)
      ctx.region.clear()
      Iterator.single(f -> count)
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
