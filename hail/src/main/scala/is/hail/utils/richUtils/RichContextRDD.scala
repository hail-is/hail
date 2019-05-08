package is.hail.utils.richUtils

import java.io._

import is.hail.HailContext
import is.hail.rvd.RVDContext
import org.apache.spark.TaskContext
import is.hail.utils._
import is.hail.sparkextras._
import org.apache.spark.rdd.RDD
import is.hail.io.fs.HadoopFS

import scala.reflect.ClassTag

class RichContextRDD[T: ClassTag](crdd: ContextRDD[RVDContext, T]) {
  // Only use on CRDD's whose T is not dependent on the context
  def clearingRun: RDD[T] =
    crdd.cmap { (ctx, v) =>
      ctx.region.clear()
      v
    }.run

  def writePartitions(path: String,
    stageLocally: Boolean,
    write: (RVDContext, Iterator[T], OutputStream) => Long): (Array[String], Array[Long]) = {
    val sc = crdd.sparkContext
    val fs = new HadoopFS(sc.hadoopConfiguration)

    fs.mkDir(path + "/parts")

    val bcFS = HailContext.bcFS

    val nPartitions = crdd.getNumPartitions

    val d = digitsNeeded(nPartitions)

    val (partFiles, partitionCounts) = crdd.cmapPartitionsWithIndex { (i, ctx, it) =>
      val fs = bcFS.value
      val f = partFile(d, i, TaskContext.get)
      val finalFilename = path + "/parts/" + f
      val filename =
        if (stageLocally) {
          val context = TaskContext.get
          val partPath = fs.getTemporaryFile("file:///tmp")
          context.addTaskCompletionListener { (context: TaskContext) =>
            fs.delete(partPath, recursive = false)
          }
          partPath
        } else
          finalFilename
      val os = fs.unsafeWriter(filename)
      val count = write(ctx, it, os)
      if (stageLocally)
        fs.copy(filename, finalFilename)
      ctx.region.clear()
      Iterator.single(f -> count)
    }
      .collect()
      .unzip

    val itemCount = partitionCounts.sum
    assert(nPartitions == partitionCounts.length)

    (partFiles, partitionCounts)
  }
}
