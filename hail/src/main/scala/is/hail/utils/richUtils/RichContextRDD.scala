package is.hail.utils.richUtils

import java.io._

import is.hail.HailContext
import is.hail.expr.ir.ExecuteContext
import is.hail.io.fs.FS
import is.hail.io.index.IndexWriter
import is.hail.rvd.RVDContext
import is.hail.utils._
import is.hail.sparkextras._
import org.apache.hadoop.conf.{Configuration => HadoopConf}
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object RichContextRDD {
  def writeParts[T](ctx: RVDContext, rootPath: String, f:String, idxRelPath: String, mkIdxWriter: (String) => IndexWriter,
                    stageLocally: Boolean, fs: FS, localTmpdir: String, it: Iterator[T],
                    write: (RVDContext, Iterator[T], OutputStream, IndexWriter) => Long): Iterator[(String, Long)] = {
    val finalFilename = rootPath + "/parts/" + f
    val finalIdxFilename = if (idxRelPath != null) rootPath + "/" + idxRelPath + "/" + f + ".idx" else null
    val (filename, idxFilename) =
      if (stageLocally) {
        val context = TaskContext.get
        val partPath = ExecuteContext.createTmpPathNoCleanup(localTmpdir, "write-partitions-part")
        val idxPath = partPath + ".idx"
        context.addTaskCompletionListener { (context: TaskContext) =>
          fs.delete(partPath, recursive = false)
          fs.delete(idxPath, recursive = true)
        }
        partPath -> idxPath
      } else
        finalFilename -> finalIdxFilename
    val os = fs.create(filename)
    val iw = mkIdxWriter(idxFilename)
    val count = write(ctx, it, os, iw)
    if (iw != null)
      iw.close()
    if (stageLocally) {
      fs.copy(filename, finalFilename)
      if (iw != null) {
        fs.copy(idxFilename + "/index", finalIdxFilename + "/index")
        fs.copy(idxFilename + "/metadata.json.gz", finalIdxFilename + "/metadata.json.gz")
      }
    }
    ctx.region.clear()
    Iterator.single(f -> count)
  }
}

class RichContextRDD[T: ClassTag](crdd: ContextRDD[T]) {

  def cleanupRegions: ContextRDD[T] = {
    crdd.cmapPartitionsAndContext { (ctx, part) =>
      val it = part.flatMap(_ (ctx))
      new Iterator[T]() {
        private[this] var cleared: Boolean = false

        def hasNext: Boolean = {
          if (!cleared) {
            cleared = true
            ctx.region.clear()
          }
          it.hasNext
        }

        def next: T = {
          if (!cleared) {
            ctx.region.clear()
          }
          cleared = false
          it.next
        }
      }
    }
  }



  // If idxPath is null, then mkIdxWriter should return null and not read its string argument
  def writePartitions(
    ctx: ExecuteContext,
    path: String,
    idxRelPath: String,
    stageLocally: Boolean,
    mkIdxWriter: (String) => IndexWriter,
    write: (RVDContext, Iterator[T], OutputStream, IndexWriter) => Long
  ): (Array[String], Array[Long]) = {
    val localTmpdir = ctx.localTmpdir
    val fs = ctx.fs
    val fsBc = ctx.fsBc

    fs.mkDir(path + "/parts")
    if (idxRelPath != null)
      fs.mkDir(path + "/" + idxRelPath)

    val nPartitions = crdd.getNumPartitions

    val d = digitsNeeded(nPartitions)

    val (partFiles, partitionCounts) = crdd.cmapPartitionsWithIndex { (i, ctx, it) =>
      val f = partFile(d, i, TaskContext.get)
      RichContextRDD.writeParts(ctx, path, f, idxRelPath, mkIdxWriter, stageLocally, fs, localTmpdir, it, write)
    }
      .collect()
      .unzip

    val itemCount = partitionCounts.sum
    assert(nPartitions == partitionCounts.length)

    (partFiles, partitionCounts)
  }
}
