package is.hail.utils.richUtils
import is.hail.annotations.RegionPool
import is.hail.backend.ExecuteContext
import is.hail.io.FileWriteMetadata
import is.hail.io.fs.FS
import is.hail.io.index.IndexWriter
import is.hail.rvd.RVDContext
import is.hail.sparkextras._
import is.hail.utils._

import scala.reflect.ClassTag

import java.io._

import org.apache.spark.TaskContext

object RichContextRDD {
  def writeParts[T](
    ctx: RVDContext,
    rootPath: String,
    f: String,
    idxRelPath: String,
    mkIdxWriter: (String, RegionPool) => IndexWriter,
    stageLocally: Boolean,
    fs: FS,
    localTmpdir: String,
    it: Iterator[T],
    write: (RVDContext, Iterator[T], OutputStream, IndexWriter) => (Long, Long),
  ): Iterator[FileWriteMetadata] = {
    val finalFilename = rootPath + "/parts/" + f
    val finalIdxFilename =
      if (idxRelPath != null) rootPath + "/" + idxRelPath + "/" + f + ".idx" else null
    val (filename, idxFilename) =
      if (stageLocally) {
        val context = TaskContext.get
        val partPath = ExecuteContext.createTmpPathNoCleanup(localTmpdir, "write-partitions-part")
        val idxPath = partPath + ".idx"
        context.addTaskCompletionListener[Unit] { (context: TaskContext) =>
          fs.delete(partPath, recursive = false)
          fs.delete(idxPath, recursive = true)
        }
        partPath -> idxPath
      } else
        finalFilename -> finalIdxFilename
    val os = fs.create(filename)
    val iw = mkIdxWriter(idxFilename, ctx.r.pool)

    // write must close `os` and `iw`
    val (rowCount, bytesWritten) = write(ctx, it, os, iw)

    if (stageLocally) {
      fs.copy(filename, finalFilename)
      if (iw != null) {
        fs.copy(idxFilename + "/index", finalIdxFilename + "/index")
        fs.copy(idxFilename + "/metadata.json.gz", finalIdxFilename + "/metadata.json.gz")
      }
    }
    ctx.region.clear()
    Iterator.single(FileWriteMetadata(f, rowCount, bytesWritten))
  }
}

class RichContextRDD[T: ClassTag](crdd: ContextRDD[T]) {

  def cleanupRegions: ContextRDD[T] = {
    crdd.cmapPartitionsAndContext { (ctx, part) =>
      val it = part.flatMap(_(ctx))
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
    mkIdxWriter: (String, RegionPool) => IndexWriter,
    write: (RVDContext, Iterator[T], OutputStream, IndexWriter) => (Long, Long),
  ): Array[FileWriteMetadata] = {
    val localTmpdir = ctx.localTmpdir
    val fs = ctx.fs

    fs.mkDir(path + "/parts")
    if (idxRelPath != null)
      fs.mkDir(path + "/" + idxRelPath)

    val nPartitions = crdd.getNumPartitions

    val d = digitsNeeded(nPartitions)

    val fileData = crdd.cmapPartitionsWithIndex { (i, ctx, it) =>
      val f = partFile(d, i, TaskContext.get)
      RichContextRDD.writeParts(ctx, path, f, idxRelPath, mkIdxWriter, stageLocally, fs,
        localTmpdir, it, write)
    }
      .collect()

    assert(nPartitions == fileData.length)

    fileData
  }
}
