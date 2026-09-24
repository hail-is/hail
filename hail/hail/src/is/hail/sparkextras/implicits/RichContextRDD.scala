package is.hail.sparkextras.implicits

import is.hail.annotations.RegionPool
import is.hail.asm4s.HailClassLoader
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.partFile
import is.hail.io.FileWriteMetadata
import is.hail.io.fs.FS
import is.hail.io.index.IndexWriter
import is.hail.rvd.RVDContext
import is.hail.sparkextras._
import is.hail.utils.digitsNeeded

import scala.reflect.ClassTag

import java.io._

import org.apache.spark.TaskContext

object RichContextRDD {
  def writeParts[T](
    hcl: HailClassLoader,
    ctx: RVDContext,
    rootPath: String,
    f: String,
    idxRelPath: String,
    mkIdxWriter: (String, HailClassLoader, RegionPool) => IndexWriter,
    fs: FS,
    it: Iterator[T],
    write: (HailClassLoader, RVDContext, Iterator[T], OutputStream, IndexWriter) => (Long, Long),
  ): Iterator[FileWriteMetadata] = {
    val filename = rootPath + "/parts/" + f
    val idxFilename =
      if (idxRelPath != null) rootPath + "/" + idxRelPath + "/" + f + ".idx" else null
    val os = fs.create(filename)
    val iw = mkIdxWriter(idxFilename, hcl, ctx.r.pool)

    // write must close `os` and `iw`
    val (rowCount, bytesWritten) = write(hcl, ctx, it, os, iw)

    ctx.region.clear()
    Iterator.single(FileWriteMetadata(f, rowCount, bytesWritten))
  }
}

class RichContextRDD[T](val crdd: ContextRDD[T]) extends AnyVal {

  def cleanupRegions(implicit CT: ClassTag[T]): ContextRDD[T] = {
    crdd.cmapPartitionsAndContext { (hcl, ctx, part) =>
      val it = part.flatMap(_(hcl, ctx))
      new Iterator[T]() {
        private[this] var cleared: Boolean = false

        override def hasNext: Boolean = {
          if (!cleared) {
            cleared = true
            ctx.region.clear()
          }
          it.hasNext
        }

        override def next(): T = {
          if (!cleared) {
            ctx.region.clear()
          }
          cleared = false
          it.next()
        }
      }
    }
  }

  // If idxPath is null, then mkIdxWriter should return null and not read its string argument
  def writePartitions(
    ctx: ExecuteContext,
    path: String,
    idxRelPath: String,
    mkIdxWriter: (String, HailClassLoader, RegionPool) => IndexWriter,
    write: (HailClassLoader, RVDContext, Iterator[T], OutputStream, IndexWriter) => (Long, Long),
  ): IndexedSeq[FileWriteMetadata] = {
    val fs = ctx.fs

    fs.mkDir(path + "/parts")
    if (idxRelPath != null)
      fs.mkDir(path + "/" + idxRelPath)

    val nPartitions = crdd.getNumPartitions

    val d = digitsNeeded(nPartitions)

    val fileData = crdd.cmapPartitionsWithIndex { (i, hcl, ctx, it) =>
      val f = partFile(d, i, TaskContext.get())
      RichContextRDD.writeParts(hcl, ctx, path, f, idxRelPath, mkIdxWriter, fs, it, write)
    }
      .collect()

    assert(nPartitions == fileData.length)

    fileData
  }
}
