package is.hail.io

import java.io._
import is.hail.asm4s.{HailClassLoader, theHailClassLoaderForSparkWorkers}
import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.SparkTaskContext
import is.hail.types.physical._
import is.hail.io.fs.FS
import is.hail.io.index.IndexWriter
import is.hail.rvd.{AbstractIndexSpec, IndexSpec, MakeRVDSpec, RVDContext, RVDPartitioner, RVDType}
import is.hail.sparkextras._
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{ExposedMetrics, TaskContext}

import scala.reflect.ClassTag

object RichContextRDDRegionValue {
  def writeRowsPartition(
    makeEnc: (OutputStream, ExecuteContext) => Encoder,
    indexKeyFieldIndices: Array[Int] = null,
    rowType: PStruct = null
  )(ctx: ExecuteContext, it: Iterator[Long], os: OutputStream, iw: IndexWriter): (Long, Long) = {
    val context = TaskContext.get
    val outputMetrics =
      if (context != null)
        context.taskMetrics().outputMetrics
      else
        null
    val trackedOS = new ByteTrackingOutputStream(os)
    val en = makeEnc(trackedOS, ctx)
    var rowCount = 0L

    it.foreach { ptr =>
      if (iw != null) {
        val off = en.indexOffset()
        val key = SafeRow.selectFields(rowType, ctx.r, ptr)(indexKeyFieldIndices)
        iw.appendRow(key, off, Row())
      }
      en.writeByte(1)
      en.writeRegionValue(ptr)
      ctx.r.clear()
      rowCount += 1

      if (outputMetrics != null) {
        ExposedMetrics.setBytes(outputMetrics, trackedOS.bytesWritten)
        ExposedMetrics.setRecords(outputMetrics, rowCount)
      }
    }

    en.writeByte(0) // end
    en.flush()
    var bytesWritten = 0L
    if (iw != null) {
      // close() flushes to the output stream, so look up bytesWritten after closing
      iw.close()
      bytesWritten += iw.trackedOS().bytesWritten
    }
    if (outputMetrics != null) {
      ExposedMetrics.setBytes(outputMetrics, trackedOS.bytesWritten)
    }
    bytesWritten += trackedOS.bytesWritten
    os.close()

    (rowCount, bytesWritten)
  }

  def writeSplitRegion(
    localTmpdir: String,
    fs: FS,
    path: String,
    t: RVDType,
    it: Iterator[Long],
    idx: Int,
    ctx: RVDContext,
    partDigits: Int,
    stageLocally: Boolean,
    makeIndexWriter: (String, RegionPool) => IndexWriter,
    makeRowsEnc: (OutputStream) => Encoder,
    makeEntriesEnc: (OutputStream) => Encoder
  ): FileWriteMetadata = {
    val fullRowType = t.rowType

    val context = TaskContext.get
    val f = partFile(partDigits, idx, context)
    val outputMetrics = context.taskMetrics().outputMetrics
    val finalRowsPartPath = path + "/rows/rows/parts/" + f
    val finalEntriesPartPath = path + "/entries/rows/parts/" + f
    val finalIdxPath = path + "/index/" + f + ".idx"
    val (rowsPartPath, entriesPartPath, idxPath) =
      if (stageLocally) {
        val rowsPartPath = ExecuteContext.createTmpPathNoCleanup(localTmpdir, "write-split-staged-rows-part")
        val entriesPartPath = ExecuteContext.createTmpPathNoCleanup(localTmpdir, "write-split-staged-entries-part")
        val idxPath = rowsPartPath + ".idx"
        context.addTaskCompletionListener[Unit] { (context: TaskContext) =>
          fs.delete(rowsPartPath, recursive = false)
          fs.delete(entriesPartPath, recursive = false)
          fs.delete(idxPath, recursive = true)
        }
        (rowsPartPath, entriesPartPath, idxPath)
      } else
        (finalRowsPartPath, finalEntriesPartPath, finalIdxPath)

    val (rowCount, totalBytesWritten) = using(fs.create(rowsPartPath)) { rowsOS =>
      val trackedRowsOS = new ByteTrackingOutputStream(rowsOS)
      using(makeRowsEnc(trackedRowsOS)) { rowsEN =>

        using(fs.create(entriesPartPath)) { entriesOS =>
          val trackedEntriesOS = new ByteTrackingOutputStream(entriesOS)
          using(makeEntriesEnc(trackedEntriesOS)) { entriesEN =>
            using(makeIndexWriter(idxPath, ctx.r.pool)) { iw =>
              var rowCount = 0L

              it.foreach { ptr =>
                val rows_off = rowsEN.indexOffset()
                val ents_off = entriesEN.indexOffset()
                val key = SafeRow.selectFields(fullRowType, ctx.r, ptr)(t.kFieldIdx)
                iw.appendRow(key, rows_off, Row(ents_off))

                rowsEN.writeByte(1)
                rowsEN.writeRegionValue(ptr)

                entriesEN.writeByte(1)
                entriesEN.writeRegionValue(ptr)

                ctx.region.clear()

                rowCount += 1

                ExposedMetrics.setBytes(outputMetrics, trackedRowsOS.bytesWritten + trackedEntriesOS.bytesWritten)
                ExposedMetrics.setRecords(outputMetrics, 2 * rowCount)
              }

              rowsEN.writeByte(0) // end
              entriesEN.writeByte(0)

              rowsEN.flush()
              entriesEN.flush()

              val totalBytesWritten = trackedRowsOS.bytesWritten + trackedEntriesOS.bytesWritten + iw.trackedOS().bytesWritten
              ExposedMetrics.setBytes(outputMetrics, totalBytesWritten)

              (rowCount, totalBytesWritten)
            }
          }
        }
      }
    }

    if (stageLocally) {
      fs.copy(rowsPartPath, finalRowsPartPath)
      fs.copy(entriesPartPath, finalEntriesPartPath)
      fs.copy(idxPath + "/index", finalIdxPath + "/index")
      fs.copy(idxPath + "/metadata.json.gz", finalIdxPath + "/metadata.json.gz")
    }

    FileWriteMetadata(f, rowCount, totalBytesWritten)
  }

  def writeSplitSpecs(
    fs: FS,
    path: String,
    rowsCodecSpec: AbstractTypedCodecSpec,
    entriesCodecSpec: AbstractTypedCodecSpec,
    rowsIndexSpec: AbstractIndexSpec,
    entriesIndexSpec: AbstractIndexSpec,
    t: RVDType,
    rowsRVType: PStruct,
    entriesRVType: PStruct,
    partFiles: Array[String],
    partitioner: RVDPartitioner
  ) {
    val rowsSpec = MakeRVDSpec(rowsCodecSpec, partFiles, partitioner, rowsIndexSpec)
    rowsSpec.write(fs, path + "/rows/rows")

    val entriesSpec = MakeRVDSpec(entriesCodecSpec, partFiles,
      RVDPartitioner.unkeyed(partitioner.sm, partitioner.numPartitions), entriesIndexSpec)
    entriesSpec.write(fs, path + "/entries/rows")
  }
}

class RichContextRDDLong(val crdd: ContextRDD[Long]) extends AnyVal {
  def boundary: ContextRDD[Long] =
    crdd.cmapPartitionsAndContext { (consumerCtx, part) =>
      val producerCtx = consumerCtx.freshContext
      val it = part.flatMap(_ (producerCtx))
      new Iterator[Long]() {
        private[this] var cleared: Boolean = false

        def hasNext: Boolean = {
          if (!cleared) {
            cleared = true
            producerCtx.region.clear()
          }
          it.hasNext
        }

        def next: Long = {
          if (!cleared) {
            producerCtx.region.clear()
          }
          cleared = false
          it.next
        }
      }
    }

  def toCRDDRegionValue: ContextRDD[RegionValue] =
    boundary.cmapPartitionsWithContext((ctx, part) => {
      val rv = RegionValue(ctx.r)
      part(ctx).map(ptr => { rv.setOffset(ptr); rv })
    })

  def writeRows(
    ctx: ExecuteContext,
    path: String,
    idxRelPath: String,
    t: RVDType,
    stageLocally: Boolean,
    encoding: AbstractTypedCodecSpec
  ): Array[FileWriteMetadata] = {
    crdd.writePartitions(
      ctx,
      path,
      idxRelPath,
      stageLocally,
      {
        val f1= IndexWriter.builder(ctx, t.kType, +PCanonicalStruct())
        f1(_, theHailClassLoaderForSparkWorkers, SparkTaskContext.get(), _)
      },
      RichContextRDDRegionValue.writeRowsPartition(
        encoding.buildEncoder(ctx, t.rowType),
        t.kFieldIdx,
        t.rowType) _)
  }

  def toRows(rowType: PStruct): RDD[Row] = {
    crdd.cmap((ctx, ptr) => SafeRow(rowType, ptr)).run
  }
}

class RichContextRDDRegionValue(val crdd: ContextRDD[RegionValue]) extends AnyVal {
  def boundary: ContextRDD[RegionValue] =
    crdd.cmapPartitionsAndContext { (consumerCtx, part) =>
      val producerCtx = consumerCtx.freshContext
      val it = part.flatMap(_ (producerCtx))
      new Iterator[RegionValue]() {
        private[this] var cleared: Boolean = false

        def hasNext: Boolean = {
          if (!cleared) {
            cleared = true
            producerCtx.region.clear()
          }
          it.hasNext
        }

        def next: RegionValue = {
          if (!cleared) {
            producerCtx.region.clear()
          }
          cleared = false
          it.next
        }
      }
    }

  def toCRDDPtr: ContextRDD[Long] =
    crdd.cmap { (consumerCtx, rv) =>
      // Need to track regions that are in use, but don't want to create a cycle.
      if (consumerCtx.region != rv.region) {
        consumerCtx.region.addReferenceTo(rv.region)
      }
      rv.offset
    }

  def cleanupRegions: ContextRDD[RegionValue] = {
    crdd.cmapPartitionsAndContext { (ctx, part) =>
      val it = part.flatMap(_ (ctx))
      new Iterator[RegionValue]() {
        private[this] var cleared: Boolean = false

        def hasNext: Boolean = {
          if (!cleared) {
            cleared = true
            ctx.region.clear()
          }
          it.hasNext
        }

        def next: RegionValue = {
          if (!cleared) {
            ctx.region.clear()
          }
          cleared = false
          it.next
        }
      }
    }
  }


  def toRows(rowType: PStruct): RDD[Row] = {
    crdd.run.map(rv => SafeRow(rowType, rv.offset))
  }
}
