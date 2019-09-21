package is.hail.io

import java.io._

import is.hail.annotations._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.fs.FS
import is.hail.io.index.IndexWriter
import is.hail.rvd.{AbstractRVDSpec, IndexSpec, IndexedRVDSpec, OrderedRVDSpec, RVDContext, RVDPartitioner, RVDType}
import is.hail.sparkextras._
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{ExposedMetrics, TaskContext}

object RichContextRDDRegionValue {
  def writeRowsPartition(
    makeEnc: (OutputStream) => Encoder,
    indexKeyFieldIndices: Array[Int] = null,
    rowType: PStruct = null
  )(ctx: RVDContext, it: Iterator[RegionValue], os: OutputStream, iw: IndexWriter): Long = {
    val context = TaskContext.get
    val outputMetrics =
      if (context != null)
        context.taskMetrics().outputMetrics
      else
        null
    val trackedOS = new ByteTrackingOutputStream(os)
    val en = makeEnc(trackedOS)
    var rowCount = 0L

    it.foreach { rv =>
      if (iw != null) {
        val off = en.indexOffset()
        val key = SafeRow.selectFields(rowType, rv)(indexKeyFieldIndices)
        iw += (key, off, Row())
      }
      en.writeByte(1)
      en.writeRegionValue(rv.region, rv.offset)
      ctx.region.clear()
      rowCount += 1

      if (outputMetrics != null) {
        ExposedMetrics.setBytes(outputMetrics, trackedOS.bytesWritten)
        ExposedMetrics.setRecords(outputMetrics, rowCount)
      }
    }

    en.writeByte(0) // end
    en.flush()
    if (outputMetrics != null) {
      ExposedMetrics.setBytes(outputMetrics, trackedOS.bytesWritten)
    }
    os.close()

    rowCount
  }

  def writeSplitRegion(
    fs: FS,
    path: String,
    t: RVDType,
    it: Iterator[RegionValue],
    idx: Int,
    ctx: RVDContext,
    partDigits: Int,
    stageLocally: Boolean,
    makeIndexWriter: (FS, String) => IndexWriter,
    makeRowsEnc: (OutputStream) => Encoder,
    makeEntriesEnc: (OutputStream) => Encoder
  ): (String, Long) = {
    val fullRowType = t.rowType

    val context = TaskContext.get
    val f = partFile(partDigits, idx, context)
    val outputMetrics = context.taskMetrics().outputMetrics
    val finalRowsPartPath = path + "/rows/rows/parts/" + f
    val finalEntriesPartPath = path + "/entries/rows/parts/" + f
    val finalIdxPath = path + "/index/" + f + ".idx"
    val (rowsPartPath, entriesPartPath, idxPath) =
      if (stageLocally) {
        val rowsPartPath = fs.getTemporaryFile("file:///tmp")
        val entriesPartPath = fs.getTemporaryFile("file:///tmp")
        val idxPath = rowsPartPath + ".idx"
        context.addTaskCompletionListener { (context: TaskContext) =>
          fs.delete(rowsPartPath, recursive = false)
          fs.delete(entriesPartPath, recursive = false)
          fs.delete(idxPath, recursive = true)
        }
        (rowsPartPath, entriesPartPath, idxPath)
      } else
        (finalRowsPartPath, finalEntriesPartPath, finalIdxPath)

    val rowCount = fs.writeFile(rowsPartPath) { rowsOS =>
      val trackedRowsOS = new ByteTrackingOutputStream(rowsOS)
      using(makeRowsEnc(trackedRowsOS)) { rowsEN =>

        fs.writeFile(entriesPartPath) { entriesOS =>
          val trackedEntriesOS = new ByteTrackingOutputStream(entriesOS)
          using(makeEntriesEnc(trackedEntriesOS)) { entriesEN =>
            using(makeIndexWriter(fs, idxPath)) { iw =>

              var rowCount = 0L

              it.foreach { rv =>
                val rows_off = rowsEN.indexOffset()
                val ents_off = entriesEN.indexOffset()
                val key = SafeRow.selectFields(fullRowType, rv)(t.kFieldIdx)
                iw += (key, rows_off, Row(ents_off))

                rowsEN.writeByte(1)
                rowsEN.writeRegionValue(rv.region, rv.offset)

                entriesEN.writeByte(1)
                entriesEN.writeRegionValue(rv.region, rv.offset)

                ctx.region.clear()

                rowCount += 1

                ExposedMetrics.setBytes(outputMetrics, trackedRowsOS.bytesWritten + trackedEntriesOS.bytesWritten)
                ExposedMetrics.setRecords(outputMetrics, 2 * rowCount)
              }

              rowsEN.writeByte(0) // end
              entriesEN.writeByte(0)

              rowsEN.flush()
              entriesEN.flush()
              ExposedMetrics.setBytes(outputMetrics, trackedRowsOS.bytesWritten + trackedEntriesOS.bytesWritten)

              rowCount
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

    f -> rowCount
  }

  def writeSplitSpecs(
    fs: FS,
    path: String,
    codecSpec: CodecSpec,
    t: RVDType,
    rowsRVType: PStruct,
    entriesRVType: PStruct,
    partFiles: Array[String],
    partitioner: RVDPartitioner
  ) {
    val rowsSpec = IndexedRVDSpec(
      rowsRVType, t.key, codecSpec, IndexSpec.defaultAnnotation("../../index", t.kType.virtualType), partFiles, partitioner)
    rowsSpec.write(fs, path + "/rows/rows")

    val entriesSpec = IndexedRVDSpec(entriesRVType, FastIndexedSeq(), codecSpec,
      IndexSpec.defaultAnnotation("../../index", t.kType.virtualType, withOffsetField = true), partFiles,
      RVDPartitioner.unkeyed(partitioner.numPartitions))
    entriesSpec.write(fs, path + "/entries/rows")
  }
}

class RichContextRDDRegionValue(val crdd: ContextRDD[RVDContext, RegionValue]) extends AnyVal {
  def boundary: ContextRDD[RVDContext, RegionValue] =
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

  def writeRows(
    path: String,
    idxRelPath: String,
    t: RVDType,
    stageLocally: Boolean,
    encoding: CodecSpec2
  ): (Array[String], Array[Long]) = {
    crdd.writePartitions(
      path,
      idxRelPath,
      stageLocally,
      IndexWriter.builder(t.kType, +PStruct()),
      RichContextRDDRegionValue.writeRowsPartition(
        encoding.buildEncoder(t.rowType, t.rowType),
        t.kFieldIdx,
        t.rowType))
  }

  def toRows(rowType: PStruct): RDD[Row] = {
    crdd.run.map(rv => SafeRow(rowType, rv.region, rv.offset))
  }
}
