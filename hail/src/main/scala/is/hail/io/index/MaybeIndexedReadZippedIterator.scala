package is.hail.io.index

import java.io.InputStream

import is.hail.annotations.Region
import is.hail.asm4s.AsmFunction3RegionLongLongLong
import is.hail.io.Decoder
import is.hail.types.virtual.TStruct
import is.hail.utils.{ByteTrackingInputStream, Interval}
import org.apache.spark.executor.InputMetrics
import org.apache.spark.sql.Row

class MaybeIndexedReadZippedIterator(
  mkRowsDec: (InputStream) => Decoder,
  mkEntriesDec: (InputStream) => Decoder,
  inserter: AsmFunction3RegionLongLongLong,
  region: Region,
  isRows: InputStream,
  isEntries: InputStream,
  idxr: IndexReader,
  rowsOffsetField: String,
  entriesOffsetField: String,
  bounds: Interval,
  metrics: InputMetrics = null
) extends Iterator[Long] {

  private[this] var closed: Boolean = false

  private[this] val startAndEnd = Option(idxr).map(_.boundsByInterval(bounds))
  private[this] val firstAnnotation = try {
    startAndEnd.flatMap { case (start, end) =>
      if (end == start || idxr.nKeys == 0) None else Some(idxr.queryByIndex(start))
    }
  } catch {
    case e: Exception =>
      if (idxr != null)
        idxr.close()
      isRows.close()
      isEntries.close()
      throw e
  }

  private[this] var n = startAndEnd.map(x => x._2 - x._1)

  private[this] val trackedRowsIn = new ByteTrackingInputStream(isRows)
  private[this] val trackedEntriesIn = new ByteTrackingInputStream(isEntries)

  private[this] val rowsIdxField = Option(rowsOffsetField).map { f => idxr.annotationType.asInstanceOf[TStruct].fieldIdx(f) }
  private[this] val entriesIdxField = Option(entriesOffsetField).map { f => idxr.annotationType.asInstanceOf[TStruct].fieldIdx(f) }

  private[this] val rows = try {
    if (n.forall(_ > 0)) {
      val dec = mkRowsDec(trackedRowsIn)
      firstAnnotation.foreach { i =>
        val off = rowsIdxField.map { j => i.annotation.asInstanceOf[Row].getAs[Long](j) }.getOrElse(i.recordOffset)
        dec.seek(off)
      }
      dec
    } else {
      isRows.close()
      isEntries.close()
      null
    }
  } catch {
    case e: Exception =>
      if (idxr != null)
        idxr.close()
      isRows.close()
      isEntries.close()
      throw e
  }
  private[this] val entries = try {
    if (rows == null) {
      null
    } else {
      val dec = mkEntriesDec(trackedEntriesIn)
      firstAnnotation.foreach { i =>
        val off = entriesIdxField.map { j => i.annotation.asInstanceOf[Row].getAs[Long](j) }.getOrElse(i.recordOffset)
        dec.seek(off)
      }
      dec
    }
  } catch {
    case e: Exception =>
      if (idxr != null)
        idxr.close()
      isRows.close()
      isEntries.close()
      throw e
  }

  require(!((rows == null) ^ (entries == null)))

  private def nextCont(): Byte = {
    val br = rows.readByte()
    val be = entries.readByte()
    assert(br == be)
    br
  }

  private var cont: Byte = if (rows != null) nextCont() else 0

  def hasNext: Boolean = cont != 0 && n.forall(_ > 0)

  def next(): Long = _next()

  def _next(): Long = {
    if (!hasNext)
      throw new NoSuchElementException("next on empty iterator")

    n = n.map(_ - 1)
    try {
      val rowOff = rows.readRegionValue(region)
      val entOff = entries.readRegionValue(region)
      val off = inserter(region, rowOff, entOff)
      cont = nextCont()

      if (cont == 0) {
        close()
      }

      off
    } catch {
      case e: Exception =>
        close()
        throw e
    }
  }

  def close(): Unit = {
    if (!closed) {
      if (idxr != null)
        idxr.close()
      if (rows != null) rows.close()
      if (entries != null) entries.close()
      closed = true
    }
  }

  override def finalize(): Unit = {
    close()
  }
}
