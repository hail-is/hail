package is.hail.io.index

import java.io.InputStream
import is.hail.asm4s.HailClassLoader
import is.hail.annotations.Region
import is.hail.io.Decoder
import is.hail.types.physical.PType
import is.hail.types.virtual.TStruct
import is.hail.utils.{ByteTrackingInputStream, Interval}
import org.apache.spark.ExposedMetrics
import org.apache.spark.executor.InputMetrics
import org.apache.spark.sql.Row

trait CountedIterator[T] extends Iterator[T] {
  def getCurIdx: Long
}

class IndexReadIterator(
  theHailClassLoader: HailClassLoader,
  makeDec: (InputStream, HailClassLoader) => Decoder,
  region: Region,
  in: InputStream,
  idxr: IndexReader,
  offsetField: String, // can be null
  bounds: Interval,
  metrics: InputMetrics = null
) extends CountedIterator[Long] {

  private[this] val (startIdx, endIdx) = idxr.boundsByInterval(bounds)
  private[this] var curIdx = startIdx

  private[this] val trackedIn = new ByteTrackingInputStream(in)
  private[this] val field = Option(offsetField).map { f =>
    idxr.annotationType.asInstanceOf[TStruct].fieldIdx(f)
  }
  private[this] val dec =
    try {
      if (endIdx > startIdx) {
        val dec = makeDec(trackedIn, theHailClassLoader)
        val i = idxr.queryByIndex(startIdx)
        val off = field.map { j =>
          i.annotation.asInstanceOf[Row].getAs[Long](j)
        }.getOrElse(i.recordOffset)
        dec.seek(off)
        dec
      } else {
        in.close()
        null
      }
    } catch {
      case e: Exception =>
        idxr.close()
        in.close()
        throw e
    }

  private[this] var closed = false

  private var cont: Byte = if (dec != null) dec.readByte() else 0
  if (cont == 0) {
    idxr.close()
    if (dec != null) dec.close()
  }

  def ptype: PType = dec.ptype

  def hasNext: Boolean = cont != 0 && curIdx < endIdx

  def getCurIdx: Long = curIdx - 1

  def next(): Long = _next()

  def _next(): Long = {
    if (!hasNext)
      throw new NoSuchElementException("next on empty iterator")

    curIdx += 1
    try {
      val res = dec.readRegionValue(region)
      cont = dec.readByte()
      if (metrics != null) {
        ExposedMetrics.incrementRecord(metrics)
        ExposedMetrics.incrementBytes(metrics, trackedIn.bytesReadAndClear())
      }

      if (cont == 0) {
        close()
      }

      res
    } catch {
      case e: Exception =>
        close()
        throw e
    }
  }

  def close(): Unit = {
    if (!closed) {
      idxr.close()
      if (dec != null) dec.close()
      closed = true
    }
  }

  override def finalize(): Unit = {
    close()
  }
}
