package is.hail.io.tabix

import java.io.InputStream
import java.nio.charset.StandardCharsets

import htsjdk.samtools.util.FileExtensions
import htsjdk.tribble.util.ParsingUtils
import is.hail.io.compress.BGzipInputStream
import is.hail.io.fs.FS
import is.hail.utils._
import is.hail.backend.BroadcastValue

import scala.collection.mutable
import scala.language.implicitConversions

// Helper data classes

class Tabix(
  val format: Int,
  val colSeq: Int,
  val colBeg: Int,
  val meta: Int,
  val seqs: Array[String],
  val chr2tid: mutable.HashMap[String, Int],
  val indices: Array[(mutable.HashMap[Int, Array[TbiPair]], Array[Long])]
)

case class TbiPair(var _1: Long, var _2: Long) extends java.lang.Comparable[TbiPair] {
  @Override
  def compareTo(other: TbiPair): Int = TbiOrd.compare(this, other)
}

object TbiPair {
  implicit def tup2Pair(t: (Long, Long)): TbiPair = new TbiPair(t._1, t._2)
}

object TbiOrd extends Ordering[TbiPair] {
  def compare(u: TbiPair, v: TbiPair): Int = if (u._1 == v._1) {
    0
  } else if (less64(u._1, v._1)) {
    -1
  } else {
    1
  }

  def less64(u: Long, v: Long): Boolean = (u < v) ^ (u < 0) ^ (v < 0)
}

// Tabix reader
// Tabix file format is described here: https://samtools.github.io/hts-specs/tabix.pdf

object TabixReader {
  val MaxBin: Int = 37450
  val TadLidxShift: Int = 14
  val DefaultBufferSize: Int = 1000
  val Magic: Array[Byte] = Array(84, 66, 73, 1) // "TBI\1"

  def readInt(is: InputStream): Int =
    (is.read() & 0xff) |
    ((is.read() & 0xff) << 8) |
    ((is.read() & 0xff) << 16) |
    ((is.read() & 0xff) << 24)

  def readLong(is: InputStream): Long =
    (is.read() & 0xff).asInstanceOf[Long] |
    ((is.read() & 0xff).asInstanceOf[Long] << 8) |
    ((is.read() & 0xff).asInstanceOf[Long] << 16) |
    ((is.read() & 0xff).asInstanceOf[Long] << 24) |
    ((is.read() & 0xff).asInstanceOf[Long] << 32) |
    ((is.read() & 0xff).asInstanceOf[Long] << 40) |
    ((is.read() & 0xff).asInstanceOf[Long] << 48) |
    ((is.read() & 0xff).asInstanceOf[Long] << 56)

  def readLine(is: InputStream): String = readLine(is, DefaultBufferSize)

  def readLine(is: InputStream, initialBufferSize: Int): String = {
    val buf = new ArrayBuilder[Byte](initialBufferSize)
    var c = is.read()
    while (c >= 0 && c != '\n') {
      if (c == '\r') {
        c = is.read()
        if (c != '\n') buf += '\r'
      } else {
        buf += c.asInstanceOf[Byte]
        c = is.read()
      }
    }
    new String(buf.result(), StandardCharsets.UTF_8)
  }
}

class TabixReader(val filePath: String, fs: FS, idxFilePath: Option[String] = None) {
  import TabixReader._

  val indexPath: String = idxFilePath match {
    case None => ParsingUtils.appendToPath(filePath, FileExtensions.TABIX_INDEX)
    case Some(s) =>
      if (s.endsWith(FileExtensions.TABIX_INDEX))
        s
      else
        fatal(s"unknown file extension for tabix index: $s")
  }

  val index: Tabix = using(fs.open(indexPath)) { is =>
    var buf = new Array[Byte](4)
    is.read(buf, 0, 4) // read magic bytes "TBI\1"
    if (!(Magic sameElements buf))
      fatal(s"""magic number failed validation
        |magic: ${ Magic.mkString("[", ",", "]") }
        |data : ${ buf.mkString("[", ",", "]") }""".stripMargin)
    val seqs = new Array[String](readInt(is))
    val format = readInt(is)
    // Require VCF for now
    if (format != 2)
      fatal(s"Hail only supports tabix indexing for VCF, found format code ${ format }")
    val colSeq = readInt(is)
    val colBeg = readInt(is)
    val colEnd = readInt(is)
    val meta = readInt(is)
    // meta char for VCF is '#'
    if (meta != '#')
      fatal(s"Meta character was ${ meta }, should be '#' for VCF")
    val chr2tid = new mutable.HashMap[String, Int]()
    readInt(is) // unused, need to consume

    // read the sequence dictionary
    buf = new Array[Byte](readInt(is)) // # sequences
    var (i, j, k) = (0, 0, 0)
    is.read(buf)
    while (i < buf.length) {
      if (buf(i) == 0) {
        val contig = new String(buf.slice(j, i))
        chr2tid += contig -> k
        seqs(k) = contig
        k += 1
        j = i + 1
      }
      i += 1
    }

    // read the index
    val indices = new ArrayBuilder[(mutable.HashMap[Int, Array[TbiPair]], Array[Long])](seqs.length)
    i = 0
    while (i < seqs.length) {
      // binning index
      val nBin = readInt(is)
      val binIdx = new mutable.HashMap[Int, Array[TbiPair]]()
      j = 0
      while (j < nBin) {
        val bin = readInt(is)
        val chunks = new Array[TbiPair](readInt(is))
        k = 0
        while (k < chunks.length) {
          chunks(k) = readLong(is) -> readLong(is)
          k += 1
        }
        binIdx += bin -> chunks
        j += 1
      }
      // linear index
      val linIdx = new Array[Long](readInt(is))
      k = 0
      while (k < linIdx.length) {
        linIdx(k) = readLong(is)
        k += 1
      }
      indices += binIdx -> linIdx
      i += 1
    }
    is.close()
    new Tabix(format, colSeq, colBeg, meta, seqs, chr2tid, indices.result())
  }

  def chr2tid(chr: String): Int = index.chr2tid.get(chr) match {
      case Some(i) => i
      case _ => -1
    }

  // This method returns an array of tuples suitable to be passed to the constructor of
  // TabixLineIterator. The arguments beg and end are endpoints to an interval of loci within tid.
  // The iterator returned will return all line with loci between beg and end inclusive, and may
  // return slightly more data on either end due to the indexing being inexact.
  def queryPairs(tid: Int, beg: Int, end: Int): Array[TbiPair] = {
    if (tid < 0 || tid > index.indices.length) {
      new Array[TbiPair](0)
    } else {
      val idx = index.indices(tid)
      val bins = reg2bins(beg, end)
      val minOff = if (idx._2.length > 0 && (beg >> TadLidxShift) >= idx._2.length)
          idx._2(idx._2.length - 1)
        else if (idx._2.length > 0)
          idx._2(beg >> TadLidxShift)
        else
          0L

      var i = 0
      var nOff = 0
      while (i < bins.length) {
        nOff += idx._1.get(bins(i)).map(_.length).getOrElse(0)
        i += 1
      }
      if (nOff == 0)
        new Array[TbiPair](0)
      else {
        val off = new Array[TbiPair](nOff)
        nOff = 0
        i = 0
        while (i < bins.length) {
          val c = idx._1.getOrElse(bins(i), null)
          val len = if (c == null) { 0 } else { c.length }
          var j = 0
          while (j < len) {
            if (TbiOrd.less64(minOff, c(j)._2)) {
              off(nOff) = TbiPair(c(j)._1, c(j)._2)
              nOff += 1
            }
            j += 1
          }
          i += 1
        }
        java.util.Arrays.sort(off, 0, nOff, null)
        // resolve contained adjacent blocks
        var l = 0
        i = 1
        while (i < nOff) {
          if (TbiOrd.less64(off(l)._2, off(i)._2)) {
            l += 1
            off(l)._1 = off(i)._1
            off(l)._2 = off(i)._2
          }
          i += 1
        }
        nOff = l + 1
        // resolve overlaps
        i = 1
        while (i < nOff) {
          if (!TbiOrd.less64(off(i - 1)._2, off(i)._1))
            off(i - 1)._2 = off(i)._1
          i += 1
        }
        // merge blocks
        i = 1
        l = 0
        while (i < nOff) {
          if ((off(l)._2 >> 16) == (off(i)._1 >> 16))
            off(l)._2 = off(i)._2
          else {
            l += 1
            off(l)._1 = off(i)._1
            off(l)._2 = off(i)._2
          }
          i += 1
        }
        nOff = l + 1
        val ret = new Array[TbiPair](nOff)
        i = 0
        while (i < nOff) {
          if (off(i) != null)
            ret(i) = TbiPair(off(i)._1, off(i)._2)
          else
            ret(i) = null
          i += 1
        }
        if (ret.length == 0 || (ret.length == 1 && ret(0) == null))
          new Array[TbiPair](0)
        else
          ret
      }
    }
  }

  private def reg2bins(beg: Int, _end: Int): Array[Int] = {
    if (beg >= _end)
      new Array[Int](0)
    else {
      var end = _end
      val bins = new ArrayBuilder[Int](MaxBin)
      if (end >= (1 << 29)) {
        end = 1 << 29
      }
      end -= 1
      bins += 0
      var k = 1 + (beg >> 26)
      while (k <= 1 + (end >> 26)) {
        bins += k
        k += 1
      }
      k = 9 + (beg >> 23)
      while (k <= 9 + (end >> 23)) {
        bins += k
        k += 1
      }
      k = 73 + (beg >> 20)
      while (k <= 73 + (end >> 20)) {
        bins += k
        k += 1
      }
      k = 585 + (beg >> 17)
      while (k <= 585 + (end >> 17)) {
        bins += k
        k += 1
      }
      k = 4681 + (beg >> 14)
      while (k <= 4681 + (end >> 14)) {
        bins += k
        k += 1
      }
      bins.result()
    }
  }
}

final class TabixLineIterator(
  private val fsBc: BroadcastValue[FS],
  private val filePath: String,
  private val offsets: Array[TbiPair]
)
  extends java.lang.AutoCloseable
{
  private var i: Int = -1
  private var isEof = false
  private var is = new BGzipInputStream(fsBc.value.openNoCompression(filePath))

  // The line iterator buffer and associated state, we use this to avoid making
  // endless calls to read() (the no arg version returning int) on the input
  // stream, we start with 64k to make to make sure that we can always read
  // a block, and then grow it as neccessary if a line is too large.
  //
  // The key invariant here is that bufferCursor should, except possibly in
  // readLine itself, be less than bufferLen. Should bufferCursor be greater
  // than or equal to bufferLen, we call refreshBuffer, which also saves the
  // current input stream virtual offset. If we uphold this invariant, then the
  // bufferCursor to bufferLen range within the buffer will always contain data
  // from the current block we are reading, meaning getVirtualOffset will
  // always return the same value as if we were calling is.read() to read the
  // lines and is.getVirtualOffset to update the current file pointer for every
  // call to next
  private var buffer = new Array[Byte](1 << 16)
  private var bufferCursor: Int = 0
  private var bufferLen: Int = 0
  private var bufferEOF: Boolean = false
  private var bufferPositionAtLastRead = 0L

  private var virtualFileOffsetAtLastRead = 0L

  private def getVirtualOffset: Long = {
    virtualFileOffsetAtLastRead + (bufferCursor - bufferPositionAtLastRead)
  }

  private def virtualSeek(l: Long): Unit = {
    is.virtualSeek(l)
    refreshBuffer(0)
    bufferCursor = 0
  }

  private def refreshBuffer(start: Int): Unit = {
    assert(start < buffer.length)
    bufferLen = start
    virtualFileOffsetAtLastRead = is.getVirtualOffset
    bufferPositionAtLastRead = start
    val bytesRead = is.read(buffer, start, buffer.length - start)
    if (bytesRead < 0)
      bufferEOF = true
    else
      bufferLen = start + bytesRead
    bufferCursor = start
  }

  def decodeString(start: Int, end: Int): String = {
    var stop = end
    if (stop > start && buffer(stop) == '\r')
      stop -= 1
    val len = end - start
    new String(buffer, start, len, StandardCharsets.UTF_8)
  }

  def readLine(): String = {
    assert(bufferCursor <= bufferLen)
    if (isEof)
      return null

    var start = bufferCursor

    while (true) {
      var str: String = null;
      while (bufferCursor < bufferLen && buffer(bufferCursor) != '\n') {
        bufferCursor += 1
      }

      if (bufferCursor == bufferLen) { // no newline before end of buffer
        if (bufferEOF) {
          // `is` indicates end of file
          isEof = true
          str = decodeString(start, bufferCursor)
        } else if (bufferLen == buffer.length) {
          // line overflows buffer, need to increase buffer size
          val tmp = new Array[Byte](buffer.length * 2)
          System.arraycopy(buffer, 0, tmp, 0, buffer.length)
          buffer = tmp
          refreshBuffer(bufferLen)
        } else if (start > 0) {
          // line does not overflow buffer, but spans buffer.
          // Copy line left to the beginning of buffer and continue
          val nToCopy = bufferLen - start
          System.arraycopy(buffer, start, buffer, 0, nToCopy)
          start = 0
          refreshBuffer(nToCopy)
        } else {
          // line does not span buffer, but does not fill buffer either, read more
          refreshBuffer(bufferCursor)
        }
      } else { // found a newline
        str = decodeString(start, bufferCursor)
      }

      if (str != null) {
        bufferCursor += 1
        // we refresh here to make sure that the cursor is never pointing to
        // one past the end of a block, making it so getVirtualOffset will
        // never return the pointer pointing past the end of a block (the one
        // that overlaps with the start of the next block).
        if (bufferCursor >= bufferLen) {
          refreshBuffer(0)
        }
        return str
      }
    }
    throw new AssertionError()
  }

  def next(): String = {
    var s: String = null
    while (s == null && !isEof) {
      val curOff = getVirtualOffset
      if (i < 0 || curOff == 0 || !TbiOrd.less64(curOff, offsets(i)._2)) { // jump to next chunk
        if (i == offsets.length - 1) {
          isEof = true
          return s
        }
        if (i < 0 || offsets(i)._2 != offsets(i + 1)._1) {
          virtualSeek(offsets(i + 1)._1)
        }
        i += 1
      }
      s = readLine()
      if (s != null) {
        if (s.isEmpty || s.charAt(0) == '#')
          s = null // continue
      } else
        isEof = true
    }
    s
  }

  override def close() {
    if (is != null) {
      is.close()
      is = null
    }
  }
}
