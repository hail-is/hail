package is.hail.io.tabix

import is.hail.HailContext
import is.hail.io.compress.BGzipInputStream
import is.hail.utils._

import htsjdk.tribble.util.{ParsingUtils, TabixUtils}
import org.apache.{hadoop => hd}
import org.apache.hadoop.io.compress.SplitCompressionInputStream

import java.io.InputStream
import java.nio.{ByteBuffer, ByteOrder}
import java.util.Arrays

import scala.collection.mutable.HashMap
import scala.language.implicitConversions

// Helper data classes

case class Tabix(
  val format: Int,
  val colSeq: Int,
  val colBeg: Int,
  val meta: Int,
  val seqs: Array[String],
  val chr2tid: HashMap[String, Int],
  val indicies: Array[(HashMap[Int, Array[TbiPair]], Array[Long])]
)

case class TbiPair(var _1: Long, var _2: Long) extends java.lang.Comparable[TbiPair] {
  @Override
  def compareTo(other: TbiPair) = TbiOrd.compare(this, other)
}

object TbiPair {
  implicit def tup2Pair(t: (Long, Long)): TbiPair = new TbiPair(t._1, t._2)
}

object TbiOrd extends Ordering[TbiPair] {
  def compare(u: TbiPair, v: TbiPair) = if (u._1 == v._1) {
    0
  } else if ((u._1 < v._1) ^ (u._1 < 0) ^ (v._1 < 0)) {
    -1
  } else {
    1
  }

  def less64(u: Long, v: Long) = (u < v) ^ (u < 0) ^ (v < 0)
}

// Tabix reader
// Tabix file format is described here: https://samtools.github.io/hts-specs/tabix.pdf

object TabixReader {
  val MaxBin: Int = 37450
  val TadLidxShift: Int = 14
  val DefaultBufferSize: Int = 1000
  val Magic: Array[Byte] = Array(84, 66, 73, 1) // "TBI\1"

  def readInt(is: InputStream): Int = {
    val buf = new Array[Byte](4)
    is.read(buf)
    ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt
  }

  def readLong(is: InputStream): Long = {
    val buf = new Array[Byte](8)
    is.read(buf)
    ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getLong
  }

  def readLine(is: InputStream): String = readLine(is, DefaultBufferSize)

  def readLine(is: InputStream, initialBufferSize: Int): String = {
    val buf = new StringBuffer(initialBufferSize)
    var c = is.read()
    while (c >= 0 && c != '\n') {
      buf.append(c)
      c = is.read()
    }
    buf.toString()
  }

  def apply(filePath: String): TabixReader = new TabixReader(filePath, None)
}

class TabixReader(val filePath: String, private val idxFilePath: Option[String]) {
  import TabixReader._

  val indexPath: String = idxFilePath match {
    case None => ParsingUtils.appendToPath(filePath, TabixUtils.STANDARD_INDEX_EXTENSION)
    case Some(s) => {
      if (s.endsWith(".tbi"))
        s
      else
        fatal(s"unknown file extension for tabix index: ${s}")
    }
  }

  private val hc = HailContext.get
  private val sc = hc.sc
  private val hConf = sc.hadoopConfiguration

  val index: Tabix = hConf.readFile(indexPath) { is =>
    var buf = new Array[Byte](4)
    is.read(buf, 0, 4) // read magic bytes "TBI\1"
    assert(Magic sameElements buf, s"""magic number failed validation
      |magic: ${ Magic.mkString("[", ",", "]") }
      |data : ${ buf.mkString("[", ",", "]") }""".stripMargin)
    val seqs = new Array[String](readInt(is))
    val format = readInt(is)
    assert(format == 2) // require VCF for now
    val colSeq = readInt(is)
    val colBeg = readInt(is)
    val colEnd = readInt(is)
    val meta = readInt(is)
    assert(meta == '#') // meta char for VCF is '#'
    val chr2tid = new HashMap[String, Int]()
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
    val indices = new ArrayBuilder[(HashMap[Int, Array[TbiPair]], Array[Long])](seqs.length)
    i = 0
    while (i < seqs.length) {
      // binning index
      val nBin = readInt(is);
      val binIdx = new HashMap[Int, Array[TbiPair]]()
      j = 0
      while (j < nBin) {
        val bin = readInt(is);
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
    Tabix(format, colSeq, colBeg, meta, seqs, chr2tid, indices.result())
  }

  def queryPairs(tid: Int, beg: Int, end: Int): Array[TbiPair] = {
    if (tid < 0 || tid > index.indicies.length) {
      new Array[TbiPair](0)
    } else {
      val idx = index.indicies(tid)
      val bins = reg2bins(beg, end)
      val minOff = if (idx._2.length > 0 && (beg >> TadLidxShift) >= idx._2.length) {
        idx._2(idx._2.length - 1)
      } else if (idx._2.length > 0) {
        idx._2(beg >> TadLidxShift)
      } else {
        0L
      }
      var i = 0
      var nOff = 0
      while (i < bins.length) {
        nOff += idx._1.get(bins(i)).map(_.length).getOrElse(0)
        i += 1
      }
      var off = new Array[TbiPair](nOff)
      nOff = 0
      i = 0
      while (i < bins.length) {
        val c = idx._1.getOrElse(bins(i), null)
        val len = if (c == null) { 0 } else { c.length }
        var j = 0
        while (j < len) {
          if (TbiOrd.less64(minOff, c(j)._2)) {
            off(nOff) = c(j)
            nOff += 1
          }
          j += 1
        }
      }
      Arrays.sort(off, 0, nOff, null)
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
          off(l)._2 = off(i)._1
        else {
          l += 1
          off(l)._1 = off(i)._1
          off(l)._2 = off(i)._2
        }
        i += 1
      }
      nOff = l + 1
      val ret = Array.fill[TbiPair](nOff)(null)
      i = 0
      while (i < nOff) {
        if (off(i) != null)
          ret(i) = TbiPair(off(i)._1, off(i)._2)
        i += 1
      }
      if (ret.length == 0 || (ret.length == 1 && ret(0) == null))
        new Array[TbiPair](0)
      else
        ret
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

class TabixLineIterator(private val filePath: String, private val offsets: Array[TbiPair])
{
  private val hConf = HailContext.get.sc.hadoopConfiguration
  private var i: Int = -1
  private var curOff: Long = 0 // virtual file offset, not real offset
  private var isEof = false
  private var is = {
    val path = new hd.fs.Path(filePath)
    val fs = path.getFileSystem(hConf)
    new BGzipInputStream(fs.open(path))
  }

  def next(): String = {
    var s: String = null
    while (s == null && !isEof) {
      if (curOff == 0 || !TbiOrd.less64(curOff, offsets(i)._2)) { // jump to next chunk
        if (i == offsets.size - 1) {
          isEof = true
          return s
        }
        if (i >= 0) assert(curOff == offsets(i)._2)
        if (i < 0 || offsets(i)._2 != offsets(i + 1)._1) {
          is.virtualSeek(offsets(i + 1)._1)
          curOff = is.getVirtualOffset()
        }
        i += 1
      }
      s = TabixReader.readLine(is)
      if (s != null) {
        curOff = is.getVirtualOffset()
        if (s.isEmpty() || s.charAt(0) == '#')
          s = null // continue
      } else
        isEof = true
    }
    s
  }
}
