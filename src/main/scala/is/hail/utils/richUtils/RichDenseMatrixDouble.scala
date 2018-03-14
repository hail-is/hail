package is.hail.utils.richUtils

import java.io.{InputStream, OutputStream, Closeable}

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.HailContext
import is.hail.annotations.Memory
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata, GridPartitioner}
import is.hail.io._
import is.hail.utils._
import org.json4s.jackson

object RichDenseMatrixDouble {

  def apply(nRows: Int, nCols: Int, data: Array[Double], isTranspose: Boolean = false): BDM[Double] = {
    require(data.length == nRows * nCols)

    new BDM[Double](
      rows = nRows,
      cols = nCols,
      data = data,
      offset = 0,
      majorStride = if (isTranspose) nCols else nRows,
      isTranspose = isTranspose)
  }

  // assumes data isCompact, caller must close
  def read(is: InputStream, bufferSpec: BufferSpec): BDM[Double] = {
    val in = bufferSpec.buildInputBuffer(is)

    val rows = in.readInt()
    val cols = in.readInt()
    val isTranspose = in.readBoolean()

    val data = new Array[Double](rows * cols)
    in.readDoubles(data)

    new BDM[Double](rows, cols, data,
      offset = 0, majorStride = if (isTranspose) cols else rows, isTranspose = isTranspose)
  }

  def read(hc: HailContext, path: String, bufferSpec: BufferSpec): BDM[Double] = {
    hc.hadoopConf.readDataFile(path)(is => read(is, bufferSpec))
  }

  def importFromDoubles(hc: HailContext, path: String, nRows: Int, nCols: Int, rowMajor: Boolean): BDM[Double] = {
    require(nRows * nCols.toLong <= Int.MaxValue)
    val data = new Array[Double](nRows * nCols)

    hc.hadoopConf.readFile(path) { is =>
      val in = new DoubleInputBuffer(is, BlockMatrix.defaultBlockSize << 3)

      in.readDoubles(data)
    }

    RichDenseMatrixDouble(nRows, nCols, data, rowMajor)
  }

  def exportToDoubles(hc: HailContext, path: String, m: BDM[Double], forceRowMajor: Boolean): Boolean = {
    val (data, rowMajor) = m.toCompactData(forceRowMajor)
    assert(data.length == m.rows * m.cols)

    hc.hadoopConf.writeFile(path) { os =>
      val out = new DoubleOutputBuffer(os, BlockMatrix.defaultBlockSize << 3)

      out.writeDoubles(data)
      out.flush()
    }

    rowMajor
  }
}

final class DoubleInputBuffer(in: InputStream, bufSize: Int) extends Closeable {
  private val buf = new Array[Byte](bufSize)
  private var end: Int = 0
  private var off: Int = 0

  def close() {
    in.close()
  }

  def readDoubles(to: Array[Double]): Unit = readDoubles(to, 0, to.length)

  def readDoubles(to: Array[Double], toOff0: Int, n0: Int) {
    assert(toOff0 >= 0)
    assert(n0 >= 0)
    assert(toOff0 <= to.length - n0)

    var toOff = toOff0
    var n = n0

    while (n > 0) {
      if (end == off) {
        val len = math.min(bufSize, n << 3)
        in.readFully(buf, 0, len)
        end = len
        off = 0
      }
      val p = math.min(end - off, n << 3) >>> 3
      assert(p > 0)
      Memory.memcpy(to, toOff, buf, off, p)
      toOff += p
      n -= p
      off += (p << 3)
    }
  }
}

final class DoubleOutputBuffer(out: OutputStream, bufSize: Int) extends Closeable {
  private val buf: Array[Byte] = new Array[Byte](bufSize)
  private var off: Int = 0

  def close() {
    flush()
    out.close()
  }

  def flush() {
    out.write(buf, 0, off)
  }

  def writeDoubles(from: Array[Double]): Unit = writeDoubles(from, 0, from.length)

  def writeDoubles(from: Array[Double], fromOff0: Int, n0: Int) {
    assert(n0 >= 0)
    assert(fromOff0 >= 0)
    assert(fromOff0 <= from.length - n0)
    var fromOff = fromOff0
    var n = n0

    while (off + (n << 3) > buf.length) {
      val p = (buf.length - off) >>> 3
      Memory.memcpy(buf, off, from, fromOff, p)
      off += (p << 3)
      fromOff += p
      n -= p
      out.write(buf, 0, off)
      off = 0
    }
    Memory.memcpy(buf, off, from, fromOff, n)
    off += (n << 3)
  }
}

class RichDenseMatrixDouble(val m: BDM[Double]) extends AnyVal {
  def forceSymmetry() {
    require(m.rows == m.cols, "only square matrices can be made symmetric")

    var i = 0
    while (i < m.rows) {
      var j = i + 1
      while (j < m.rows) {
        m(i, j) = m(j, i)
        j += 1
      }
      i += 1
    }
  }

  def isCompact: Boolean = m.rows * m.cols == m.data.length

  def toCompactData(forceRowMajor: Boolean = false): (Array[Double], Boolean) = {
    if (isCompact && (!forceRowMajor || m.isTranspose))
      (m.data, m.isTranspose)
    else if (forceRowMajor)
      (m.t.toArray, true)
    else
      (m.toArray, false)
  }

  // caller must close
  def write(os: OutputStream, forceRowMajor: Boolean, bufferSpec: BufferSpec) {
    val (data, isTranspose) = m.toCompactData(forceRowMajor)
    assert(data.length == m.rows * m.cols)

    val out = bufferSpec.buildOutputBuffer(os)

    out.writeInt(m.rows)
    out.writeInt(m.cols)
    out.writeBoolean(isTranspose)
    out.writeDoubles(data)
    out.flush()
  }

  def write(hc: HailContext, path: String, forceRowMajor: Boolean = false, bufferSpec: BufferSpec) {
    hc.hadoopConf.writeFile(path)(os => write(os, forceRowMajor, bufferSpec: BufferSpec))
  }

  def writeBlockMatrix(hc: HailContext, path: String, blockSize: Int, forceRowMajor: Boolean = false) {
    val hadoop = hc.hadoopConf
    hadoop.mkDir(path)

    hadoop.writeDataFile(path + BlockMatrix.metadataRelativePath) { os =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, m.rows, m.cols),
        os)
    }

    val gp = GridPartitioner(blockSize, m.rows, m.cols)
    val nParts = gp.numPartitions
    val d = digitsNeeded(nParts)

    (0 until nParts).par.foreach { pi =>
      val filename = path + "/parts/" + partFile(d, pi)

      val (i, j) = gp.blockCoordinates(pi)
      val (blockNRows, blockNCols) = gp.blockDims(pi)
      val iOffset = i * blockSize
      val jOffset = j * blockSize
      var block = m(iOffset until iOffset + blockNRows, jOffset until jOffset + blockNCols)

      block.write(hc, filename, forceRowMajor, BlockMatrix.bufferSpec)
    }

    info(s"wrote $nParts ${ plural(nParts, "item") } in $nParts ${ plural(nParts, "partition") }")

    hadoop.writeTextFile(path + "/_SUCCESS")(out => ())
  }
}
