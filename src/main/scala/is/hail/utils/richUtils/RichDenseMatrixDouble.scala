package is.hail.utils.richUtils

import java.io.{DataInputStream, DataOutputStream}

import breeze.linalg.DenseMatrix
import is.hail.HailContext
import is.hail.annotations.Memory
import is.hail.distributedmatrix.{BlockMatrix, BlockMatrixMetadata, GridPartitioner}
import is.hail.utils._
import org.apache.commons.lang3.StringUtils
import org.json4s.jackson

object RichDenseMatrixDouble {
  // copies n doubles as bytes from data to dos
  def writeDoubles(dos: DataOutputStream, data: Array[Double], n: Int) {
    assert(n <= data.length)
    var nLeft = n
    val bufSize = math.min(nLeft, 8192)
    val buf = new Array[Byte](bufSize << 3) // up to 64KB of doubles

    var off = 0
    while (nLeft > 0) {
      val nCopy = math.min(nLeft, bufSize)

      Memory.memcpy(buf, 0, data, off, nCopy)
      dos.write(buf, 0, nCopy << 3)

      off += nCopy
      nLeft -= nCopy
    }
  }

  // copies n doubles as bytes from dis to data
  def readDoubles(dis: DataInputStream, data: Array[Double], n: Int) {
    assert(n <= data.length)
    var nLeft = n
    val bufSize = math.min(nLeft, 8192)
    val buf = new Array[Byte](bufSize << 3) // up to 64KB of doubles

    var off = 0
    while (nLeft > 0) {
      val nCopy = math.min(nLeft, bufSize)

      dis.readFully(buf, 0, nCopy << 3)
      Memory.memcpy(data, off, buf, 0, nCopy)

      off += nCopy
      nLeft -= nCopy
    }
  }

  // assumes zero offset and minimal majorStride 
  def read(dis: DataInputStream): DenseMatrix[Double] = {
    val rows = dis.readInt()
    val cols = dis.readInt()
    val isTranspose = dis.readBoolean()

    val data = new Array[Double](rows * cols)
    readDoubles(dis, data, data.length)

    new DenseMatrix[Double](rows, cols, data,
      offset = 0, majorStride = if (isTranspose) cols else rows, isTranspose = isTranspose)
  }
  
  def read(hc: HailContext, path: String): DenseMatrix[Double] = {
    hc.hadoopConf.readDataFile(path)(read)
  }

  def from(nRows: Int, nCols: Int, data: Array[Double], isTranspose: Boolean = false): DenseMatrix[Double] = {
    return new DenseMatrix[Double](rows = nRows, cols = nCols, data = data,
      offset = 0, majorStride = nCols, isTranspose = isTranspose)
  }
}

// Not supporting generic T because its difficult to do with ArrayBuilder and not needed yet. See:
// http://stackoverflow.com/questions/16306408/boilerplate-free-scala-arraybuilder-specialization
class RichDenseMatrixDouble(val m: DenseMatrix[Double]) extends AnyVal {
  def filterRows(keepRow: Int => Boolean): Option[DenseMatrix[Double]] = {
    val ab = new ArrayBuilder[Double]()

    var nRows = 0
    for (row <- 0 until m.rows)
      if (keepRow(row)) {
        nRows += 1
        for (col <- 0 until m.cols)
          ab += m.unsafeValueAt(row, col)
      }

    if (nRows > 0)
      Some(new DenseMatrix[Double](rows = nRows, cols = m.cols, data = ab.result(),
        offset = 0, majorStride = m.cols, isTranspose = true))
    else
      None
  }

  def filterCols(keepCol: Int => Boolean): Option[DenseMatrix[Double]] = {
    val ab = new ArrayBuilder[Double]()

    var nCols = 0
    for (col <- 0 until m.cols)
      if (keepCol(col)) {
        nCols += 1
        for (row <- 0 until m.rows)
          ab += m.unsafeValueAt(row, col)
      }

    if (nCols > 0)
      Some(new DenseMatrix[Double](rows = m.rows, cols = nCols, data = ab.result()))
    else
      None
  }

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

  def write(dos: DataOutputStream, forceRowMajor: Boolean) {
    val (data, isTranspose) = m.toCompactData(forceRowMajor)
    assert(data.length == m.rows * m.cols)
    
    dos.writeInt(m.rows)
    dos.writeInt(m.cols)
    dos.writeBoolean(isTranspose)
    
    RichDenseMatrixDouble.writeDoubles(dos, data, data.length)
  }
  
  def write(hc: HailContext, path: String, forceRowMajor: Boolean = false) {
    hc.hadoopConf.writeDataFile(path)(dos => write(dos, forceRowMajor))
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
      val pis = StringUtils.leftPad(pi.toString, d, "0")
      val filename = path + "/parts/part-" + pis
      
      val (i, j) = gp.blockCoordinates(pi)
      val (blockNRows, blockNCols) = gp.blockDims(pi)
      val iOffset = i * blockSize
      val jOffset = j * blockSize      
      var block = m(iOffset until iOffset + blockNRows, jOffset until jOffset + blockNCols)

      block.write(hc, filename, forceRowMajor)
    }

    info(s"wrote $nParts ${ plural(nParts, "item") } in $nParts ${ plural(nParts, "partition") }")

    hadoop.writeTextFile(path + "/_SUCCESS")(out => ())
  }
}
