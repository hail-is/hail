package is.hail.utils.richUtils

import java.io.{InputStream, OutputStream}

import breeze.linalg.DenseMatrix
import is.hail.HailContext
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata, GridPartitioner}
import is.hail.io.BufferSpec
import is.hail.utils._
import org.json4s.jackson

object RichDenseMatrixDouble {
  // assumes zero offset and minimal majorStride
  // caller must close
  def read(is: InputStream, bufferSpec: BufferSpec): DenseMatrix[Double] = {
    val in = bufferSpec.buildInputBuffer(is)
    
    val rows = in.readInt()
    val cols = in.readInt()
    val isTranspose = in.readBoolean()

    val data = new Array[Double](rows * cols)
    in.readDoubles(data)

    new DenseMatrix[Double](rows, cols, data,
      offset = 0, majorStride = if (isTranspose) cols else rows, isTranspose = isTranspose)
  }
  
  def read(hc: HailContext, path: String, bufferSpec: BufferSpec): DenseMatrix[Double] = {
    hc.hadoopConf.readDataFile(path)(is => read(is, bufferSpec))
  }

  def from(nRows: Int, nCols: Int, data: Array[Double], isTranspose: Boolean = false): DenseMatrix[Double] = {
    new DenseMatrix[Double](rows = nRows, cols = nCols, data = data,
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
