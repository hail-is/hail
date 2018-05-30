package is.hail.utils.richUtils

import java.io.{InputStream, OutputStream}

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.HailContext
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
    val data = RichArray.importFromDoubles(hc, path, nRows * nCols)

    RichDenseMatrixDouble(nRows, nCols, data, rowMajor)
  }

  def exportToDoubles(hc: HailContext, path: String, m: BDM[Double], forceRowMajor: Boolean): Boolean = {
    val (data, rowMajor) = m.toCompactData(forceRowMajor)
    assert(data.length == m.rows * m.cols)
    
    RichArray.exportToDoubles(hc, path, data)

    rowMajor
  }
}

class RichDenseMatrixDouble(val m: BDM[Double]) extends AnyVal {
  // dot is overloaded in Breeze
  def matrixMultiply(bm: BlockMatrix): BlockMatrix = {
    require(m.cols == bm.nRows,
      s"incompatible matrix dimensions: ${ m.rows } x ${ m.cols } and ${ bm.nRows } x ${ bm.nCols } ")
    BlockMatrix.fromBreezeMatrix(bm.blocks.sparkContext, m, bm.blockSize).dot(bm)
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

    val gp = GridPartitioner(blockSize, m.rows, m.cols)
    val nParts = gp.numPartitions
    val d = digitsNeeded(nParts)

    val partFiles = (0 until nParts).par.map { pi =>
      val f = partFile(d, pi)
      val filename = path + "/parts/" + f

      val (i, j) = gp.blockCoordinates(pi)
      val (blockNRows, blockNCols) = gp.blockDims(pi)
      val iOffset = i * blockSize
      val jOffset = j * blockSize
      var block = m(iOffset until iOffset + blockNRows, jOffset until jOffset + blockNCols)

      block.write(hc, filename, forceRowMajor, BlockMatrix.bufferSpec)

      f
    }
      .toArray

    hadoop.writeDataFile(path + BlockMatrix.metadataRelativePath) { os =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, m.rows, m.cols, gp.maybeBlocks, partFiles),
        os)
    }

    info(s"wrote $nParts ${ plural(nParts, "item") } in $nParts ${ plural(nParts, "partition") }")

    hadoop.writeTextFile(path + "/_SUCCESS")(out => ())
  }
}
