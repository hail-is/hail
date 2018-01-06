package is.hail.utils.richUtils

import java.io.{DataInputStream, DataOutputStream}

import breeze.linalg.DenseMatrix
import is.hail.annotations.Memory
import is.hail.utils.ArrayBuilder

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
  
  def write(dos: DataOutputStream) {
    assert(m.offset == 0)
    assert(m.majorStride == (if (m.isTranspose) m.cols else m.rows))

    dos.writeInt(m.rows)
    dos.writeInt(m.cols)
    assert(m.data.length == m.rows * m.cols)
    dos.writeBoolean(m.isTranspose)
    
    RichDenseMatrixDouble.writeDoubles(dos, m.data, m.data.length)
  }
  
  def forceRowMajor(): DenseMatrix[Double] =
    if (m.isTranspose)
      m
    else
      new DenseMatrix(m.rows, m.cols, m.t.toArray, 0, m.cols, isTranspose = true)
}
