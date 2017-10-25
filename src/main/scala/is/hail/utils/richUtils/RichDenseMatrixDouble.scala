package is.hail.utils.richUtils

import java.io.{DataInputStream, DataOutputStream}

import breeze.linalg.DenseMatrix
import is.hail.utils.ArrayBuilder

object RichDenseMatrixDouble {
  def horzcat(oms: Option[DenseMatrix[Double]]*): Option[DenseMatrix[Double]] = {
    val ms = oms.flatten
    if (ms.isEmpty)
      None
    else
      Some(DenseMatrix.horzcat(ms: _*))
  }
  
  def read(in: DataInputStream): DenseMatrix[Double] = {
    val rows = in.readInt()
    val cols = in.readInt()
    
    val data = new Array[Double](rows * cols)
    var i = 0
    while (i < rows * cols) {
      data(i) = in.readDouble()
      i += 1
    }
    
    new DenseMatrix[Double](rows, cols, data)
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
  
  def write(out: DataOutputStream) {
    out.writeInt(m.rows)
    out.writeInt(m.cols)

    val data = m.toArray
    var i = 0
    while (i < m.rows * m.cols) {
      out.writeDouble(data(i))
      i += 1
    }
  }
}
