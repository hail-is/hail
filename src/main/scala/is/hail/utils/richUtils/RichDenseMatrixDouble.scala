package is.hail.utils.richUtils

import breeze.linalg.DenseMatrix
import is.hail.utils.ArrayBuilder
import org.apache.spark.mllib.linalg.{DenseMatrix => SparkDenseMatrix}

object RichDenseMatrixDouble {
  def horzcat(oms: Option[DenseMatrix[Double]]*): Option[DenseMatrix[Double]] = {
    val ms = oms.flatten
    if (ms.isEmpty)
      None
    else
      Some(DenseMatrix.horzcat(ms: _*))
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
  
  def toArrayShallow: Array[Double] =
    if (m.offset == 0 && m.majorStride == m.rows && !m.isTranspose)
      m.data
    else
      m.toArray
  
  def asSpark(): SparkDenseMatrix =
    if (m.offset == 0 && m.majorStride == m.rows)
      if (m.isTranspose)
        new SparkDenseMatrix(m.rows, m.cols, m.data, true)
      else
        new SparkDenseMatrix(m.rows, m.cols, m.data, false)
    else
      new SparkDenseMatrix(m.rows, m.cols, m.toArray, false)
}
