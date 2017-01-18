package is.hail.sparkextras

import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, SparseMatrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import is.hail.utils

// Back-ported with modification from Spark 2.0 so as to avoid going through CoordinateMatrix as in Spark 1.5 and 1.6
// Specialized for dense blocks with GRM in mind
object ToIndexedRowMatrixFromBlockMatrix {
  def apply(bm: BlockMatrix): IndexedRowMatrix = {
    val cols = bm.numCols().toInt
    val rowsPerBlock = bm.rowsPerBlock
    val colsPerBlock = bm.colsPerBlock

    require(cols < Int.MaxValue, s"The number of columns should be less than Int.MaxValue ($cols).")

    val rows = bm.blocks.flatMap { case ((blockRowIdx, blockColIdx), mat) =>
      val dmat = mat match {
        case mat: DenseMatrix => mat
        case mat: SparseMatrix => mat.toDense
      }
      rowIter(mat.asInstanceOf[DenseMatrix]).zipWithIndex.map {
        case (vector, rowIdx) =>
          blockRowIdx * rowsPerBlock + rowIdx -> (blockColIdx, utils.toBDenseVector(vector))
      }
    }.groupByKey().map { case (rowIdx, vectors) =>
      val wholeVector = breeze.linalg.DenseVector.zeros[Double](cols)

      vectors.foreach { case (blockColIdx: Int, vec: breeze.linalg.DenseVector[Double]) =>
        val offset = colsPerBlock * blockColIdx
        wholeVector(offset until Math.min(cols, offset + colsPerBlock)) := vec
      }
      IndexedRow(rowIdx, utils.toSDenseVector(wholeVector))
    }
    new IndexedRowMatrix(rows)
  }

  def rowIter(mat: DenseMatrix): Iterator[DenseVector] = colIter(mat.transpose)

  def colIter(mat: DenseMatrix): Iterator[DenseVector] = {
    if (mat.isTransposed) {
      Iterator.tabulate(mat.numCols) { j =>
        val col = new Array[Double](mat.numRows)
        var i = 0
        var k = j
        while (i < mat.numRows) {
          col(i) = mat.values(k)
          i += 1
          k += mat.numCols
        }
        new DenseVector(col)
      }
    } else {
      Iterator.tabulate(mat.numCols) { j =>
        new DenseVector(mat.values.slice(j * mat.numRows, (j + 1) * mat.numRows))
      }
    }
  }
}
