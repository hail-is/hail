package is.hail.utils.richUtils

import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.distributedmatrix._
import is.hail.utils._

class RichIndexedRowMatrix(indexedRowMatrix: IndexedRowMatrix) {
  def toHailBlockMatrix(blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix = {
    require(blockSize > 0,
      s"blockSize needs to be greater than 0. blockSize: $blockSize")

    val rows = indexedRowMatrix.numRows()
    val cols = indexedRowMatrix.numCols()
    // NB: if excessRows == 0, we never reach the truncatedBlockRow
    val truncatedBlockRow = rows / blockSize
    val truncatedBlockCol = cols / blockSize
    val excessRows = (rows % blockSize).toInt
    val excessCols = (cols % blockSize).toInt

    val blocks = indexedRowMatrix.rows.flatMap { ir =>
      val i = ir.index / blockSize
      val ii = ir.index % blockSize

      ir.vector.toArray
        .grouped(blockSize)
        .zipWithIndex
        .map { case (values, blockColumn) =>
          ((i.toInt, blockColumn), (ii.toInt, values))
        }
    }.groupByKey(GridPartitioner(rows, cols, blockSize)).mapValuesWithKey {
      case ((i, j), it) =>
        val rowsInBlock: Int = if (i == truncatedBlockRow) excessRows else blockSize
        val colsInBlock: Int = if (j == truncatedBlockCol) excessCols else blockSize

        val a = new Array[Double](rowsInBlock * colsInBlock)
        it.foreach{ case (ii, values) =>
          var jj = 0
          while (jj < values.length) {
            a(jj * rowsInBlock + ii) = values(jj)
            jj += 1
          }
        }
        new BDM[Double](rowsInBlock, colsInBlock, a)
    }
    new BlockMatrix(blocks, blockSize, rows, cols)
  }
}


