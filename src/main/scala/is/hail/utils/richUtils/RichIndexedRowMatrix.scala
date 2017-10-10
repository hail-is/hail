package is.hail.utils.richUtils

import org.apache.spark.mllib.linalg.distributed._
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.distributedmatrix._
import is.hail.utils._

class RichIndexedRowMatrix(indexedRowMatrix: IndexedRowMatrix) {
  def toHailBlockMatrix(): BlockMatrix = {
    toHailBlockMatrix(1024)
  }

  def toHailBlockMatrix(blockSize: Int): BlockMatrix = {
    require(blockSize > 0,
      s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $blockSize")

    val m = indexedRowMatrix.numRows()
    val n = indexedRowMatrix.numCols()
    val lastRowBlockIndex = m / blockSize
    val lastColBlockIndex = n / blockSize
    val lastRowBlockSize = (m % blockSize).toInt
    val lastColBlockSize = (n % blockSize).toInt
    val numRowBlocks = (m / blockSize + (if (m % blockSize == 0) 0 else 1)).toInt
    val numColBlocks = (n / blockSize + (if (n % blockSize == 0) 0 else 1)).toInt

    val blocks = indexedRowMatrix.rows.flatMap { ir =>
      val blockRow = ir.index / blockSize
      val rowInBlock = ir.index % blockSize

      ir.vector.toArray
        .grouped(blockSize)
        .zipWithIndex
        .map { case (values, blockColumn) =>
          ((blockRow.toInt, blockColumn), (rowInBlock.toInt, values))
        }
    }.groupByKey(HailGridPartitioner(m, n, blockSize)).mapValuesWithKey {
      case ((blockRow, blockColumn), itr) =>
        val actualNumRows: Int = if (blockRow == lastRowBlockIndex) lastRowBlockSize else blockSize
        val actualNumColumns: Int = if (blockColumn == lastColBlockIndex) lastColBlockSize else blockSize

        val arraySize = actualNumRows * actualNumColumns
        val matrixAsArray = new Array[Double](arraySize)
        itr.foreach{ case (rowWithinBlock, values) =>
          var i = 0
          while (i < values.length) {
            matrixAsArray.update(i * actualNumRows + rowWithinBlock, values(i))
            i += 1
          }
        }
        new BDM[Double](actualNumRows, actualNumColumns, matrixAsArray)
    }
    new BlockMatrix(blocks, blockSize, m, n)
  }
}


