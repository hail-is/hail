package is.hail.utils.richUtils

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.distributedmatrix._
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRowMatrix}
import org.apache.spark.rdd.RDD

import is.hail.utils._

/**
  * Adds toBlockMatrixDense method to IndexedRowMatrix
  *
  * The hope is that this class is temporary, as I'm going to make a PR to Spark with this method
  * since it seems broadly useful.
  */

class RichIndexedRowMatrix(indexedRowMatrix: IndexedRowMatrix) {
  def toBlockMatrixDense(): BlockMatrix = {
    toBlockMatrixDense(1024, 1024)
  }

  def toBlockMatrixDense(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
    require(rowsPerBlock > 0,
      s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $rowsPerBlock")
    require(colsPerBlock > 0,
      s"colsPerBlock needs to be greater than 0. colsPerBlock: $colsPerBlock")

    val m = indexedRowMatrix.numRows()
    val n = indexedRowMatrix.numCols()
    val lastRowBlockIndex = m / rowsPerBlock
    val lastColBlockIndex = n / colsPerBlock
    val lastRowBlockSize = (m % rowsPerBlock).toInt
    val lastColBlockSize = (n % colsPerBlock).toInt
    val numRowBlocks = (m / rowsPerBlock + (if (m % rowsPerBlock == 0) 0 else 1)).toInt
    val numColBlocks = (n / colsPerBlock + (if (n % colsPerBlock == 0) 0 else 1)).toInt


    val blocks: RDD[((Int, Int), Matrix)] = indexedRowMatrix.rows.flatMap{ ir =>
      val blockRow = ir.index / rowsPerBlock
      val rowInBlock = ir.index % rowsPerBlock

      ir.vector.toArray
        .grouped(colsPerBlock)
        .zipWithIndex
        .map{ case (values, blockColumn) =>
          ((blockRow.toInt, blockColumn), (rowInBlock.toInt, values))
        }
    }.groupByKey(GridPartitioner(numRowBlocks, numColBlocks)).mapValuesWithKey{
      case ((blockRow, blockColumn), itr) =>
        val actualNumRows: Int = if (blockRow == lastRowBlockIndex) lastRowBlockSize else rowsPerBlock
        val actualNumColumns: Int = if (blockColumn == lastColBlockIndex) lastColBlockSize else colsPerBlock

        val arraySize = actualNumRows * actualNumColumns
        val matrixAsArray = new Array[Double](arraySize)
        itr.foreach{ case (rowWithinBlock, values) =>
          var i = 0
          while (i < values.length) {
            matrixAsArray.update(i * actualNumRows + rowWithinBlock, values(i))
            i += 1
          }
        }
        new DenseMatrix(actualNumRows, actualNumColumns, matrixAsArray)
    }
    new BlockMatrix(blocks, rowsPerBlock, colsPerBlock, m, n)
  }

  def toHailBlockMatrixDense(): HailBlockMatrix = {
    toHailBlockMatrixDense(1024)
  }

  def toHailBlockMatrixDense(blockSize: Int): HailBlockMatrix = {
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


    val blocks: RDD[((Int, Int), BDM[Double])] = indexedRowMatrix.rows.flatMap{ ir =>
      val blockRow = ir.index / blockSize
      val rowInBlock = ir.index % blockSize

      ir.vector.toArray
        .grouped(blockSize)
        .zipWithIndex
        .map{ case (values, blockColumn) =>
          ((blockRow.toInt, blockColumn), (rowInBlock.toInt, values))
        }
    }.groupByKey(HailGridPartitioner(m, n, blockSize)).mapValuesWithKey{
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
    new HailBlockMatrix(blocks, blockSize, m, n)
  }
}


