package is.hail.utils.richUtils

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


    val blocks: RDD[((Int, Int), Matrix)] = indexedRowMatrix.rows.flatMap { ir =>
      val blockRow = ir.index / rowsPerBlock
      val rowInBlock = ir.index % rowsPerBlock

      ir.vector.toArray
        .grouped(colsPerBlock)
        .zipWithIndex
        .map{ case (values, blockColumn) =>
          ((blockRow.toInt, blockColumn), (blockRow.toInt, blockColumn, rowInBlock.toInt, values))
        }
    }.aggregateByKey((rowsPerBlock, colsPerBlock, null: Array[Double]), GridPartitioner(numRowBlocks, numColBlocks)
    )({ (m, row) =>
      val (blockRow, blockCol, rowWithinBlock, values) = row
      val actualNumRows: Int = if (blockRow == lastRowBlockIndex) lastRowBlockSize else rowsPerBlock
      val actualNumColumns: Int = if (blockCol == lastColBlockIndex) lastColBlockSize else colsPerBlock
      val m2 = if (m._3 == null)
        (actualNumRows, actualNumColumns, new Array[Double](actualNumRows * actualNumColumns))
      else
        m

      var i = 0
      while (i < values.length) {
        m2._3.update(i * actualNumRows + rowWithinBlock, values(i))
        i += 1
      }
      m2
    }, { (m1, m2) =>
      var i = 0
      while (i < m1._3.length) {
        val x = m2._3(i)
        if (x != 0)
          m1._3.update(i, x)
        i += 1
      }
      m1
    }).mapValues { case (rows, cols, a) => new DenseMatrix(rows, cols, a) }

    new BlockMatrix(blocks, rowsPerBlock, colsPerBlock)
  }
}

