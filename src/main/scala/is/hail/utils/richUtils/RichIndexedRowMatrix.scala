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

  /**
    * Gets a GridParitioner instance from spark mllib using reflection, since the constructor is private.
    */
  private def sneakyGridPartitioner(nRowBlocks: Int, nColBlocks: Int, suggestedNumPartitions: Int): Partitioner = {
    val intClass = classTag[Int].runtimeClass
    val gpObjectClass = Class.forName("org.apache.spark.mllib.linalg.distributed.GridPartitioner$")
    val gpApply = gpObjectClass.getMethod("apply", intClass, intClass, intClass)

    try {
      gpApply.setAccessible(true)
      gpApply.invoke(gpObjectClass.getField("MODULE$").get(null), nRowBlocks: java.lang.Integer,
        nColBlocks: java.lang.Integer, suggestedNumPartitions: java.lang.Integer).asInstanceOf[Partitioner]
    } finally {
      gpApply.setAccessible(false)
    }
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
    }.groupByKey(sneakyGridPartitioner(numRowBlocks, numColBlocks, indexedRowMatrix.rows.getNumPartitions)).mapValuesWithKey{
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
    new BlockMatrix(blocks, rowsPerBlock, colsPerBlock)
  }
}


