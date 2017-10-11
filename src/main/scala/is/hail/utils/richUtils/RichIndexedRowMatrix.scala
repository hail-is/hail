package is.hail.utils.richUtils

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.distributedmatrix._
import is.hail.utils._
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix

object RichIndexedRowMatrix {
  private def seqOp(truncatedBlockRow: Int, truncatedBlockCol: Int, excessRows: Int, excessCols: Int, blockSize: Int)
    (block: Array[Double], row: (Int, Int, Int, Array[Double])): Array[Double] = row match {
    case (i, j, ii, a) =>
      val rowsInBlock: Int = if (i == truncatedBlockRow) excessRows else blockSize
      val colsInBlock: Int = if (j == truncatedBlockCol) excessCols else blockSize

      var jj = 0
      while (jj < a.length) {
        block(jj * rowsInBlock + ii) = a(jj)
        jj += 1
      }
      block
  }

  private def combOp(l: Array[Double], r: Array[Double]): Array[Double] = {
    var k = 0
    while (k < l.length) {
      if (r(k) != 0)
        l(k) = r(k)
      k += 1
    }
    l
  }
}

class RichIndexedRowMatrix(indexedRowMatrix: IndexedRowMatrix) {
  import RichIndexedRowMatrix._

  def toHailBlockMatrix(blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix = {
    require(blockSize > 0,
      s"blockSize needs to be greater than 0. blockSize: $blockSize")

    val rows = indexedRowMatrix.numRows()
    val cols = indexedRowMatrix.numCols()
    val partitioner = GridPartitioner(rows, cols, blockSize)
    val colPartitions = partitioner.colPartitions
    // NB: if excessRows == 0, we never reach the truncatedBlockRow
    val truncatedBlockRow = (rows / blockSize).toInt
    val truncatedBlockCol = (cols / blockSize).toInt
    val excessRows = (rows % blockSize).toInt
    val excessCols = (cols % blockSize).toInt

    val blocks = indexedRowMatrix.rows.flatMap { ir =>
      val i = ir.index / blockSize
      val ii = ir.index % blockSize
      val a = ir.vector.toArray
      val grouped = new Array[((Int, Int), (Int, Int, Int, Array[Double]))](colPartitions)

      var j = 0
      while (j < colPartitions) {
        val group = new Array[Double](blockSize)
        grouped(j) = ((i.toInt, j.toInt), (i.toInt, j.toInt, ii.toInt, group))
        if (j != truncatedBlockCol)
          System.arraycopy(a, j*blockSize, group, 0, blockSize)
        else
          System.arraycopy(a, j*blockSize, group, 0, excessCols)
        j += 1
      }

      grouped.iterator
    }.aggregateByKey(new Array[Double](blockSize * blockSize), partitioner)(
      seqOp(truncatedBlockRow, truncatedBlockCol, excessRows, excessCols, blockSize), combOp)
      .mapValuesWithKey { case ((i, j), a) =>
        if (j == truncatedBlockCol) {
          if (i == truncatedBlockRow) {
            val a2 = new Array[Double](excessRows * excessCols)
            System.arraycopy(a, 0, a2, 0, a2.length)
            new BDM[Double](excessRows, excessCols, a2)
          } else {
            val a2 = new Array[Double](blockSize * excessCols)
            System.arraycopy(a, 0, a2, 0, a2.length)
            new BDM[Double](blockSize, excessCols, a2)
          }
        } else {
          if (i == truncatedBlockRow) {
            val a2 = new Array[Double](excessRows * blockSize)
            System.arraycopy(a, 0, a2, 0, a2.length)
            new BDM[Double](excessRows, blockSize, a2)
          } else {
            new BDM[Double](blockSize, blockSize, a)
          }
        }
    }

    new BlockMatrix(blocks, blockSize, rows, cols)
  }
}


