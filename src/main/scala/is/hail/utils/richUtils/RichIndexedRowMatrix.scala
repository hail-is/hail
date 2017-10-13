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
      val block2 = if (block == null) new Array[Double](rowsInBlock * colsInBlock) else block

      var jj = 0
      while (jj < a.length) {
        block2(jj * rowsInBlock + ii) = a(jj)
        jj += 1
      }
      block2
  }

  private def combOp(l: Array[Double], r: Array[Double]): Array[Double] = {
    if (l == null)
      r
    else if (r == null)
      l
    else {
      var k = 0
      while (k < l.length) {
        if (r(k) != 0)
          l(k) = r(k)
        k += 1
      }
      l
    }
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

      val fullColPartitions = if (excessCols == 0) colPartitions else colPartitions - 1
      var j = 0
      while (j < fullColPartitions) {
        val group = new Array[Double](blockSize)
        grouped(j) = ((i.toInt, j.toInt), (i.toInt, j.toInt, ii.toInt, group))
        System.arraycopy(a, j * blockSize, group, 0, blockSize)
        j += 1
      }
      if (excessCols > 0) {
        val group = new Array[Double](excessCols)
        grouped(j) = ((i.toInt, j.toInt), (i.toInt, j.toInt, ii.toInt, group))
        System.arraycopy(a, j * blockSize, group, 0, excessCols)
      }

      grouped.iterator
    }.aggregateByKey(null: Array[Double], partitioner)(
      seqOp(truncatedBlockRow, truncatedBlockCol, excessRows, excessCols, blockSize), combOp)
      .mapValuesWithKey { case ((i, j), a) =>
        val rowsInBlock: Int = if (i == truncatedBlockRow) excessRows else blockSize
        val colsInBlock: Int = if (j == truncatedBlockCol) excessCols else blockSize
        new BDM[Double](rowsInBlock, colsInBlock, a)
    }

    new BlockMatrix(blocks, blockSize, rows, cols)
  }
}


