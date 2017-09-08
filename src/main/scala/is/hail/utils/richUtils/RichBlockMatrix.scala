package is.hail.utils.richUtils

import org.apache.spark.mllib.linalg.{DenseVector => SparkDenseVector}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import breeze.linalg.{DenseVector, Vector}
import is.hail.sparkextras.{OrderedPartitioner, OrderedRDD}
import org.apache.spark.rdd.ShuffledRDD

class RichBlockMatrix(bm: BlockMatrix) {
  def toIndexedRowMatrixOrderedPartitioner(partitioner: OrderedPartitioner[Int, Int]): IndexedRowMatrix = {
    val cols = bm.numCols().toInt

    require(cols < Int.MaxValue, s"The number of columns must be less than 2^31, got $cols.")

    val rowsPerBlock = bm.rowsPerBlock
    val colsPerBlock = bm.colsPerBlock

    import partitioner.kOk
    val rows = OrderedRDD(new ShuffledRDD[Int, (Int, DenseVector[Double]), (Int, DenseVector[Double])](bm.blocks
      .flatMap { case ((blockRowIdx, blockColIdx), mat) =>
        mat.rowIter.zipWithIndex.map {
          case (vector, rowIdx) =>
            blockRowIdx * rowsPerBlock + rowIdx -> (blockColIdx, new DenseVector[Double](vector.toDense.values))
        }
      }, partitioner)
      .setKeyOrdering(Ordering[Int]), partitioner)
      .groupByKey()
      .map { case (rowIdx, vectors) =>
        val wholeVector = DenseVector.zeros[Double](cols)

        vectors.foreach { case (blockColIdx: Int, vec: Vector[Double]) =>
          val offset = colsPerBlock * blockColIdx
          wholeVector(offset until Math.min(cols, offset + colsPerBlock)) := vec
        }
        IndexedRow(rowIdx, new SparkDenseVector(wholeVector.data))
      }

    new IndexedRowMatrix(rows, bm.numRows(), cols)
  }
}
