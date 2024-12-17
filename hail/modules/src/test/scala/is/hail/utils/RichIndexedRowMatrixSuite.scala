package is.hail.utils

import is.hail.HailSuite
import is.hail.linalg.BlockMatrix.ops._

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

/** Testing RichIndexedRowMatrix. */
class RichIndexedRowMatrixSuite extends HailSuite {

  @Test def testToBlockMatrixDense(): Unit = {
    val nRows = 9L
    val nCols = 6L
    val data = Seq(
      (0L, Vectors.dense(0.0, 1.0, 2.0, 1.0, 3.0, 4.0)),
      (1L, Vectors.dense(3.0, 4.0, 5.0, 1.0, 1.0, 1.0)),
      (3L, Vectors.dense(9.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
      (4L, Vectors.dense(9.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
      (5L, Vectors.dense(9.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
      (6L, Vectors.dense(1.0, 2.0, 3.0, 1.0, 1.0, 1.0)),
      (7L, Vectors.dense(4.0, 5.0, 6.0, 1.0, 1.0, 1.0)),
      (8L, Vectors.dense(7.0, 8.0, 9.0, 1.0, 1.0, 1.0)),
    ).map(IndexedRow.tupled)
    val indexedRows: RDD[IndexedRow] = sc.parallelize(data)

    val irm = new IndexedRowMatrix(indexedRows)
    val irmLocal = irm.toBlockMatrix().toLocalMatrix()

    for {
      blockSize <- Seq(1, 2, 3, 4, 6, 7, 9, 10)
    } {
      val blockMat = irm.toHailBlockMatrix(blockSize)
      assert(blockMat.nRows === nRows)
      assert(blockMat.nCols === nCols)
      val blockMatAsBreeze = blockMat.toBreezeMatrix()
      assert(blockMatAsBreeze.rows == irmLocal.numRows)
      assert(blockMatAsBreeze.cols == irmLocal.numCols)
      assert(blockMatAsBreeze.toArray.toIndexedSeq == irmLocal.toArray.toIndexedSeq)
    }

    intercept[IllegalArgumentException] {
      irm.toHailBlockMatrix(-1)
    }
    intercept[IllegalArgumentException] {
      irm.toHailBlockMatrix(0)
    }
  }

  @Test def emptyBlocks(): Unit = {
    val nRows = 9
    val nCols = 2
    val data = Seq(
      (3L, Vectors.dense(1.0, 2.0)),
      (4L, Vectors.dense(1.0, 2.0)),
      (5L, Vectors.dense(1.0, 2.0)),
      (8L, Vectors.dense(1.0, 2.0)),
    ).map(IndexedRow.tupled)

    val irm = new IndexedRowMatrix(sc.parallelize(data))
    val irmLocal = irm.toBlockMatrix().toLocalMatrix()

    val m = irm.toHailBlockMatrix(2)
    assert(m.nRows == nRows)
    assert(m.nCols == nCols)
    val blockMatAsBreeze = m.toBreezeMatrix()
    assert(blockMatAsBreeze.toArray.toIndexedSeq == irmLocal.toArray.toIndexedSeq)
    assert(m.blocks.count() == 5)

    (m.dot(m.T)).toBreezeMatrix() // assert no exception

    assert(m.mapWithIndex { case (i, j, v) => i + 10 * j + v }.toBreezeMatrix() ===
      new BDM[Double](
        nRows,
        nCols,
        Array[Double](
          0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 6.0, 7.0, 9.0,
          10.0, 11.0, 12.0, 15.0, 16.0, 17.0, 16.0, 17.0, 20.0,
        ),
      ))
  }
}
