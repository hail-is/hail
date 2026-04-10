package is.hail.linalg

import is.hail.HailSuite
import is.hail.linalg.BlockMatrix.ops._
import is.hail.linalg.implicits.toRichIndexedRowMatrix

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD

/** Testing RichIndexedRowMatrix. */
class RichIndexedRowMatrixSuite extends HailSuite {

  test("toBlockMatrixDense") {
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

    Seq(1, 2, 3, 4, 6, 7, 9, 10).foreach { blockSize =>
      val blockMat = irm.toHailBlockMatrix(blockSize)
      assertEquals(blockMat.nRows, nRows)
      assertEquals(blockMat.nCols, nCols)
      val blockMatAsBreeze = blockMat.toBreezeMatrix()
      assertEquals(blockMatAsBreeze.rows, irmLocal.numRows)
      assertEquals(blockMatAsBreeze.cols, irmLocal.numCols)
      assertEquals(blockMatAsBreeze.toArray.toIndexedSeq, irmLocal.toArray.toIndexedSeq)
    }

    intercept[IllegalArgumentException] {
      irm.toHailBlockMatrix(-1)
    }: Unit
    intercept[IllegalArgumentException] {
      irm.toHailBlockMatrix(0)
    }: Unit
  }

  test("emptyBlocks") {
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
    assertEquals(m.nRows, nRows.toLong)
    assertEquals(m.nCols, nCols.toLong)
    val blockMatAsBreeze = m.toBreezeMatrix()
    assertEquals(blockMatAsBreeze.toArray.toIndexedSeq, irmLocal.toArray.toIndexedSeq)
    assertEquals(m.blocks.count(), 5L)

    m.dot(m.T).toBreezeMatrix(): Unit // assert no exception

    assertEquals(
      m.mapWithIndex { case (i, j, v) => i + 10 * j + v }.toBreezeMatrix(),
      new BDM[Double](
        nRows,
        nCols,
        Array[Double](
          0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 6.0, 7.0, 9.0,
          10.0, 11.0, 12.0, 15.0, 16.0, 17.0, 16.0, 17.0, 20.0,
        ),
      ),
    )
  }
}
