package is.hail.utils

import breeze.linalg.Matrix
import is.hail.SparkSuite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{DistributedMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

/**
  * Testing RichIndexedRowMatrix.
  */
class RichIndexedRowMatrixSuite extends SparkSuite {

  @Test def testToBlockMatrixDense() {
    val rows = 9L
    val cols = 6L
    val data = Seq(
      (0L, Vectors.dense(0.0, 1.0, 2.0)),
      (1L, Vectors.dense(3.0, 4.0, 5.0)),
      (3L, Vectors.dense(9.0, 0.0, 1.0)),
      (4L, Vectors.dense(9.0, 0.0, 1.0)),
      (5L, Vectors.dense(9.0, 0.0, 1.0)),
      (6L, Vectors.dense(1.0, 2.0, 3.0)),
      (7L, Vectors.dense(4.0, 5.0, 6.0)),
      (8L, Vectors.dense(7.0, 8.0, 9.0)),
      (9L, Vectors.dense(10.0, 11.0, 12.0))
    ).map(IndexedRow.tupled)
    val indexedRows: RDD[IndexedRow] = sc.parallelize(data)

    val idxRowMat = new IndexedRowMatrix(indexedRows)

    def convertDistributedMatrixToBreeze(sparkMatrix: DistributedMatrix): Matrix[Double] = {
      val breezeConverter = sparkMatrix.getClass.getMethod("toBreeze")
      breezeConverter.invoke(sparkMatrix).asInstanceOf[Matrix[Double]]
    }

    for {
      blockSize <- Seq(1,2,3,4,6,7,9,10)
    } {
      val blockMat = idxRowMat.toHailBlockMatrix(blockSize)
      assert(blockMat.rows === rows)
      assert(blockMat.cols === cols)
      assert(blockMat.toLocalMatrix() === convertDistributedMatrixToBreeze(idxRowMat))
    }

    intercept[IllegalArgumentException] {
      idxRowMat.toHailBlockMatrix(-1)
    }
    intercept[IllegalArgumentException] {
      idxRowMat.toHailBlockMatrix(0)
    }
  }
}
