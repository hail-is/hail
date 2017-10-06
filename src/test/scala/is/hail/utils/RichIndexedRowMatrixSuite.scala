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
    val m = 6L
    val n = 3L
    val data = Seq(
      (0L, Vectors.dense(0.0, 1.0, 2.0)),
      (1L, Vectors.dense(3.0, 4.0, 5.0)),
      (3L, Vectors.dense(9.0, 0.0, 1.0)),
      (4L, Vectors.dense(9.0, 0.0, 1.0)),
      (5L, Vectors.dense(9.0, 0.0, 1.0))
    ).map(IndexedRow.tupled)
    val indexedRows: RDD[IndexedRow] = sc.parallelize(data)

    val idxRowMat = new IndexedRowMatrix(indexedRows)

    def convertDistributedMatrixToBreeze(sparkMatrix: DistributedMatrix): Matrix[Double] = {
      val breezeConverter = sparkMatrix.getClass.getMethod("toBreeze")
      breezeConverter.invoke(sparkMatrix).asInstanceOf[Matrix[Double]]
    }

    val blockMat = idxRowMat.toHailBlockMatrix(2)
    assert(blockMat.rows === m)
    assert(blockMat.cols === n)
    assert(blockMat.toLocalMatrix() === convertDistributedMatrixToBreeze(idxRowMat))

    val blockMat2 = idxRowMat.toHailBlockMatrix(4)
    assert(blockMat2.rows === m)
    assert(blockMat2.cols === n)
    assert(blockMat2.toLocalMatrix() === convertDistributedMatrixToBreeze(idxRowMat))

    val blockMat3 = idxRowMat.toHailBlockMatrix(10)
    assert(blockMat2.rows === m)
    assert(blockMat2.cols === n)
    assert(blockMat2.toLocalMatrix() === convertDistributedMatrixToBreeze(idxRowMat))

    intercept[IllegalArgumentException] {
      idxRowMat.toHailBlockMatrix(-1)
    }
    intercept[IllegalArgumentException] {
      idxRowMat.toHailBlockMatrix(0)
    }
  }
}
