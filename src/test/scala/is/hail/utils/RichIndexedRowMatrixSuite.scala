package is.hail.utils

import is.hail.{SparkSuite}
import breeze.linalg.{DenseMatrix, Matrix}
import is.hail.distributedmatrix.BlockMatrixIsDistributedMatrix._
import org.apache.spark.mllib.linalg.{Vectors}
import org.apache.spark.mllib.linalg.distributed.{DistributedMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

/**
  * Testing RichIndexedRowMatrix.
  */
class RichIndexedRowMatrixSuite extends SparkSuite {

  @Test def testToBlockMatrixDense() {
    val m = 4
    val n = 3
    val data = Seq(
      (0L, Vectors.dense(0.0, 1.0, 2.0)),
      (1L, Vectors.dense(3.0, 4.0, 5.0)),
      (3L, Vectors.dense(9.0, 0.0, 1.0))
    ).map(IndexedRow.tupled)
    val indexedRows: RDD[IndexedRow] = sc.parallelize(data)

    val idxRowMat = new IndexedRowMatrix(indexedRows)

    def convertDistributedMatrixToBreeze(sparkMatrix: DistributedMatrix): Matrix[Double] = {
      val breezeConverter = sparkMatrix.getClass.getMethod("toBreeze")
      breezeConverter.invoke(sparkMatrix).asInstanceOf[Matrix[Double]]
    }

    // Tests when n % colsPerBlock != 0
    val blockMat = idxRowMat.toBlockMatrixDense(2, 2)
    assert(blockMat.numRows() === m)
    assert(blockMat.numCols() === n)
    assert(convertDistributedMatrixToBreeze(blockMat) === convertDistributedMatrixToBreeze(idxRowMat))

    // Tests when m % rowsPerBlock != 0
    val blockMat2 = idxRowMat.toBlockMatrixDense(3, 1)
    assert(blockMat2.numRows() === m)
    assert(blockMat2.numCols() === n)
    assert(convertDistributedMatrixToBreeze(blockMat2) === convertDistributedMatrixToBreeze(idxRowMat))

    intercept[IllegalArgumentException] {
      idxRowMat.toBlockMatrix(-1, 2)
    }
    intercept[IllegalArgumentException] {
      idxRowMat.toBlockMatrix(2, 0)
    }
  }
  
  @Test def testToLocalMatrix() {
    val m = new DenseMatrix[Double](8, 10, (0.0 until 80.0 by 1.0).toArray)
    val irm = from(sc, m.asSpark(), 3, 3).toIndexedRowMatrix()
    assert(m == irm.toLocalMatrix())
  }
}
