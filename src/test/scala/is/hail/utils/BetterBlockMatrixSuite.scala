package is.hail.utils

import is.hail.SparkSuite
import is.hail.distributedmatrix.DistributedMatrix
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.testng.annotations.Test

class BetterBlockMatrixSuite extends SparkSuite {
  import is.hail.distributedmatrix.DistributedMatrix.implicits._

  @Test def testMultiply() {
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val rowRDD = sc.parallelize(Seq(
      Array[Double](-0.0, -0.0, 0.0),
      Array[Double](0.24999999999999994, 0.5000000000000001, -0.5),
      Array[Double](0.4999999999999998, 2.220446049250313E-16, 2.220446049250313E-16),
      Array[Double](0.75, 0.5, -0.5),
      Array[Double](0.25, -0.5, 0.5),
      Array[Double](0.5000000000000001, 1.232595164407831E-32, -2.220446049250313E-16),
      Array[Double](0.75, -0.5000000000000001, 0.5),
      Array[Double](1.0, -0.0, 0.0)))

    val irm =new IndexedRowMatrix(rowRDD.zipWithIndex().map{case(values, idx) => IndexedRow(idx, new DenseVector(values))})
    val sbm = irm.toBlockMatrixDense()

    val bbm = dm.from(rowRDD)

    val betterProduct = (bbm * bbm.t).toLocalMatrix().toArray.toIndexedSeq

    val sparkProduct = (sbm multiply sbm.transpose).toLocalMatrix().toArray.toIndexedSeq

    println(betterProduct)

    assert(betterProduct == sparkProduct)

  }
}
