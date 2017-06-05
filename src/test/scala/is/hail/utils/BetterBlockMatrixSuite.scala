package is.hail.utils

import is.hail.SparkSuite
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.distributedmatrix.DistributedMatrix
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.testng.annotations.Test
import breeze.linalg._
import scala.util.Random

class BetterBlockMatrixSuite extends SparkSuite {
  import is.hail.distributedmatrix.DistributedMatrix.implicits._
  val dm = DistributedMatrix[BlockMatrix]
  import dm.ops._


  @Test def testMultiplyRandom() {

    val r = new Random()
    val m = r.nextInt(100) + 30
    val n = r.nextInt(100) + 30
    val k = r.nextInt(100) + 30
    val matGen1 = Gen.denseMatrix[Double](m, n)(Gen.choose(-100.0, 100.0))
    val matGen2 = Gen.denseMatrix[Double](n, k)(Gen.choose(-100.0, 100.0))

    forAll(matGen1, matGen2) { (mat1, mat2) =>
      val range1 = 0 until(m)
      val irs1  = range1.map(i => IndexedRow(i, mat1(i, ::).t))

      val range2 = 0 until n
      val irs2 = range2.map(i => IndexedRow(i, mat2(i, ::).t))

      val mBlock = 1 + r.nextInt(m)
      val nBlock = 1 + r.nextInt(n)
      val kBlock = 1 + r.nextInt(k)
      val bmat1 = new IndexedRowMatrix(sc.parallelize(irs1)).toBlockMatrixDense(mBlock, nBlock)
      val bmat2 = new IndexedRowMatrix(sc.parallelize(irs2)).toBlockMatrixDense(nBlock, kBlock)

      val betterResultArray = (bmat1 * bmat2).toLocalMatrix().toArray
      val breezeResultArray = (mat1 * mat2).toArray

      breezeResultArray.zip(betterResultArray).forall{case(d1, d2) => D_==(d1, d2)}
    }.check()
  }

}
