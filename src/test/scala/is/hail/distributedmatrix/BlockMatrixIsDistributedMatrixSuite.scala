package is.hail.distributedmatrix

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

import is.hail.SparkSuite
import org.testng.annotations.Test

class BlockMatrixIsDistributedMatrixSuite extends SparkSuite {
  import is.hail.distributedmatrix.DistributedMatrix.implicits._

  @Test
  def pointwiseSubtractCorrect() {
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val m = dm.from(sc.parallelize(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16))))

    val expected = Array[IndexedSeq[Double]](
      Array[Double](0,-3,-6,-9),
      Array[Double](3,0,-3,-6),
      Array[Double](6,3,0,-3),
      Array[Double](9,6,3,0)):IndexedSeq[IndexedSeq[Double]]

    val actual = (m :- (m.t)).toLocalMatrix().rowIter.map(x => x.toArray: IndexedSeq[Double]).toArray[IndexedSeq[Double]]: IndexedSeq[IndexedSeq[Double]]
    assert(actual == expected)
  }

  @Test
  def multiplyByLocalMatrix() {
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val l = dm.from(sc.parallelize(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16))))

    val r = new DenseMatrix(4, 1, Array[Double](1,2,3,4))

    assert((l.toIndexedRowMatrix().multiply(r).toBlockMatrix().toLocalMatrix().toArray: IndexedSeq[Double]) == ((l * r).toLocalMatrix().toArray: IndexedSeq[Double]))
  }

  @Test
  def multiplyByLocalMatrix2() {
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val l = dm.from(sc.parallelize(Seq(
      Array[Double](-0.0, -0.0, 0.0),
      Array[Double](0.24999999999999994, 0.5000000000000001, -0.5),
      Array[Double](0.4999999999999998, 2.220446049250313E-16, 2.220446049250313E-16),
      Array[Double](0.75, 0.5, -0.5),
      Array[Double](0.25, -0.5, 0.5),
      Array[Double](0.5000000000000001, 1.232595164407831E-32, -2.220446049250313E-16),
      Array[Double](0.75, -0.5000000000000001, 0.5),
      Array[Double](1.0, -0.0, 0.0))))

    val r = new DenseMatrix(3, 4, Array[Double](1.0,0.0,1.0,
      1.0,1.0,1.0,
      1.0,1.0,0.0,
      1.0,0.0,0.0))

    assert((l.toIndexedRowMatrix().multiply(r).toBlockMatrix().toLocalMatrix().toArray: IndexedSeq[Double]) == ((l * r).toLocalMatrix().toArray: IndexedSeq[Double]))
  }

  @Test
  def rowwiseMultiplication() {
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val l = dm.from(sc.parallelize(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)
    )))

    val r = Array[Double](1,2,3,4)

    val result = new DenseMatrix(4,4, Array[Double](
      1,5,9,13,
      4,12,20,28,
      9,21,33,45,
      16,32,48,64
    ))

    assert(dm.toLocalMatrix(l --* r) == result)
  }

  // FIXME: rowmise multiplication random matrices compare with breeze
}
