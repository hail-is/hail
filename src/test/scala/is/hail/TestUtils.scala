package is.hail

import breeze.linalg.{DenseMatrix, Matrix, Vector}
import is.hail.utils._
import is.hail.variant.VariantDataset

object TestUtils {

  import org.scalatest.Assertions._

  def interceptFatal(regex: String)(f: => Any) {
    val thrown = intercept[FatalException](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    if (!p)
      println(
        s"""expected fatal exception with pattern `$regex'
           |  Found: ${thrown.getMessage} """.stripMargin)
    assert(p)
  }

  def assertVectorEqualityDouble(A: Vector[Double], B: Vector[Double], tolerance: Double = utils.defaultTolerance) {
    assert(A.size == B.size)
    assert((0 until A.size).forall(i => D_==(A(i), B(i), tolerance)))
  }

  def assertMatrixEqualityDouble(A: Matrix[Double], B: Matrix[Double], tolerance: Double = utils.defaultTolerance) {
    assert(A.rows == B.rows)
    assert(A.cols == B.cols)
    assert((0 until A.rows).forall(i => (0 until A.cols).forall(j => D_==(A(i, j), B(i, j), tolerance))))
  }

  def isConstant(A: Vector[Int]): Boolean = {
    (0 until A.length - 1).foreach(i => if (A(i) != A(i + 1)) return false)
    true
  }

  def removeConstantCols(A: DenseMatrix[Int]): DenseMatrix[Int] = {
    val data = (0 until A.cols).flatMap { j =>
      val col = A(::, j)
      if (TestUtils.isConstant(col))
        Array[Int]()
      else
        col.toArray
    }.toArray

    val newCols = data.length / A.rows
    new DenseMatrix(A.rows, newCols, data)
  }

  // missing is -1
  def vdsToMatrixInt(vds: VariantDataset): DenseMatrix[Int] =
    new DenseMatrix[Int](
      vds.nSamples,
      vds.countVariants.toInt,
      vds.rdd.map(_._2._2.map(_.unboxedGT)).collect().flatten)

  // missing is Double.NaN
  def vdsToMatrixDouble(vds: VariantDataset): DenseMatrix[Double] =
    new DenseMatrix[Double](
      vds.nSamples,
      vds.countVariants.toInt,
      vds.rdd.map(_._2._2.map(_.gt.map(_.toDouble).getOrElse(Double.NaN))).collect().flatten)
}
