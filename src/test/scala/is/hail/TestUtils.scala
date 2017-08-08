package is.hail

import breeze.linalg.{DenseMatrix, Matrix, Vector}
import is.hail.annotations.Annotation
import is.hail.keytable.KeyTable
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.sql.Row

object TestUtils {

  import org.scalatest.Assertions._

  def interceptFatal(regex: String)(f: => Any) {
    val thrown = intercept[HailException](f)
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

  def assertDouble(a: Annotation, value: Double, tol: Double = 1e-6) {
    val d1 = a.asInstanceOf[Double]
    assert(D_==(d1, value, tol), s"$d1 did not equal $value with tol $tol")
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

  def indexedSeqBoxedDoubleEquals(tol: Double)
    (xs: IndexedSeq[java.lang.Double], ys: IndexedSeq[java.lang.Double]): Boolean =
    (xs, ys).zipped.forall { case (x, y) =>
      if (x == null || y == null)
        x == null && y == null
      else
        D_==(x.doubleValue(), y.doubleValue(), tolerance = tol)
    }

  def keyTableBoxedDoubleToMap[T](kt: KeyTable): Map[T, IndexedSeq[java.lang.Double]] =
    kt.collect().map { r =>
      val s = r.asInstanceOf[Row].toSeq
      s.head.asInstanceOf[T] -> s.tail.map(_.asInstanceOf[java.lang.Double]).toIndexedSeq
    }.toMap
}
