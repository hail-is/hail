package is.hail.distributedmatrix

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

import scala.reflect.ClassTag

trait DistributedMatrix[M] {
  def cache(m: M): M

  def from(irm: IndexedRowMatrix, dense: Boolean = true): M
  def from(bm: BlockMatrix): M
  def from(cm: CoordinateMatrix): M

  def transpose(m: M): M
  def diagonal(m: M): Array[Double]

  def multiply(l: M, r: M): M
  def multiply(l: M, r: DenseMatrix): M

  def map4(f: (Double, Double, Double, Double) => Double)(a: M, b: M, c: M, d: M): M

  def map2(f: (Double, Double) => Double)(l: M, r: M): M
  def pointwiseAdd(l: M, r: M): M
  def pointwiseSubtract(l: M, r: M): M
  def pointwiseMultiply(l: M, r: M): M
  def pointwiseDivide(l: M, r: M): M

  def map(f: Double => Double)(m: M): M
  def scalarAdd(m: M, i: Double): M
  def scalarSubtract(m: M, i: Double): M
  def scalarMultiply(m: M, i: Double): M
  def scalarDivide(m: M, i: Double): M
  def scalarAdd(i: Double, m: M): M
  def scalarSubtract(i: Double, m: M): M
  def scalarMultiply(i: Double, m: M): M
  def scalarDivide(i: Double, m: M): M

  def vectorAddToEveryColumn(v: Array[Double])(m: M): M
  def vectorPointwiseMultiplyEveryColumn(v: Array[Double])(m: M): M

  def vectorPointwiseMultiplyEveryRow(v: Array[Double])(m: M): M

  def mapRows[U](m: M, f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U]

  def toBlockRdd(m: M): RDD[((Int, Int), Matrix)]
  def toCoordinateMatrix(m: M): CoordinateMatrix

  def toLocalMatrix(m: M): Matrix

  object ops {
    implicit class Shim(l: M) {
      def t: M =
        transpose(l)
      def diag: Array[Double] =
        diagonal(l)

      def *(r: M): M =
        multiply(l, r)
      def *(r: DenseMatrix): M =
        multiply(l, r)

      def :+(r: M): M =
        pointwiseAdd(l, r)
      def :-(r: M): M =
        pointwiseSubtract(l, r)
      def :*(r: M): M =
        pointwiseMultiply(l, r)
      def :/(r: M): M =
        pointwiseDivide(l, r)

      def +(r: Double): M =
        scalarAdd(l, r)
      def -(r: Double): M =
        scalarSubtract(l, r)
      def *(r: Double): M =
        scalarMultiply(l, r)
      def /(r: Double): M =
        scalarDivide(l, r)

      def :+(v: Array[Double]): M =
        vectorAddToEveryColumn(v)(l)
      def :*(v: Array[Double]): M =
        vectorPointwiseMultiplyEveryColumn(v)(l)

      def --*(v: Array[Double]): M =
        vectorPointwiseMultiplyEveryRow(v)(l)

      def sqrt: M =
        map(Math.sqrt _)(l)
      def cache(): M =
        DistributedMatrix.this.cache(l)
    }
    implicit class ScalarShim(l: Double) {
      def +(r: M): M =
        scalarAdd(l, r)
      def -(r: M): M =
        scalarSubtract(l, r)
      def *(r: M): M =
        scalarMultiply(l, r)
      def /(r: M): M =
        scalarDivide(l, r)
    }
  }
}

object DistributedMatrix {
  def apply[M: DistributedMatrix] = implicitly[DistributedMatrix[M]]

  object implicits {
    implicit val blockMatrixIsDistributedMatrix = BlockMatrixIsDistributedMatrix
  }
}
