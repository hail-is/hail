package is.hail.linalg

import _root_.breeze.linalg.operators.{OpAdd, OpSub}
import breeze.{linalg => breeze}
import org.apache.spark.mllib.{linalg => spark}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

package object implicits {
  implicit def toRichDenseMatrixDouble(m: breeze.DenseMatrix[Double]): RichDenseMatrixDouble =
    new RichDenseMatrixDouble(m)

  implicit def toRichIndexedRowMatrix(irm: IndexedRowMatrix): RichIndexedRowMatrix =
    new RichIndexedRowMatrix(irm)

  implicit def sparkToBreezeDenseVector(v: spark.DenseVector): breeze.DenseVector[Double] =
    new breeze.DenseVector(v.values)

  implicit def sparkToBreezeSparseVector(v: spark.SparseVector): breeze.SparseVector[Double] =
    new breeze.SparseVector(v.indices, v.values, v.size)

  implicit def sparkToBreezeVector(v: spark.Vector): breeze.Vector[Double] =
    v match {
      case v: spark.SparseVector => v
      case v: spark.DenseVector => v
    }

  implicit def breezeToSparkDenseVector(v: breeze.DenseVector[Double]): spark.DenseVector =
    new spark.DenseVector(v.toArray)

  implicit def breezeToSparkSparseVector(v: breeze.SparseVector[Double]): spark.SparseVector =
    new spark.SparseVector(v.length, v.array.index, v.array.data)

  implicit def breezeToSparkVector(v: breeze.Vector[Double]): spark.Vector =
    v match {
      case v: breeze.DenseVector[Double] => v
      case v: breeze.SparseVector[Double] => v
    }

  implicit lazy val subBVectorSVector
    : OpSub.Impl2[breeze.Vector[Double], spark.Vector, breeze.Vector[Double]] =
    (a: breeze.Vector[Double], b: spark.Vector) =>
      a - sparkToBreezeVector(b)

  implicit lazy val subBVectorIndexedRow
    : OpSub.Impl2[breeze.Vector[Double], IndexedRow, IndexedRow] =
    (a: breeze.Vector[Double], b: IndexedRow) =>
      IndexedRow(b.index, a - sparkToBreezeVector(b.vector))

  implicit lazy val addBVectorSVector
    : OpAdd.Impl2[breeze.Vector[Double], spark.Vector, breeze.Vector[Double]] =
    (a: breeze.Vector[Double], b: spark.Vector) =>
      a + sparkToBreezeVector(b)

  implicit lazy val addBVectorIndexedRow
    : OpAdd.Impl2[breeze.Vector[Double], IndexedRow, IndexedRow] =
    (a: breeze.Vector[Double], b: IndexedRow) =>
      IndexedRow(b.index, a + sparkToBreezeVector(b.vector))
}
