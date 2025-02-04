package is.hail.utils

import scala.language.implicitConversions

import breeze.linalg.{DenseVector => BDenseVector, SparseVector => BSparseVector, Vector => BVector}
import breeze.linalg.operators.{OpAdd, OpSub}
import org.apache.spark.mllib.linalg.{
  DenseVector => SDenseVector, SparseVector => SSparseVector, Vector => SVector,
}
import org.apache.spark.mllib.linalg.distributed.IndexedRow

trait NumericImplicits {

  implicit def toBDenseVector(v: SDenseVector): BDenseVector[Double] = new BDenseVector(v.values)

  implicit def toBSparseVector(v: SSparseVector): BSparseVector[Double] =
    new BSparseVector(v.indices, v.values, v.size)

  implicit def toBVector(v: SVector): BVector[Double] = v match {
    case v: SSparseVector => v
    case v: SDenseVector => v
  }

  implicit def toSDenseVector(v: BDenseVector[Double]): SDenseVector = new SDenseVector(v.toArray)

  implicit def toSSparseVector(v: BSparseVector[Double]): SSparseVector =
    new SSparseVector(v.length, v.array.index, v.array.data)

  implicit def toSVector(v: BVector[Double]): SVector = v match {
    case v: BDenseVector[Double] => v
    case v: BSparseVector[Double] => v
  }

  implicit object subBVectorSVector extends OpSub.Impl2[BVector[Double], SVector, BVector[Double]] {
    def apply(a: BVector[Double], b: SVector): BVector[Double] = a - toBVector(b)
  }

  implicit object subBVectorIndexedRow
      extends OpSub.Impl2[BVector[Double], IndexedRow, IndexedRow] {
    def apply(a: BVector[Double], b: IndexedRow): IndexedRow =
      IndexedRow(b.index, a - toBVector(b.vector))
  }

  implicit object addBVectorSVector extends OpAdd.Impl2[BVector[Double], SVector, BVector[Double]] {
    def apply(a: BVector[Double], b: SVector): BVector[Double] = a + toBVector(b)
  }

  implicit object addBVectorIndexedRow
      extends OpAdd.Impl2[BVector[Double], IndexedRow, IndexedRow] {
    def apply(a: BVector[Double], b: IndexedRow): IndexedRow =
      IndexedRow(b.index, a + toBVector(b.vector))
  }

}
