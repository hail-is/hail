package is.hail.methods


import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.keytable.KeyTable
import is.hail.variant.{Genotype, Variant, VariantDataset}
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds
import scala.reflect.ClassTag

object PCRelate {

  /**
    *
    * @param vds
    * @param mean (variant, (sample, mean))
    * @return
    */
  def apply(vds: VariantDataset, mean: RDD[(Variant, (Int, Double))]): RDD[((String, String), Double)] = {
    assert(vds.wasSplit, "PCRelate requires biallelic VDSes")

    // (variant, (sample, gt))
    val g = vds.rdd.flatMap { case (v, (va, gs)) =>
      gs.zipWithIndex.map { case (g, i) =>
        (v, (i, g.nNonRefAlleles.getOrElse[Int](-1): Double)) } }

    val meanPairs = mean.join(mean)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, mean1), (s2, mean2))) =>
        ((s1, s2, vi), (mean1, mean2))
    }

    val numerator = g.join(g)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, gt1), (s2, gt2))) =>
        ((s1, s2, vi), (gt1, gt2))
    }
      .join(meanPairs)
      .map { case ((s1, s2, vi), ((gt1, gt2), (mean1, mean2))) =>
        ((s1, s2), (gt1 - 2 * mean1) * (gt2 - 2 * mean2))
    }
      .reduceByKey(_ + _)

    val denominator = mean.join(mean)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, mean1), (s2, mean2))) =>
        ((s1, s2), Math.sqrt(mean1 * (1 - mean1) * mean2 * (1 - mean2)))
    }
      .reduceByKey(_ + _)

    val sampleIndexToId =
      vds.sampleIds.zipWithIndex.map { case (s, i) => (i, s) }.toMap

    numerator.join(denominator)
      .map { case ((s1, s2), (numerator, denominator)) => ((s1, s2), numerator / denominator / 4) }
      .map { case ((s1, s2), x) => ((sampleIndexToId(s1), sampleIndexToId(s2)), x) }
  }

  trait Vector[T <: Vector[T]] {
    def apply(i: Int): Double
    def plus(that: T): T
    def size: Long
  }
  implicit class ArrayIsVector(val a: Array[Double]) extends Vector[ArrayIsVector] {
    type V = ArrayIsVector

    def apply(i: Int): Double = a(i)
    def plus(that: V): V = new ArrayIsVector(a.zip(that.a).map { case (x: Double, y: Double) => x + y } )
    def size: Long = a.length
  }
  object Vector {
    def from(rdd: RDD[Double]): ArrayIsVector = {
      require(rdd.count() < Int.MaxValue)
      new ArrayIsVector(rdd.collect())
    }
  }
  type MyVector = Vector[ArrayIsVector]

  trait Matrix[T <: Matrix[T]] {
    def multiply(that: T): T
    def transpose: T
    def pointwiseAdd(that: T): T
    def pointwiseSubtract(that: T): T
    def pointwiseMultiply(that: T): T
    def pointwiseDivide(that: T): T
    def scalarMultiply(i: Double): T
    def scalarAdd(i: Double): T

    def vectorExtendAddRowWise(v: MyVector): T

    def toArrayArray: Array[Array[Double]]

    def mapRows[U](f: Array[Double] => U): RDD[U]
  }
  object BetterIndexedRowMatrix {
    private def irmToKvRdd(x: IndexedRowMatrix) =
      x.rows.map { case IndexedRow(i, v) => (i, v) }
    private def pointwiseOp(op: (Double, Double) => Double)(x: IndexedRowMatrix, y: IndexedRowMatrix): IndexedRowMatrix = {
      require(x.numRows() == y.numRows())
      require(x.numCols() == y.numCols())
      val rows = irmToKvRdd(x).join(irmToKvRdd(y)).map { case (i, (v1, v2)) =>
        IndexedRow(i, new DenseVector(v1.toArray.zip(v2.toArray).map(op.tupled))) }
      new IndexedRowMatrix(rows, x.numRows(), x.numCols().toInt)
    }
    private def elementMap(op: (Double) => Double)(x: IndexedRowMatrix): IndexedRowMatrix = {
      val rows = x.rows.map { case IndexedRow(i, v) => IndexedRow(i, new DenseVector(v.toArray.map(op))) }
      new IndexedRowMatrix(rows, x.numRows(), x.numCols().toInt)
    }
    private def elementMapWithRowIndex(op: (Double, Int) => Double)(x: IndexedRowMatrix): IndexedRowMatrix = {
      val rows = x.rows.map { case IndexedRow(i, v) => IndexedRow(i, new DenseVector(v.toArray.zipWithIndex.map(op.tupled))) }
      new IndexedRowMatrix(rows, x.numRows(), x.numCols().toInt)
    }
    private def transposeIrm(x: IndexedRowMatrix): IndexedRowMatrix = {
      require(x.numRows() < Int.MaxValue)
      val cols = x.rows
        .flatMap { case IndexedRow(i, v) => v.toArray.zipWithIndex.map { case (j, x) => (j, (i, x)) } }
        .aggregateByKey(Array.fill(x.numCols().toInt)(0))(
          { case (a, (i, x)) => a(i.toInt) = x; a },
          { case (l, r) =>
            var i = 0
            while (i < l.length) { l(i) = r(i) }
            l
          })
        .map { case (i, a) => IndexedRow(i, new DenseVector(a)) }
      new IndexedRowMatrix(cols)
    }
  }
  implicit class BetterIndexedRowMatrix(val irm: IndexedRowMatrix) extends Matrix[BetterIndexedRowMatrix] {
    import BetterIndexedRowMatrix._
    type M = BetterIndexedRowMatrix

    private def lift(f: IndexedRowMatrix => IndexedRowMatrix) =
      new BetterIndexedRowMatrix(f(this.irm))
    private def lift2(f: (IndexedRowMatrix, IndexedRowMatrix) => IndexedRowMatrix)(that: BetterIndexedRowMatrix) =
      new BetterIndexedRowMatrix(f(this.irm, that.irm))

    def multiply(that: M): M = lift2(_.multiply(_))(that)
    def transpose: M = lift(transposeIrm)

    def pointwiseAdd(that: M): M =
      lift2(pointwiseOp(_ - _))(that)
    def pointwiseSubtract(that: M): M =
      lift2(pointwiseOp(_ - _))(that)
    def pointwiseMultiply(that: M): M =
      lift2(pointwiseOp(_ * _))(that)
    def pointwiseDivide(that: M): M =
      lift2(pointwiseOp(_ / _))(that)
    def scalarMultiply(i: Double): M =
      lift(elementMap(_ * i))
    def scalarAdd(i: Double): M =
      lift(elementMap(_ + i))
    def vectorExtendAddRowWise(v: MyVector): M = {
      require(v.size == this.irm.numRows())
      lift(elementMapWithRowIndex((i,x) => x * v(i)))
    }

    def toArrayArray: Array[Array[Double]] = {
      require(this.irm.numRows() < Int.MaxValue)
      this.irm.rows.collect()
        .sortBy(_.index)
        .map(_.vector.toArray)
        .toArray
    }

    def mapRows[U](f: Array[Double] => U): RDD[U] =
      this.irm.rows.map(f(_.vector.toArray))
  }
  object Matrix {
    // def from(rdd: RDD[Array[Double]]): Matrix = ???
    def from(sc: SparkContext, dm: DenseMatrix): Matrix = {
      val m = dm.toArray
      val rows = Array.fill(dm.numRows, dm.numCols)(0)

      for (
        i <- 0 until dm.numRows;
        j <- 0 until dm.numCols
      ) { rows(i)(j) = m(i*dm.numCols + j) }

      val indexedRows = rows
        .zipWithIndex
        .map { case (i, a) => IndexedRow(i, new DenseVector(a)) }

      new IndexedRowMatrix(sc.parallelize(indexedRows))
    }
  }

  type MyMatrix = Matrix[BetterIndexedRowMatrix]

  case class Result(phiHat: MyMatrix)

  def pcRelate(vds: VariantDataset, pcs: MyMatrix): Result = {
    val g = vdsToMeanImputedMatrix(vds)

    val (beta0, betas) = fitBeta(g, pcs)

    Result(phiHat(g, muHat(pcs, beta0, betas)))
  }

  def vdsToMeanImputedMatrix(vds: VariantDataset): MyMatrix = {
    val rdd = vds.rdd.mapPartitions { stuff =>
      val ols = new OLSMultipleLinearRegression()
      stuff.map { case (v, (va, gs)) =>
        val goptions = gs.map(_.gt.map(_.toDouble)).toArray
        val defined = goptions.flatMap(x => x)
        val mean: Double = defined.sum / defined.size
        goptions.map(_.getOrElse(mean))
      }
    }
    Matrix.from(rdd)
  }

  /**
    *  g: SNP x Sample
    *  pcs: Sample x D
    *
    *  result: (D, SNP x D)
    */
  def fitBeta(g: MyMatrix, pcs: MyMatrix): (MyVector, MyMatrix) = {
    val rdd: RDD[(Double, Array[Double])] = g.mapRows { row =>
      val ols = new OLSMultipleLinearRegression()
      ols.newSampleData(row, pcs.toArrayArray)
      val allBetas = ols.estimateRegressionParameters().toArray
      (allBetas(0), allBetas.slice(1,allBetas.size))
    }
    val vecRdd = rdd.map(_._1)
    val matRdd = rdd.map(_._2)
    (Vector.from(vecRdd), Matrix.from(matRdd))
  }

  /**
    *  pcs: Sample x D
    *  betas: SNP x D
    *  beta0: SNP
    *
    *  result: SNP x Sample
    */
  def muHat(pcs: MyMatrix, beta0: MyVector, betas: MyMatrix): MyMatrix =
    betas.multiply(pcs.transpose).vectorExtendAddRowWise(beta0)

  def phiHat(g: MyMatrix, muHat: MyMatrix): MyMatrix = {
    val gMinusMu = g.pointwiseSubtract(muHat)
    val oneMinusMu = muHat.scalarMultiply(-1).scalarAdd(1)
    val varianceHat = muHat.pointwiseMultiply(oneMinusMu)
    gMinusMu.multiply(gMinusMu.transpose)
      .pointwiseDivide(varianceHat.multiply(varianceHat))
  }

}
