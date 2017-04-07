package is.hail.methods

import org.apache.spark.SparkContext
import org.apache.spark.broadcast._
import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.keytable.KeyTable
import is.hail.variant.{Variant, VariantDataset}
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg.Matrix

import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds
import scala.language.implicitConversions
import scala.reflect.ClassTag

object PCRelate {

  trait DistributedMatrix[M] {
    def from(rdd: RDD[Array[Double]]): M

    def transpose(m: M): M
    def diagonal(m: M): Array[Double]

    def multiply(l: M, r: M): M
    def multiply(l: M, r: DenseMatrix): M

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

        def :+:(r: M): M =
          pointwiseAdd(l, r)
        def :-:(r: M): M =
          pointwiseSubtract(l, r)
        def :*:(r: M): M =
          pointwiseMultiply(l, r)
        def :/:(r: M): M =
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
  }

  object DistributedMatrixImplicits {
    implicit object BlockMatrixIsDistributedMatrix extends DistributedMatrix[BlockMatrix] {
      type M = BlockMatrix

      def from(rdd: RDD[Array[Double]]): M =
        new IndexedRowMatrix(rdd.zipWithIndex().map { case (x, i) => new IndexedRow(i, new DenseVector(x)) })
          .toBlockMatrix()

      def transpose(m: M): M = m.transpose
      def diagonal(m: M): Array[Double] = toCoordinateMatrix(m)
        .entries
        .filter(me => me.i == me.j)
        .map(me => (me.i, me.value))
        .collect()
        .sortBy(_._1)
        .map(_._2)

      def multiply(l: M, r: M): M = l.multiply(r)
      def multiply(l: M, r: DenseMatrix): M =
        l.toIndexedRowMatrix().multiply(r).toBlockMatrix()

      def map2(op: (Double, Double) => Double)(l: M, r: M): M = {
        require(l.numRows() == r.numRows())
        require(l.numCols() == r.numCols())
        require(l.rowsPerBlock == r.rowsPerBlock)
        require(l.colsPerBlock == r.colsPerBlock)
        val blocks: RDD[((Int, Int), Matrix)] = l.blocks.join(r.blocks).map { case (block, (m1, m2)) =>
          (block, new DenseMatrix(m1.numRows, m1.numCols, m1.toArray.zip(m2.toArray).map(op.tupled)))
        }
        new BlockMatrix(blocks, l.rowsPerBlock, l.colsPerBlock, l.numRows(), l.numCols())
      }

      def pointwiseAdd(l: M, r: M): M = map2(_ + _)(l, r)
      def pointwiseSubtract(l: M, r: M): M = map2(_ - _)(l, r)
      def pointwiseMultiply(l: M, r: M): M = map2(_ * _)(l, r)
      def pointwiseDivide(l: M, r: M): M = map2(_ / _)(l, r)

      def map(op: Double => Double)(m: M): M = {
        val blocks: RDD[((Int, Int), Matrix)] = m.blocks.map { case (block, m) =>
          (block, new DenseMatrix(m.numRows, m.numCols, m.toArray.map(op)))
        }
        new BlockMatrix(blocks, m.rowsPerBlock, m.colsPerBlock, m.numRows(), m.numCols())
      }
      def scalarAdd(m: M, i: Double): M = map(_ + i)(m)
      def scalarSubtract(m: M, i: Double): M = map(_ - i)(m)
      def scalarMultiply(m: M, i: Double): M = map(_ * i)(m)
      def scalarDivide(m: M, i: Double): M = map(_ / i)(m)
      def scalarAdd(i: Double, m: M): M = map(i + _)(m)
      def scalarSubtract(i: Double, m: M): M = map(i - _)(m)
      def scalarMultiply(i: Double, m: M): M = map(i * _)(m)
      def scalarDivide(i: Double, m: M): M = map(i / _)(m)

      private def mapWithRowIndex(op: (Double, Int) => Double)(x: M): M = {
        val nRows = x.numRows
        val nCols = x.numCols
        val blocks: RDD[((Int, Int), Matrix)] = x.blocks.map { case ((blockRow, blockCol), m) =>
          ((blockRow, blockCol), new DenseMatrix(m.numRows, m.numCols, m.toArray.zipWithIndex.map { case (e, j) =>
            if (blockRow * x.rowsPerBlock + j % x.colsPerBlock < nRows &&
              blockCol * x.colsPerBlock + j / x.colsPerBlock < nCols)
              op(e, blockRow * x.rowsPerBlock + j % x.colsPerBlock)
            else
              e
          }))
        }
        new BlockMatrix(blocks, x.rowsPerBlock, x.colsPerBlock, x.numRows(), x.numCols())
      }
      private def mapWithColIndex(op: (Double, Int) => Double)(x: M): M = {
        val nRows = x.numRows
        val nCols = x.numCols
        val blocks: RDD[((Int, Int), Matrix)] = x.blocks.map { case ((blockRow, blockCol), m) =>
          ((blockRow, blockCol), new DenseMatrix(m.numRows, m.numCols, m.toArray.zipWithIndex.map { case (e, j) =>
            if (blockRow * x.rowsPerBlock + j % x.colsPerBlock < nRows &&
              blockCol * x.colsPerBlock + j / x.colsPerBlock < nCols)
              op(e, blockCol * x.colsPerBlock + j / x.colsPerBlock)
            else
              e
          }))
        }
        new BlockMatrix(blocks, x.rowsPerBlock, x.colsPerBlock, x.numRows(), x.numCols())
      }

      def vectorAddToEveryColumn(v: Array[Double])(m: M): M = {
        require(v.length == m.numRows())
        mapWithRowIndex((x,i) => x + v(i))(m)
      }
      def vectorPointwiseMultiplyEveryColumn(v: Array[Double])(m: M): M = {
        require(v.length == m.numRows())
        mapWithRowIndex((x,i) => x * v(i))(m)
      }
      def vectorPointwiseMultiplyEveryRow(v: Array[Double])(m: M): M = {
        require(v.length == m.numCols())
        mapWithColIndex((x,i) => x * v(i))(m)
      }

      def mapRows[U](m: M, f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U] =
        m.toIndexedRowMatrix().rows.map((ir: IndexedRow) => f(ir.vector.toArray))

      def toBlockRdd(m: M): RDD[((Int, Int), Matrix)] = m.blocks
      def toCoordinateMatrix(m: M): CoordinateMatrix = m.toCoordinateMatrix()
    }
  }

  case class Result[M: DistributedMatrix](phiHat: M)

  def apply[M: DistributedMatrix](vds: VariantDataset, pcs: DenseMatrix): Result[M] = {
    val g = vdsToMeanImputedMatrix(vds)

    val pcsbc = vds.sparkContext.broadcast(pcs)

    val (beta0, betas) = fitBeta(g, pcsbc)

    val phihat = phiHat(g, muHat(pcsbc, beta0, betas))

    Result(phihat)
  }

  def vdsToMeanImputedMatrix[M: DistributedMatrix](vds: VariantDataset): M = {
    val rdd = vds.rdd.mapPartitions { part =>
      part.map { case (v, (va, gs)) =>
        val goptions = gs.map(_.gt.map(_.toDouble)).toArray
        val defined = goptions.flatMap(x => x)
        val mean = defined.sum / defined.length
        goptions.map(_.getOrElse(mean))
      }
    }
    DistributedMatrix[M].from(rdd)
  }

  /**
    *  g: SNP x Sample
    *  pcs: Sample x D
    *
    *  result: (SNP, SNP x D)
    */
  def fitBeta[M: DistributedMatrix](g: M, pcs: Broadcast[DenseMatrix]): (Array[Double], M) = {
    val aa = pcs.value.rowIter.map(_.toArray).toArray
    val rdd: RDD[(Double, Array[Double])] = DistributedMatrix[M].mapRows(g, { row =>
      val ols = new OLSMultipleLinearRegression()
      ols.newSampleData(row, aa)
      val allBetas = ols.estimateRegressionParameters()
      (allBetas(0), allBetas.slice(1, allBetas.length))
    })
    val vecRdd = rdd.map(_._1)
    val matRdd = rdd.map(_._2)
    (vecRdd.collect(), DistributedMatrix[M].from(matRdd))
  }

  private def clipToInterval(x: Double): Double =
    if (x <= 0)
      Double.MinPositiveValue
    else if (x >= 1)
      1 - Double.MinPositiveValue
    else
      x

  /**
    *  pcs: Sample x D
    *  betas: SNP x D
    *  beta0: SNP
    *
    *  result: SNP x Sample
    */
  def muHat[M: DistributedMatrix](pcs: Broadcast[DenseMatrix], beta0: Array[Double], betas: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    dm.map(clipToInterval _)((betas * pcs.value.transpose :+ beta0) / 2.0)
  }

  /**
    * g: SNP x Sample
    * muHat: SNP x Sample
    **/
  def phiHat[M: DistributedMatrix](g: M, muHat: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    val gMinusMu = g :-: (muHat * 2.0)
    val varianceHat = muHat :*: (1.0 - muHat)

    println("means")
    println(dm.toBlockRdd(muHat * 2.0).take(1)(0))
    println("numerator")
    println(dm.toBlockRdd(gMinusMu.t * gMinusMu).take(1)(0))
    println("denominator")
    println(dm.toBlockRdd(varianceHat.t * varianceHat).take(1)(0))

    ((gMinusMu.t * gMinusMu) :/: (varianceHat.t * varianceHat).sqrt) / 4.0
  }

  def f[M: DistributedMatrix](phiHat: M): Array[Double] =
    DistributedMatrix[M].diagonal(phiHat).map(2.0 * _ - 1.0)

  def k2[M: DistributedMatrix](f: Array[Double], gD: M, mu: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    val normalizedGD = gD :-: ((mu :*: (1.0 - mu)) --* (f.map(1 + _)))
    val variance = mu * (1.0 - mu)

    (normalizedGD.t * normalizedGD) :/: (variance.t * variance)
  }
}
