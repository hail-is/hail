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

  trait DistributedMatrix[M] {
    def from(rdd: RDD[Array[Double]]): M

    def transpose(m: M): M

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

    def vectorAddToEveryColumn(v: Array[Double])(m: M): M

    def mapRows[U](m: M, f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U]

    def toBlockRdd(m: M): RDD[((Int, Int), Matrix)]
    def toCoordinateMatrix(m: M): CoordinateMatrix

    object ops {
      implicit class Shim(l: M) {
        def t: M =
          transpose(l)

        def *(r: M): M =
          multiply(l, r)
        def *(r: DenseMatrix): M =
          multiply(l, r)

        def +(r: M): M =
          pointwiseAdd(l, r)
        def -(r: M): M =
          pointwiseSubtract(l, r)
        // def *(r: M): M =
        //   pointwiseMultiply(l, r)
        def /(r: M): M =
          pointwiseDivide(l, r)

        def +(r: Double): M =
          scalarAdd(l, r)
        def -(r: Double): M =
          scalarSubtract(l, r)
        def *(r: Double): M =
          scalarMultiply(l, r)
        def /(r: Double): M =
          scalarDivide(l, r)

        def :+(v: Array[Double]) =
          vectorAddToEveryColumn(v)(l)
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

      def vectorAddToEveryColumn(v: Array[Double])(m: M): M = {
        require(v.length == m.numRows())
        mapWithRowIndex((x,i) => x * v(i))(m)
      }

      def mapRows[U](m: M, f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U] =
        m.toIndexedRowMatrix().rows.map((ir: IndexedRow) => f(ir.vector.toArray))

      def toBlockRdd(m: M): RDD[((Int, Int), Matrix)] = m.blocks
      def toCoordinateMatrix(m: M): CoordinateMatrix = m.toCoordinateMatrix()
    }
  }

  case class Result[M: DistributedMatrix](phiHat: M)

  def pcRelate[M: DistributedMatrix](vds: VariantDataset, pcs: DenseMatrix): Result[M] = {
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

    val gMinusMu = g - (muHat * 2.0)
    val oneMinusMu = dm.map(1 - _)(muHat)
    val varianceHat = dm.pointwiseMultiply(muHat, oneMinusMu)
    println("means")
    println(dm.toBlockRdd(muHat * 2.0).take(1)(0))
    println("numerator")
    println(dm.toBlockRdd(gMinusMu.t * gMinusMu).take(1)(0))
    println("denominator")
    println(dm.toBlockRdd(dm.map(Math.sqrt _)(varianceHat.t * varianceHat)).take(1)(0))

    ((gMinusMu.t * gMinusMu) / dm.map(Math.sqrt _)(varianceHat.t * varianceHat)) / 4.0
  }

}
