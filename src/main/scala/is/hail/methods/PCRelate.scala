package is.hail.methods

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, TString, TFloat64}
import is.hail.utils._
import is.hail.keytable.KeyTable
import is.hail.variant.{Variant, VariantDataset}
import is.hail.distributedmatrix.BlockMatrixIsDistributedMatrix
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds
import scala.language.implicitConversions

object PCRelate {
  type M = BlockMatrixIsDistributedMatrix.M
  val dm = BlockMatrixIsDistributedMatrix
  import dm.ops._

  type StatisticSubset = Int
  val PhiOnly: StatisticSubset = 0
  val PhiK2: StatisticSubset = 1
  val PhiK2K0: StatisticSubset = 2
  val PhiK2K0K1: StatisticSubset = 3

  case class Result[M](phiHat: M, k0: M, k1: M, k2: M) {
    def map[N](f: M => N): Result[N] = Result(f(phiHat), f(k0), f(k1), f(k2))
  }

  val defaultMinKinship = Double.NegativeInfinity
  val defaultStatisticSubset: StatisticSubset = PhiK2K0K1

  def apply(vds: VariantDataset, pcs: DenseMatrix, maf: Double, blockSize: Int, statistics: StatisticSubset = defaultStatisticSubset): Result[M] =
    new PCRelate(maf, blockSize)(vds, pcs, statistics)

  private val signature =
    TStruct(("i", TString), ("j", TString), ("kin", TFloat64), ("k0", TFloat64), ("k1", TFloat64), ("k2", TFloat64))
  private val keys = Array("i", "j")

  private def toRowRdd(vds: VariantDataset, pcs: DenseMatrix, maf: Double, blockSize: Int, minKinship: Double, statistics: StatisticSubset): RDD[Row] = {
    val indexToId: Map[Int, Annotation] = vds.sampleIds.zipWithIndex.map { case (id, index) => (index, id) }.toMap
    val Result(phi, k0, k1, k2) = apply(vds, pcs, maf, blockSize, statistics)

    def fuseBlocks(blocki: Int, blockj: Int, mphi: Matrix, mk0: Matrix, mk1: Matrix, mk2: Matrix) = {
      val i = blocki * phi.rowsPerBlock
      val j = blockj * phi.colsPerBlock
      val i2 = i + phi.rowsPerBlock
      val j2 = j + phi.colsPerBlock

      if (blocki <= blockj) {
        val size = mphi.numRows * mphi.numCols
        val ab = new ArrayBuilder[Row]()
          var jj = 1
          while (jj < mphi.numCols) {
            // fixme: broken for non-square blocks
            var ii = 0
            val rowsAboveDiagonal = if (blocki < blockj) mphi.numRows else jj
            while (ii < rowsAboveDiagonal) {
              val kin = mphi(ii, jj)
              if (kin >= minKinship) {
                val k0 = if (mk0 == null) null else mk0(ii, jj)
                val k1 = if (mk1 == null) null else mk1(ii, jj)
                val k2 = if (mk2 == null) null else mk2(ii, jj)
                ab += Annotation(indexToId(i + ii), indexToId(j + jj), kin, k0, k1, k2).asInstanceOf[Row]
              }
              ii += 1
            }
          jj += 1
        }
        ab.result()
      } else
        new Array[Row](0)
    }

    statistics match {
      case PhiOnly => phi.blocks
          .flatMap { case ((blocki, blockj), phi) => fuseBlocks(blocki, blockj, phi, null, null, null) }
      case PhiK2 => (phi.blocks join k2.blocks)
          .flatMap { case ((blocki, blockj), (phi, k2)) => fuseBlocks(blocki, blockj, phi, null, null, k2) }
      case PhiK2K0 => (phi.blocks join k0.blocks join k2.blocks)
          .flatMap { case ((blocki, blockj), ((phi, k0), k2)) => fuseBlocks(blocki, blockj, phi, k0, null, k2) }
      case PhiK2K0K1 => (phi.blocks join k0.blocks join k1.blocks join k2.blocks)
          .flatMap { case ((blocki, blockj), (((phi, k0), k1), k2)) => fuseBlocks(blocki, blockj, phi, k0, k1, k2) }
    }
  }

  def toKeyTable(vds: VariantDataset, pcs: DenseMatrix, maf: Double, blockSize: Int, minKinship: Double = defaultMinKinship, statistics: StatisticSubset = defaultStatisticSubset): KeyTable =
    KeyTable(vds.hc, toRowRdd(vds, pcs, maf, blockSize, minKinship, statistics), signature, keys)

  private val k0cutoff = math.pow(2.0, (-5.0 / 2.0))

  private def prependConstantColumn(k: Double, m: DenseMatrix): DenseMatrix = {
    val a = m.toArray
    val result = new Array[Double](m.numRows * (m.numCols + 1))
    var i = 0
    while (i < m.numRows) {
      result(i) = 1
      i += 1
    }
    i = 0
    while (i < m.numRows * m.numCols) {
      result(m.numRows + i) = a(i)
      i += 1
    }
    new DenseMatrix(m.numRows, m.numCols + 1, result)
  }

  def vdsToMeanImputedMatrix(vds: VariantDataset): IndexedRowMatrix = {
    val nSamples = vds.nSamples
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)
    val rdd = vds.rdd.mapPartitions { part =>
      part.map { case (v, (va, gs)) =>
        var sum = 0
        var nNonMissing = 0
        val missingIndices = new ArrayBuilder[Int]()
        val a = new Array[Double](nSamples)

        var i = 0
        val it = gs.hardCallIterator
        while (it.hasNext) {
          val gt = it.next()
          if (gt == -1) {
            missingIndices += i
          } else {
            sum += gt
            a(i) = gt
            nNonMissing += 1
          }
          i += 1
        }

        val mean = sum.toDouble / nNonMissing

        for (i <- missingIndices.result()) {
          a(i) = mean
        }

        // FIXME: this should probably be a sparse vector
        new IndexedRow(variantIdxBc.value(v), new DenseVector(a))
      }
    }
    new IndexedRowMatrix(rdd, variants.length, nSamples)
  }

  /**
    *  g: SNP x Sample
    *  pcs: Sample x D
    *
    *  result: (SNP x (D+1))
    */
  def fitBeta(g: IndexedRowMatrix, pcs: DenseMatrix, blockSize: Int): M = {
    val aa = g.rows.sparkContext.broadcast(pcs.rowIter.map(_.toArray).toArray)
    val rdd = g.rows.map { case IndexedRow(i, v) =>
      val ols = new OLSMultipleLinearRegression()
      ols.newSampleData(v.toArray, aa.value)
      val a = ols.estimateRegressionParameters()
      IndexedRow(i, new DenseVector(a))
    }
    dm.from(new IndexedRowMatrix(rdd, g.numRows(), pcs.numCols + 1), blockSize, blockSize)
  }

  def k1(k2: M, k0: M): M = {
    1.0 - (k2 :+ k0)
  }

}

class PCRelate(maf: Double, blockSize: Int) extends Serializable {
  import PCRelate._
  import dm.ops._

  require(maf >= 0.0)
  require(maf <= 1.0)
  val antimaf = (1.0 - maf)
  def badmu(mu: Double): Boolean =
    mu <= maf || mu >= antimaf || mu <= 0.0 || mu >= 1.0
  def badgt(gt: Double): Boolean =
    gt != 0.0 && gt != 1.0 && gt != 2.0

  def apply(vds: VariantDataset, pcs: DenseMatrix, statistics: StatisticSubset = defaultStatisticSubset): Result[M] = {
    vds.requireUniqueSamples("pc_relate")
    val g = vdsToMeanImputedMatrix(vds)

    val mu = this.mu(g, pcs)
    val blockedG = dm.from(g, blockSize, blockSize)

    val variance = dm.map2 { (g, mu) =>
      if (badgt(g) || badmu(mu))
        0.0
      else
        mu * (1.0 - mu)
    }(blockedG, mu)

    val phi = this.phi(mu, variance, blockedG)

    if (statistics >= PhiK2) {
      val k2 = this.k2(phi, mu, variance, blockedG)
      if (statistics >= PhiK2K0) {
        val k0 = this.k0(phi, mu, k2, blockedG, ibs0(blockedG, mu, blockSize))
        if (statistics >= PhiK2K0K1) {
          val k1 = 1.0 - (k2 :+ k0)
          Result(phi, k0, k1, k2)
        } else
          Result(phi, k0, null, k2)
      } else
        Result(phi, null, null, k2)
    } else
      Result(phi, null, null, null)
  }

  /**
    * {@code g} is variant by sample
    * {@code pcs} is sample by numPCs
    *
    **/
  private[methods] def mu(g: IndexedRowMatrix, pcs: DenseMatrix): M = {
    val beta = fitBeta(g, pcs, blockSize)

    val pcsWithIntercept = prependConstantColumn(1.0, pcs)

    ((beta * pcsWithIntercept.transpose) / 2.0)
  }

  private[methods] def phi(mu: M, variance: M, g: M): M = {
    val centeredG = dm.map2 { (g,mu) =>
      if (badgt(g) || badmu(mu))
        0.0
      else
        g - mu * 2.0
    }(g, mu)
    val stddev = dm.map(math.sqrt _)(variance)

    ((centeredG.t * centeredG) :/ (stddev.t * stddev)) / 4.0
  }

  private[methods] def ibs0(g: M, mu: M, blockSize: Int): M = {
    val homalt = dm.map2 { (g, mu) =>
      if (badgt(g) || badmu(mu) || g != 2.0) 0.0 else 1.0
    } (g, mu)
    val homref = dm.map2 { (g, mu) =>
      if (badgt(g) || badmu(mu) || g != 0.0) 0.0 else 1.0
    } (g, mu)
    (homalt.t * homref) :+ (homref.t * homalt)
  }

  private[methods] def k2(phi: M, mu: M, variance: M, g: M): M = {
    val twoPhi_ii = dm.diagonal(phi).map(2.0 * _)
    val normalizedGD = dm.map2WithIndex { (_, i, g, mu) =>
        if (badmu(mu) || badgt(g))
          0.0  // https://github.com/Bioconductor-mirror/GENESIS/blob/release-3.5/R/pcrelate.R#L391
        else {
          val gd = if (g == 0.0) mu
          else if (g == 1.0) 0.0
          else 1.0 - mu

          gd - mu * (1.0 - mu) * twoPhi_ii(i.toInt)
        }
    } (g, mu)

    (normalizedGD.t * normalizedGD) :/ (variance.t * variance)
  }

  private[methods] def k0(phi: M, mu: M, k2: M, g: M, ibs0: M): M = {
    val mu2 = dm.map2 { (g, mu) =>
      if (badgt(g) || badmu(mu))
        0.0
      else
        mu * mu
    }(g, mu)
    val oneMinusMu2 = dm.map2 { (g, mu) =>
      if (badgt(g) || badmu(mu))
        0.0
      else
        (1.0 - mu) * (1.0 - mu)
    }(g, mu)
    val denom = (mu2.t * oneMinusMu2) :+ (oneMinusMu2.t * mu2)
    dm.map4 { (phi: Double, denom: Double, k2: Double, ibs0: Double) =>
      if (phi <= k0cutoff)
        1.0 - 4.0 * phi + k2
      else
        ibs0 / denom
    }(phi, denom, k2, ibs0)
  }

}
