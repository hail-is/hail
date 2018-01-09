package is.hail.methods

import breeze.linalg.{*, DenseMatrix}
import is.hail.annotations.{Annotation, UnsafeRow}
import is.hail.distributedmatrix.BlockMatrix
import is.hail.distributedmatrix.BlockMatrix.ops._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant.{HardCallView, MatrixTable, Variant}
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.language.{higherKinds, implicitConversions}

object PCRelate {
  type M = BlockMatrix

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

  def apply(vds: MatrixTable, k: Int, pcaScores: Table, maf: Double, blockSize: Int, minKinship: Double = PCRelate.defaultMinKinship, statistics: PCRelate.StatisticSubset = PCRelate.defaultStatisticSubset): Table = {
    val scoreArray = new Array[Double](vds.nSamples * k)
    val pcs = pcaScores.collect().asInstanceOf[IndexedSeq[UnsafeRow]]
    var i = 0
    while (i < vds.nSamples) {
      val row = pcs(i).getAs[IndexedSeq[Double]](1)
      var j = 0
      while (j < k) {
        scoreArray(j * vds.nSamples + i) = row(j)
        j += 1
      }
      i += 1
    }
    PCRelate.toTable(vds, new DenseMatrix[Double](vds.nSamples, k, scoreArray), maf, blockSize, minKinship, statistics)
  }

  private val signature =
    TStruct(("i", TString()), ("j", TString()), ("kin", TFloat64()), ("k0", TFloat64()), ("k1", TFloat64()), ("k2", TFloat64()))
  private val keys = Array("i", "j")

  private def toRowRdd(vds: MatrixTable, pcs: DenseMatrix[Double], maf: Double, blockSize: Int, minKinship: Double, statistics: StatisticSubset): RDD[Row] = {
    val localSampleIds = vds.sampleIds
    val Result(phi, k0, k1, k2) = new PCRelate(maf, blockSize)(vds, pcs, statistics)

    def fuseBlocks(i: Int, j: Int, lmPhi: DenseMatrix[Double], lmK0: DenseMatrix[Double], lmK1: DenseMatrix[Double], lmK2: DenseMatrix[Double]) = {
      val iOffset = i * blockSize
      val jOffset = j * blockSize

      if (i <= j) {
        val size = lmPhi.rows * lmPhi.cols
        val ab = new ArrayBuilder[Row]()
        var jj = 1
        while (jj < lmPhi.cols) {
          var ii = 0
          // assumes square blocks
          val rowsAboveDiagonal = if (i < j) lmPhi.rows else jj
          while (ii < rowsAboveDiagonal) {
            val kin = lmPhi(ii, jj)
            if (kin >= minKinship) {
              val k0 = if (lmK0 == null) null else lmK0(ii, jj)
              val k1 = if (lmK1 == null) null else lmK1(ii, jj)
              val k2 = if (lmK2 == null) null else lmK2(ii, jj)
              ab += Annotation(localSampleIds(iOffset + ii), localSampleIds(jOffset + jj), kin, k0, k1, k2).asInstanceOf[Row]
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

  def toTable(vds: MatrixTable, pcs: DenseMatrix[Double], maf: Double, blockSize: Int, minKinship: Double = defaultMinKinship, statistics: StatisticSubset = defaultStatisticSubset): Table =
    Table(vds.hc, toRowRdd(vds, pcs, maf, blockSize, minKinship, statistics), signature, keys)

  private val k0cutoff = math.pow(2.0, -5.0 / 2.0)

  private def prependConstantColumn(k: Double, m: DenseMatrix[Double]): DenseMatrix[Double] = {    
    val result = new Array[Double](m.rows * (m.cols + 1))
    var i = 0
    while (i < m.rows) {
      result(i) = 1
      i += 1
    }
    System.arraycopy(m.toArray, 0, result, m.rows, m.rows * m.cols)
    
    new DenseMatrix(m.rows, m.cols + 1, result)
  }

  def vdsToMeanImputedMatrix(vds: MatrixTable): IndexedRowMatrix = {
    val nSamples = vds.nSamples
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)
    val localRowType = vds.rowType
    val rdd = vds.rdd2.mapPartitions { it =>
      val view = HardCallView(localRowType)
      val missingIndices = new ArrayBuilder[Int]()

      it.map { rv =>
        val v = Variant.fromRegionValue(rv.region,
          localRowType.loadField(rv, 1))
        view.setRegion(rv)

        missingIndices.clear()
        var sum = 0
        var nNonMissing = 0
        val a = new Array[Double](nSamples)
        var i = 0
        while (i < nSamples) {
          view.setGenotype(i)
          if (view.hasGT) {
            val gt = view.getGT
            sum += gt
            a(i) = gt
            nNonMissing += 1
          } else
            missingIndices += i
          i += 1
        }

        val mean = sum.toDouble / nNonMissing

        i = 0
        while (i < missingIndices.length) {
          a(missingIndices(i)) = mean
          i += 1
        }

        IndexedRow(variantIdxBc.value(v), Vectors.dense(a))
      }
    }
    new IndexedRowMatrix(rdd.cache(), variants.length, nSamples)
  }

  /**
    * g: SNP x Sample
    * pcs: Sample x D
    *
    * result: (SNP x (D+1))
    */
  def fitBeta(g: IndexedRowMatrix, pcs: DenseMatrix[Double], blockSize: Int): M = {
    val aa = g.rows.sparkContext.broadcast(pcs(*, ::).map(_.toArray).toArray)
    val rdd = g.rows.map { case IndexedRow(i, v) =>
      val ols = new OLSMultipleLinearRegression()
      ols.newSampleData(v.toArray, aa.value)
      val a = ols.estimateRegressionParameters()
      IndexedRow(i, Vectors.dense(a))
    }
    BlockMatrix.from(new IndexedRowMatrix(rdd, g.numRows(), pcs.cols + 1), blockSize)
  }

  def k1(k2: M, k0: M): M = {
    1.0 - (k2 :+ k0)
  }

}

class PCRelate(maf: Double, blockSize: Int) extends Serializable {
  import PCRelate._

  require(maf >= 0.0)
  require(maf <= 1.0)

  def badmu(mu: Double): Boolean =
    mu <= maf || mu >= (1.0 - maf) || mu <= 0.0 || mu >= 1.0

  def badgt(gt: Double): Boolean =
    gt != 0.0 && gt != 1.0 && gt != 2.0

  private def gram(m: M): M = {
    val mc = m.cache()
    mc.t * mc
  }

  def apply(vds: MatrixTable, pcs: DenseMatrix[Double], statistics: StatisticSubset = defaultStatisticSubset): Result[M] = {
    vds.requireUniqueSamples("pc_relate")
    val g = vdsToMeanImputedMatrix(vds)

    val mu = this.mu(g, pcs).cache()
    val blockedG = BlockMatrix.from(g, blockSize).cache()

    val variance =
      BlockMatrix.map2 { (g, mu) =>
        if (badgt(g) || badmu(mu))
          0.0
        else
          mu * (1.0 - mu)
      } (blockedG, mu).cache()

    val phi = this.phi(mu, variance, blockedG).cache()

    if (statistics >= PhiK2) {
      val k2 = this.k2(phi, mu, variance, blockedG).cache()
      if (statistics >= PhiK2K0) {
        val k0 = this.k0(phi, mu, k2, blockedG, ibs0(blockedG, mu, blockSize)).cache()
        if (statistics >= PhiK2K0K1) {
          val k1 = (1.0 - (k2 :+ k0)).cache()
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
  private[methods] def mu(g: IndexedRowMatrix, pcs: DenseMatrix[Double]): M = {
    val beta = fitBeta(g, pcs, blockSize)

    val pcsWithIntercept = prependConstantColumn(1.0, pcs)

    (beta * pcsWithIntercept.t) / 2.0
  }

  private[methods] def phi(mu: M, variance: M, g: M): M = {
    val centeredG = BlockMatrix.map2 { (g, mu) =>
      if (badgt(g) || badmu(mu))
        0.0
      else
        g - mu * 2.0
    } (g, mu)
    
    val stddev = variance.map(math.sqrt)

    (gram(centeredG) :/ gram(stddev)) / 4.0
  }

  private[methods] def ibs0(g: M, mu: M, blockSize: Int): M = {
    val homalt =
      BlockMatrix.map2 { (g, mu) =>
        if (badgt(g) || badmu(mu) || g != 2.0) 0.0 else 1.0
      } (g, mu).cache()
    
    val homref =
      BlockMatrix.map2 { (g, mu) =>
        if (badgt(g) || badmu(mu) || g != 0.0) 0.0 else 1.0
      } (g, mu).cache()
    
    (homalt.t * homref) :+ (homref.t * homalt)
  }

  private[methods] def k2(phi: M, mu: M, variance: M, g: M): M = {
    val twoPhi_ii = phi.diagonal.map(2.0 * _)
    val normalizedGD = g.map2WithIndex(mu, { case (_, i, g, mu) =>
      if (badmu(mu) || badgt(g))
        0.0 // https://github.com/Bioconductor-mirror/GENESIS/blob/release-3.5/R/pcrelate.R#L391
      else {
        val gd = if (g == 0.0) mu
        else if (g == 1.0) 0.0
        else 1.0 - mu

        gd - mu * (1.0 - mu) * twoPhi_ii(i.toInt)
      }
    })

    gram(normalizedGD) :/ gram(variance)
  }

  private[methods] def k0(phi: M, mu: M, k2: M, g: M, ibs0: M): M = {
    val mu2 =
      BlockMatrix.map2 { (g, mu) =>
        if (badgt(g) || badmu(mu))
          0.0
        else
          mu * mu
      } (g, mu).cache()
    
    val oneMinusMu2 =
      BlockMatrix.map2 { (g, mu) =>
        if (badgt(g) || badmu(mu))
          0.0
        else
          (1.0 - mu) * (1.0 - mu)
      } (g, mu).cache()
    
    val denom = (mu2.t * oneMinusMu2) :+ (oneMinusMu2.t * mu2)
    BlockMatrix.map4 { (phi: Double, denom: Double, k2: Double, ibs0: Double) =>
      if (phi <= k0cutoff)
        1.0 - 4.0 * phi + k2
      else
        ibs0 / denom
    } (phi, denom, k2, ibs0)
  }
}
