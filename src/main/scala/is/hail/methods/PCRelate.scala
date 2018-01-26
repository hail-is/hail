package is.hail.methods

import breeze.linalg.{*, DenseMatrix}
import is.hail.annotations.{Annotation, UnsafeRow}
import is.hail.distributedmatrix.BlockMatrix
import is.hail.distributedmatrix.BlockMatrix.ops._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.HailContext
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

  def apply(hc: HailContext,
    blockedG: M,
    columnKeys: Array[Annotation],
    columnKeyType: Type,
    k: Int,
    pcaScores: Table,
    maf: Double,
    blockSize: Int,
    minKinship: Double,
    statistics: PCRelate.StatisticSubset): Table = {

    if (blockedG.nCols > Integer.MAX_VALUE) {
      fatal(s"PC Relate is incompatible with variant sample matrices containing more than ${Integer.MAX_VALUE} samples")
    }

    val pcs = tableToBDM(pcaScores, blockedG.nCols.toInt, k)
    val result = new PCRelate(maf, blockSize)(blockedG, pcs, statistics)

    PCRelate.toTable(hc, result, columnKeys, columnKeyType, blockSize, minKinship, statistics)
  }

  def apply(vds: MatrixTable,
    k: Int,
    pcaScores: Table,
    maf: Double,
    blockSize: Int): Table =
    apply(vds, k, pcaScores, maf, blockSize, defaultMinKinship, defaultStatisticSubset)

  def apply(vds: MatrixTable,
    k: Int,
    pcaScores: Table,
    maf: Double,
    blockSize: Int,
    statisticSubset: StatisticSubset): Table =
    apply(vds, k, pcaScores, maf, blockSize, defaultMinKinship, statisticSubset)

  def apply(vds: MatrixTable,
    k: Int,
    pcaScores: Table,
    maf: Double,
    blockSize: Int,
    minKinship: Double): Table =
    apply(vds, k, pcaScores, maf, blockSize, minKinship, defaultStatisticSubset)

  def apply(vds: MatrixTable,
    k: Int,
    pcaScores: Table,
    maf: Double,
    blockSize: Int,
    minKinship: Double,
    statistics: PCRelate.StatisticSubset): Table = {

    vds.requireUniqueSamples("pc_relate")
    val g = vdsToMeanImputedMatrix(vds)
    val blockedG = BlockMatrix.from(g, blockSize).cache()

    apply(vds.hc, blockedG, vds.sampleIds.toArray, vds.sSignature, k, pcaScores, maf, blockSize, minKinship, statistics)
  }

  private def tableToBDM(t: Table, nRows: Int, nCols: Int): DenseMatrix[Double] = {
    val scoreArray = new Array[Double](nRows * nCols)
    val pcs = t.collect().asInstanceOf[IndexedSeq[UnsafeRow]]
    var i = 0
    while (i < nRows) {
      val row = pcs(i).getAs[IndexedSeq[Double]](1)
      var j = 0
      while (j < nCols) {
        scoreArray(j * nRows + i) = row(j)
        j += 1
      }
      i += 1
    }
    new DenseMatrix[Double](nRows, nCols, scoreArray)
  }

  private def signature(columnKeyType: Type) =
    TStruct(("i", columnKeyType), ("j", columnKeyType), ("kin", TFloat64()), ("k0", TFloat64()), ("k1", TFloat64()), ("k2", TFloat64()))
  private val keys = Array("i", "j")

  def toTable(hc: HailContext, r: Result[M], columnKeys: Array[Annotation], columnKeyType: Type, blockSize: Int, minKinship: Double = defaultMinKinship, statistics: StatisticSubset = defaultStatisticSubset): Table =
    Table(hc, toRowRdd(r, columnKeys, blockSize, minKinship, statistics), signature(columnKeyType), keys)

  private def toRowRdd(r: Result[M], columnKeys: Array[Annotation], blockSize: Int, minKinship: Double, statistics: StatisticSubset): RDD[Row] = {
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
              ab += Annotation(columnKeys(iOffset + ii), columnKeys(jOffset + jj), kin, k0, k1, k2).asInstanceOf[Row]
            }
            ii += 1
          }
          jj += 1
        }
        ab.result()
      } else
        new Array[Row](0)
    }

    val Result(phi, k0, k1, k2) = r

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

  private val k0cutoff = math.pow(2.0, -5.0 / 2.0)

  def vdsToMeanImputedMatrix(vds: MatrixTable): IndexedRowMatrix = {
    val nSamples = vds.nSamples
    val localRowType = vds.rvRowType
    val partStarts = vds.partitionStarts()
    val partStartsBc = vds.sparkContext.broadcast(partStarts)
    val rdd = vds.rdd2.mapPartitionsWithIndex { case (partIdx, it) =>
      val view = HardCallView(localRowType)
      val missingIndices = new ArrayBuilder[Int]()

      var rowIdx = partStartsBc.value(partIdx)
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

        rowIdx += 1
        IndexedRow(rowIdx - 1, Vectors.dense(a))
      }
    }

    new IndexedRowMatrix(rdd.cache(), partStarts.last, nSamples)
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

  def apply(uncachedBlockedG: M, pcs: DenseMatrix[Double], statistics: StatisticSubset = defaultStatisticSubset): Result[M] = {
    val blockedG = uncachedBlockedG.cache()
    val mu = this.mu(blockedG, pcs)
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
  private[methods] def mu(blockedG: M, pcs: DenseMatrix[Double]): M = {
    import breeze.linalg._

    val pcsWithIntercept = DenseMatrix.horzcat(DenseMatrix.ones[Double](pcs.rows, 1), pcs)

    val qr.QR(q, r) = qr.reduced(pcsWithIntercept)

    val beta = blockedG.leftMultiply(q.t).leftMultiply(inv(r))

    (beta.leftMultiply(pcsWithIntercept) / 2.0).t
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
