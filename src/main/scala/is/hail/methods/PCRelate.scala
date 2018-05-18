package is.hail.methods

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.annotations.Annotation
import is.hail.linalg.BlockMatrix
import is.hail.linalg.BlockMatrix.ops._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant.{Call, HardCallView, MatrixTable}
import is.hail.HailContext
import org.apache.spark.storage.StorageLevel
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

  val defaultMinKinship: Double = Double.NegativeInfinity
  val defaultStatisticSubset: StatisticSubset = PhiK2K0K1
  val defaultStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  private val sig = TStruct(
      ("i", TInt32()),
      ("j", TInt32()),
      ("kin", TFloat64()),
      ("ibd0", TFloat64()),
      ("ibd1", TFloat64()),
      ("ibd2", TFloat64()))

  private val keys: IndexedSeq[String] = Array("i", "j")

  def apply(
    hc: HailContext,
    blockedG: M,
    scoresTable: Table,
    maf: Double,
    blockSize: Int,
    minKinship: Double,
    statistics: PCRelate.StatisticSubset): Table = {
    
    val scoresLocal = scoresTable.collect()
    assert(scoresLocal.length == blockedG.nCols)
    
    val pcs = rowsToBDM(scoresLocal.map(_.getAs[IndexedSeq[java.lang.Double]](0))) // non-missing in Python
    
    val result = new PCRelate(maf, blockSize, statistics, defaultStorageLevel)(hc, blockedG, pcs)

    Table(hc, toRowRdd(result, blockSize, minKinship, statistics), sig, Some(keys))
  }

  private[methods] def apply(hc: HailContext,
    vds: MatrixTable,
    pcs: BDM[Double],
    maf: Double,
    blockSize: Int,
    minKinship: Double,
    statistics: PCRelate.StatisticSubset): Table = {

    val g = vdsToMeanImputedMatrix(vds)
    val blockedG = BlockMatrix.fromIRM(g, blockSize).cache()
    val sampleIds = vds.stringSampleIds.toArray[Annotation]
    val idType = TString()

    val result = new PCRelate(maf, blockSize, statistics, defaultStorageLevel)(hc, blockedG, pcs)

    Table(vds.hc, toRowRdd(result, blockSize, minKinship, statistics), sig, Some(keys))
      .annotateGlobal(sampleIds.toFastIndexedSeq, TArray(TString()), "sample_ids")
      .select("{i: global.sample_ids[row.i], j: global.sample_ids[row.j], kin: row.kin, ibd0: row.ibd0, ibd1: row.ibd1, ibd2: row.ibd2}",
        Some(keys.toFastIndexedSeq), Some(0))
  }

  // FIXME move matrix formation to Python
  private def rowsToBDM(x: Array[IndexedSeq[java.lang.Double]]): BDM[Double] = {
    val nRows = x.length
    assert(nRows > 0)
    val nCols = x(0).length
    val a = new Array[Double](nRows * nCols)
    var i = 0
    while (i < nRows) {
      val row = x(i)
      if (row.length != nCols)
        fatal(s"pc_relate: column index 0 has $nCols scores but column index $i has ${row.length} scores.")
      var j = 0
      while (j < nCols) {
        val e = row(j)
        if (e == null)
          fatal(s"pc_relate: found missing score at array index $j for column index $i")
        a(j * nRows + i) = e
        j += 1
      }
      i += 1
    }
    new BDM(nRows, nCols, a)
  }

  private def toRowRdd(
    r: Result[M],
    blockSize: Int,
    minKinship: Double,
    statistics: StatisticSubset): RDD[Row] = {
    
    def fuseBlocks(i: Int, j: Int,
      lmPhi: BDM[Double],
      lmK0: BDM[Double],
      lmK1: BDM[Double],
      lmK2: BDM[Double]) = {
            
      if (i <= j) {
        val iOffset = i * blockSize
        val jOffset = j * blockSize

        val pairs = new ArrayBuilder[Row]()
        var jj = 0
        while (jj < lmPhi.cols) {
          var ii = 0
          val nRowsAboveDiagonal = if (i < j) lmPhi.rows else jj // assumes square blocks on diagonal
          while (ii < nRowsAboveDiagonal) {
            val kin = lmPhi(ii, jj)
            if (kin >= minKinship) {
              val k0 = if (lmK0 == null) null else lmK0(ii, jj)
              val k1 = if (lmK1 == null) null else lmK1(ii, jj)
              val k2 = if (lmK2 == null) null else lmK2(ii, jj)
              pairs += Row(iOffset + ii, jOffset + jj, kin, k0, k1, k2)
            }
            ii += 1
          }
          jj += 1
        }
        pairs.result()
      } else
        Array.empty[Row]
    }

    val Result(phi, k0, k1, k2) = r

    // FIXME replace join with zipPartitions, throw away lower triangular blocks first, avoid the nulls
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

  // now only used in tests
  private[methods] def vdsToMeanImputedMatrix(vds: MatrixTable): IndexedRowMatrix = {
    val nSamples = vds.numCols
    val localRowType = vds.rvRowType
    val partStarts = vds.partitionStarts()
    val partStartsBc = vds.sparkContext.broadcast(partStarts)
    val rdd = vds.rvd.mapPartitionsWithIndex { (partIdx, it) =>
      val view = HardCallView(localRowType)
      val missingIndices = new ArrayBuilder[Int]()

      var rowIdx = partStartsBc.value(partIdx)
      it.map { rv =>
        view.setRegion(rv)

        missingIndices.clear()
        var sum = 0
        var nNonMissing = 0
        val a = new Array[Double](nSamples)
        var i = 0
        while (i < nSamples) {
          view.setGenotype(i)
          if (view.hasGT) {
            val gt = Call.unphasedDiploidGtIndex(view.getGT)
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
    1.0 - (k2 + k0)
  }
}

class PCRelate(maf: Double, blockSize: Int, statistics: PCRelate.StatisticSubset, storageLevel: StorageLevel) extends Serializable {
  import PCRelate._

  require(maf >= 0.0)
  require(maf <= 1.0)

  def badmu(mu: Double): Boolean =
    mu <= maf || mu >= (1.0 - maf) || mu <= 0.0 || mu >= 1.0

  def badgt(gt: Double): Boolean =
    gt != 0.0 && gt != 1.0 && gt != 2.0

  private def gram(m: M): M = {
    val mc = m.persist(storageLevel)
    mc.T.dot(mc)
  }

  private[this] def cacheWhen(statisticsLevel: StatisticSubset)(m: M): M =
    if (statistics >= statisticsLevel) m.persist(storageLevel) else m

  def apply(hc: HailContext, blockedG: M, pcs: BDM[Double]): Result[M] = {
    val preMu = this.mu(blockedG, pcs)
    val mu = BlockMatrix.map2 { (g, mu) =>
      if (badgt(g) || badmu(mu))
        Double.NaN
      else
        mu
   } (blockedG, preMu).persist(storageLevel)
    val variance = cacheWhen(PhiK2)(
      mu.map(mu => if (mu.isNaN) 0.0 else mu * (1.0 - mu)))

    // write phi to cache and increase parallelism of multiplies before phi.diagonal()
    val phiFile = hc.getTemporaryFile(suffix=Some("bm"))
    this.phi(mu, variance, blockedG).write(phiFile)
    val phi = BlockMatrix.read(hc, phiFile)

    if (statistics >= PhiK2) {
      val k2 = cacheWhen(PhiK2K0)(
        this.k2(phi, mu, variance, blockedG))
      if (statistics >= PhiK2K0) {
        val k0 = cacheWhen(PhiK2K0K1)(
          this.k0(phi, mu, k2, blockedG, ibs0(blockedG, mu, blockSize)))
        if (statistics >= PhiK2K0K1) {
          val k1 = 1.0 - (k2 + k0)
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
  private[methods] def mu(blockedG: M, pcs: BDM[Double]): M = {
    import breeze.linalg._

    val pcsWithIntercept = BDM.horzcat(BDM.ones[Double](pcs.rows, 1), pcs)

    val qr.QR(q, r) = qr.reduced(pcsWithIntercept)

    val halfBeta = (inv(2.0 * r) * q.t).matrixMultiply(blockedG.T)

    pcsWithIntercept.matrixMultiply(halfBeta).T
  }

  private[methods] def phi(mu: M, variance: M, g: M): M = {
    val centeredAF = BlockMatrix.map2 { (g, mu) =>
      if (mu.isNaN) 0.0 else g / 2 - mu
    } (g, mu)

    val stddev = variance.sqrt()

    gram(centeredAF) / gram(stddev)
  }

  private[methods] def ibs0(g: M, mu: M, blockSize: Int): M = {
    val homalt =
      BlockMatrix.map2 { (g, mu) =>
        if (mu.isNaN || g != 2.0) 0.0 else 1.0
      } (g, mu)

    val homref =
      BlockMatrix.map2 { (g, mu) =>
        if (mu.isNaN || g != 0.0) 0.0 else 1.0
      } (g, mu)

    val temp = homalt.T.dot(homref).persist(storageLevel)

    temp + temp.T
  }

  private[methods] def k2(phi: M, mu: M, variance: M, g: M): M = {
    val twoPhi_ii = phi.diagonal().map(2.0 * _)
    val normalizedGD = g.map2WithIndex(mu, { case (_, i, g, mu) =>
      if (mu.isNaN)
        0.0 // https://github.com/Bioconductor-mirror/GENESIS/blob/release-3.5/R/pcrelate.R#L391
      else {
        val gd = if (g == 0.0) mu
        else if (g == 1.0) 0.0
        else 1.0 - mu

        gd - mu * (1.0 - mu) * twoPhi_ii(i.toInt)
      }
    })

    gram(normalizedGD) / gram(variance)
  }

  private[methods] def k0(phi: M, mu: M, k2: M, g: M, ibs0: M): M = {
    val mu2 =
      mu.map(mu => if (mu.isNaN) 0.0 else mu * mu)

    val oneMinusMu2 =
      mu.map(mu => if (mu.isNaN) 0.0 else (1.0 - mu) * (1.0 - mu))

    val temp = mu2.T.dot(oneMinusMu2)
    val denom = temp + temp.T

    BlockMatrix.map4 { (phi: Double, denom: Double, k2: Double, ibs0: Double) =>
      if (phi <= k0cutoff)
        1.0 - 4.0 * phi + k2
      else
        ibs0 / denom
    } (phi, denom, k2, ibs0)
  }
}
