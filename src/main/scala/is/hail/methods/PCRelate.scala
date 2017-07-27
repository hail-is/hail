package is.hail.methods

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, TString, TDouble}
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

  case class Result[M](phiHat: M, k0: M, k1: M, k2: M) {
    def map[N](f: M => N): Result[N] =

    Result(f(phiHat), f(k0), f(k1), f(k2))
  }

  def toPairRdd(vds: VariantDataset, pcs: DenseMatrix, maf: Double, blockSize: Int): RDD[((Annotation, Annotation), (Double, Double, Double, Double))] = {
    val indexToId: Map[Int, Annotation] = vds.sampleIds.zipWithIndex.map { case (id, index) => (index, id) }.toMap
    def upperTriangularEntires(bm: BlockMatrix): RDD[((Annotation, Annotation), Double)] = {
      val rowsPerBlock = bm.rowsPerBlock
      val colsPerBlock = bm.colsPerBlock

      bm.blocks.filter { case ((blocki, blockj), m) =>
        blocki < blockj || {
          val i = blocki * rowsPerBlock
          val j = blockj * colsPerBlock
          val i2 = i + rowsPerBlock
          val j2 = i + colsPerBlock
          i <= j && j < i2 ||
          i < j2 && j2 < i2 ||
          j <= i && i < j2 ||
          j < i2 && i2 < j2
        }
      }.flatMap { case ((blocki, blockj), m) =>
          val ioffset = blocki * rowsPerBlock
          val joffset = blockj * colsPerBlock
          for {
            i <- 0 until m.numRows
            // FIXME: only works with square blocks
            jstart = if (blocki < blockj) 0 else i+1
            j <- jstart until m.numCols
          } yield ((indexToId(i + ioffset), indexToId(j + joffset)), m(i, j))
      }
    }

    val result = maybefast(vds, pcs, maf, blockSize)

    val a = upperTriangularEntires(result.phiHat)
    val b = upperTriangularEntires(result.k0)
    val c = upperTriangularEntires(result.k1)
    val d = upperTriangularEntires(result.k2)

    (a join b join c join d)
        .mapValues { case (((kin, k0), k1), k2) => (kin, k0, k1, k2) }
  }

  private val signature =
    TStruct(("i", TString), ("j", TString), ("kin", TDouble), ("k0", TDouble), ("k1", TDouble), ("k2", TDouble))
  private val keys = Array("i", "j")
  def toKeyTable(vds: VariantDataset, pcs: DenseMatrix, maf: Double, blockSize: Int): KeyTable =
    KeyTable(vds.hc,
      toPairRdd(vds, pcs, maf, blockSize).map { case ((i, j), (kin, k0, k1, k2)) => Annotation(i, j, kin, k0, k1, k2).asInstanceOf[Row] },
      signature,
      keys)

  private val k0cutoff = math.pow(2.0, (-5.0/2.0))

  def maybefast(vds: VariantDataset, pcs: DenseMatrix, maf: Double, blockSize: Int): Result[M] = {
    require(maf >= 0.0)
    require(maf <= 1.0)
    val antimaf = (1.0 - maf)
    def badmu(mu: Double): Boolean =
      mu <= maf || mu >= antimaf || mu <= 0.0 || mu >= 1.0
    def badgt(gt: Double): Boolean =
      gt != 0.0 && gt != 1.0 && gt != 2.0

    val g = vdsToMeanImputedMatrix(vds)

    val beta = fitBeta(g, pcs, blockSize)

    val pcsArray = pcs.toArray
    val pcsWithInterceptArray = new Array[Double](pcs.numRows * (pcs.numCols + 1))

    var i = 0
    while (i < pcs.numRows) {
      pcsWithInterceptArray(i) = 1
      i += 1
    }

    i = 0
    while (i < pcs.numRows * pcs.numCols) {
      pcsWithInterceptArray(pcs.numRows + i) = pcsArray(i)
      i += 1
    }

    val pcsWithIntercept =
      new DenseMatrix(pcs.numRows, pcs.numCols + 1, pcsWithInterceptArray)

    val blockedG = dm.from(g, blockSize, blockSize)
    val mu = ((beta * pcsWithIntercept.transpose) / 2.0)

    val gMinusMu = dm.map2 { (g,mu) =>
      if (badgt(g) || badmu(mu))
        0.0
      else
        g - mu * 2.0
    }(blockedG, mu)

    // println("blockedG spark")
    // println(blockedG.toLocalMatrix())
    // println("mu spark")
    // println(mu.toLocalMatrix())
    // println("gMinusMu spark")
    // println(gMinusMu.toLocalMatrix())

    val variance = dm.map2 {
      case (g, mu) =>
        if (badgt(g) || badmu(mu))
          0.0
        else
          mu * (1.0 - mu)
    }(blockedG, mu)

    // println("stddev")
    // println(variance.sqrt.toLocalMatrix())

    val phi = (((gMinusMu.t * gMinusMu) :/ (variance.t.sqrt * variance.sqrt)) / 4.0)

    val twoPhi_ii = fTwiddle(phi)

    val normalizedGD = dm.map2WithIndex({
      case (_, i, g, mu) =>
        if (badmu(mu) || badgt(g))
          0.0  // https://github.com/Bioconductor-mirror/GENESIS/blob/release-3.5/R/pcrelate.R#L391
        else {
          val gd = if (g == 0.0) mu
          else if (g == 1.0) 0.0
          else 1.0 - mu

          gd - mu * (1.0 - mu) * twoPhi_ii(i.toInt)
        }
    })(blockedG, mu)

    val kTwo = ((normalizedGD.t * normalizedGD) :/ (variance.t * variance))

    val mu2 = dm.map2 {
      case (g, mu) =>
        if (badgt(g) || badmu(mu))
          0.0
        else
          mu * mu
    }(blockedG, mu)

    // FIXME: actually zero this one
    val oneMinusMu2 = dm.map2 {
      case (g, mu) =>
        if (badgt(g) || badmu(mu))
          0.0
        else
          (1.0 - mu) * (1.0 - mu)
    }(blockedG, mu)

    val denom = (mu2.t * oneMinusMu2) :+ (oneMinusMu2.t * mu2)

    val ibsZero = ibs0(vds, blockSize)

    def _k0(phiHat: Double, denom: Double, k2: Double, ibs0: Double) =
      if (phiHat <= k0cutoff)
        1 - 4 * phiHat + k2
      else if (denom == 0)
        0 // denom == 0 ==> mu is bad, https://github.com/Bioconductor-mirror/GENESIS/blob/release-3.5/R/pcrelate.R#L414
      else
        ibs0 / denom
    val kZero = dm.map4(_k0)(phi, denom, kTwo, ibsZero)

    Result(phi, kZero, 1.0 - (kTwo :+ kZero), kTwo)
  }

  // def apply(vds: VariantDataset, pcs: DenseMatrix, blockSize: Int): Result[M] = {
  //   val g = vdsToMeanImputedMatrix(vds)
  //   val blockedG = dm.from(g, blockSize, blockSize)

  //   val beta = fitBeta(g, pcs, blockSize)

  //   val mu = muHat(pcs, beta, blockedG)

  //   val phihat = phiHat(blockedG, mu)

  //   // FIXME: what should I do if the genotype is missing?
  //   val kTwo = k2(f(phihat), dm.map2({ case (g, mu) => if (g == 0.0) mu else if (g == 1.0) 0.0 else if (g == 2.0) 1.0 - mu else g })(blockedG, mu), mu)

  //   val kZero = k0(phihat, mu, kTwo, ibs0(vds, blockSize))

  //   // println(dm.toLocalMatrix(kTwo))

  //   Result(phihat, kZero, k1(kTwo, kZero), kTwo)
  // }

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

  // private val ksi = 1e-10
  // private def clipToInterval(x: Double): Double =
  //   if (x <= 0)
  //     ksi
  //   else if (x >= 1)
  //     1 - ksi
  //   else
  //     x
  // private def clipMu(mu: M): M =
  //   dm.map(clipToInterval _)(mu)

  // private def filter(mu: Double, g: Double): Double =
  //   if (mu <= 0 || mu >= 1)
  //     g / 2.0
  //   else
  //     mu

  // private def filterOutOfRangeMus(mu: M, g: M): M =
  //   dm.map2(filter _)(mu, g)

  // /**
  //   *  pcs: Sample x D
  //   *  beta: SNP x (D+1)
  //   *  g: SNP x Sample
  //   *
  //   *  result: SNP x Sample
  //   */
  // def muHat(pcs: DenseMatrix, beta: M, g: M): M = {
  //   val pcsArray = pcs.toArray
  //   val pcsWithInterceptArray = new Array[Double](pcs.numRows * (pcs.numCols + 1))

  //   var i = 0
  //   while (i < pcs.numRows) {
  //     pcsWithInterceptArray(i) = 1
  //     i += 1
  //   }

  //   i = 0
  //   while (i < pcs.numRows * pcs.numCols) {
  //     pcsWithInterceptArray(pcs.numRows + i) = pcsArray(i)
  //     i += 1
  //   }

  //   val pcsWithIntercept =
  //     new DenseMatrix(pcs.numRows, pcs.numCols + 1, pcsWithInterceptArray)

  //   // println(pcsWithIntercept)
  //   // println(dm.toLocalMatrix(beta))

  //   clipMu((beta * pcsWithIntercept.transpose) / 2.0)
  // }

  // /**
  //   * g: SNP x Sample
  //   * muHat: SNP x Sample
  //   **/
  // def phiHat(g: M, muHat: M): M = {
  //   val gMinusMu = g :- (muHat * 2.0)
  //   val varianceHat = muHat :* (1.0 - muHat)

  //   ((gMinusMu.t * gMinusMu) :/ (varianceHat.t.sqrt * varianceHat.sqrt)) / 4.0
  // }

  def fTwiddle(phiHat: M): Array[Double] =
    dm.diagonal(phiHat).map(2.0 * _)

  // def f(phiHat: M): Array[Double] =
  //   dm.diagonal(phiHat).map(2.0 * _ - 1.0)

  // def k2(f: Array[Double], gD: M, mu: M): M = {
  //   // println("mu")
  //   // println(dm.toLocalMatrix(mu))
  //   // println("gD")
  //   // println(dm.toLocalMatrix(gD))
  //   // println("f.map(1 + _)")
  //   // println(f.map(1 + _): IndexedSeq[Double])
  //   // println("mu :* (1.0 - mu)")
  //   // println(dm.toLocalMatrix(mu :* (1.0 - mu)))
  //   // println("(mu :* (1.0 - mu)) --* f.map(1 + _)")
  //   // println(dm.toLocalMatrix((mu :* (1.0 - mu)) --* (f.map(1 + _))))

  //   val normalizedGD = gD :- ((mu :* (1.0 - mu)) --* (f.map(1 + _)))
  //   val variance = mu :* (1.0 - mu)

  //   // println("numer")
  //   // println(dm.toLocalMatrix(normalizedGD.t * normalizedGD))
  //   // println("denom")
  //   // println(dm.toLocalMatrix(variance.t * variance))

  //   (normalizedGD.t * normalizedGD) :/ (variance.t * variance)
  // }

  // /**
  //   * muHat: SNP x Sample
  //   *
  //   **/
  // private def sqr(x: Double): Double = x * x
  // private val cutoff = math.pow(2.0, (-5.0/2.0))
  // private def _k0(phiHat: Double, denom: Double, k2: Double, ibs0: Double) =
  //   if (phiHat <= cutoff)
  //     1 - 4 * phiHat + k2
  //   else
  //     ibs0 / denom
  // def k0(phiHat: M, mu: M, k2: M, ibs0: M): M = {
  //   val mu2 = dm.map(sqr _)(mu)
  //   val oneMinusMu2 = dm.map(sqr _)(1.0 - mu)

  //   val denom = (mu2.t * oneMinusMu2) :+ (oneMinusMu2.t * mu2)

  //   dm.map4(_k0)(phiHat, denom, k2, ibs0)
  // }

  def k1(k2: M, k0: M): M = {
    1.0 - (k2 :+ k0)
  }

  def ibs0(vds: VariantDataset, blockSize: Int): M =
    dm.from(IBD.ibs(vds)._1, blockSize, blockSize)

}
