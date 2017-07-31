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

    val result = apply(vds, pcs, maf, blockSize)

    val a = upperTriangularEntires(result.phiHat)
    val b = upperTriangularEntires(result.k0)
    val c = upperTriangularEntires(result.k1)
    val d = upperTriangularEntires(result.k2)

    (a join b join c join d)
        .mapValues { case (((kin, k0), k1), k2) => (kin, k0, k1, k2) }
  }

  def apply(vds: VariantDataset, pcs: DenseMatrix, maf: Double, blockSize: Int): Result[M] =
    new PCRelate(maf, blockSize)(vds, pcs)

  private val signature =
    TStruct(("i", TString), ("j", TString), ("kin", TDouble), ("k0", TDouble), ("k1", TDouble), ("k2", TDouble))
  private val keys = Array("i", "j")
  def toKeyTable(vds: VariantDataset, pcs: DenseMatrix, maf: Double, blockSize: Int): KeyTable =
    KeyTable(vds.hc,
      toPairRdd(vds, pcs, maf, blockSize).map { case ((i, j), (kin, k0, k1, k2)) => Annotation(i, j, kin, k0, k1, k2).asInstanceOf[Row] },
      signature,
      keys)

  private val k0cutoff = math.pow(2.0, (-5.0/2.0))

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

  def ibs0(vds: VariantDataset, blockSize: Int): M =
    dm.from(IBD.ibs(vds)._1, blockSize, blockSize)

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

  def apply(vds: VariantDataset, pcs: DenseMatrix): Result[M] = {
    val g = vdsToMeanImputedMatrix(vds)
    val beta = fitBeta(g, pcs, blockSize)

    val pcsWithIntercept = prependConstantColumn(1.0, pcs)

    val blockedG = dm.from(g, blockSize, blockSize)
    val mu = ((beta * pcsWithIntercept.transpose) / 2.0)

    val variance = dm.map2 {
      case (g, mu) =>
        if (badgt(g) || badmu(mu))
          0.0
        else
          mu * (1.0 - mu)
    }(blockedG, mu)

    val phi = this.phi(mu, variance, blockedG)

    // println("blockedG spark")
    // println(blockedG.toLocalMatrix())
    // println("mu spark")
    // println(mu.toLocalMatrix())

    val k2 = this.k2(phi, mu, variance, blockedG)
    val k0 = this.k0(phi, mu, k2, blockedG, ibs0(vds, blockSize))
    val k1 = 1.0 - (k2 :+ k0)

    Result(phi, k0, k1, k2)
  }

  private def phi(mu: M, variance: M, g: M): M = {
    val centeredG = dm.map2 { (g,mu) =>
      if (badgt(g) || badmu(mu))
        0.0
      else
        g - mu * 2.0
    }(g, mu)
    val stddev = dm.map(math.sqrt _)(variance)

    // println("gMinusMu spark")
    // println(centeredG.toLocalMatrix())
    // println("stddev")
    // println(stddev.toLocalMatrix())

    ((centeredG.t * centeredG) :/ (stddev.t * stddev)) / 4.0
  }

  private def k2(phi: M, mu: M, variance: M, g: M): M = {
    val twoPhi_ii = dm.diagonal(phi).map(2.0 * _)
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
    })(g, mu)

    (normalizedGD.t * normalizedGD) :/ (variance.t * variance)
  }

  private def k0(phi: M, mu: M, k2: M, g: M, ibs0: M): M = {
    // println("ibs0 spark")
    // println(ibs0.toLocalMatrix())
    val mu2 = dm.map2 {
      case (g, mu) =>
        if (badgt(g) || badmu(mu))
          0.0
        else
          mu * mu
    }(g, mu)
    val oneMinusMu2 = dm.map2 {
      case (g, mu) =>
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
