package is.hail.methods

import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Paths

import breeze.linalg.{DenseMatrix => BDM, _}
import is.hail.keytable._
import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import org.apache.spark.mllib.linalg.distributed._
import is.hail.SparkSuite
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg._
import org.testng.annotations.Test
import is.hail.check._
import is.hail.check.Prop._
import is.hail.expr.{TDouble, TInt, TString}
import is.hail.variant.VariantDataset
import is.hail.variant.VSMSubgen
import is.hail.stats._
import is.hail.utils.{TextTableReader, _}
import scala.sys.process._
import is.hail.distributedmatrix.DistributedMatrix
import is.hail.distributedmatrix.DistributedMatrix.implicits._

object PCRelateReferenceImplementation {
  def apply(vds: VariantDataset, pcs: DenseMatrix, maf: Double = 0.0): Map[(String, String), (Double, Double, Double, Double)] = {
    val indexToId: Map[Int, String] = vds.stringSampleIds.zipWithIndex.map { case (id, index) => (index, id) }.toMap

    val gts = vds.rdd.map { case (v, (va, gs)) =>
      val a = gs.map(_.gt).toArray
      val mean = a.flatten.sum.toDouble / a.length
      a.map { case Some(v) => v.toDouble ; case None => mean }.toArray
    }.collect().flatten

    val mat = new BDM[Double](vds.nSamples, vds.countVariants().toInt, gts)

    val PCRelate.Result(phi, k0, k1, k2) = forMatrices(mat, new BDM[Double](pcs.numRows, pcs.numCols, pcs.toArray))
      .map(symmetricMatrixToMap(indexToId,_))

    phi.keys.map(k => (k, (phi(k), k0(k), k1(k), k2(k)))).toMap
  }

  // keys.length == mat.rows == mat.cols
  private def symmetricMatrixToMap[T,U](keys: Int => T, mat: BDM[U]): Map[(T, T), U] = (for {
    i <- 0 until mat.rows
    j <- (i+1) until mat.cols
  } yield ((keys(i), keys(j)), mat(i, j))).toMap

  // g : N x M
  // pcs : N x K
  def forMatrices(g: BDM[Double], pcs: BDM[Double], maf: Double = 0.0): PCRelate.Result[BDM[Double]] = {
    val n = g.rows
    val m = g.cols
    require(n == pcs.rows)
    val k = pcs.cols

    // N x (K + 1)
    val pcsWithIntercept = BDM.horzcat(new BDM(n, 1, Array.fill[Double](n)(1.0)), pcs)

    // beta : M x K
    val beta = (inv(pcsWithIntercept.t * pcsWithIntercept) * pcsWithIntercept.t * g).t

    // mu_si : M x N
    val mu_si: BDM[Double] = (beta * pcsWithIntercept.t) / 2.0

    def goodMu(mu: Double): Boolean =
      (mu > 0.0 && mu > maf &&
       mu < 1.0 && mu < (1.0 - maf))
    def goodGT(gt: Double): Boolean =
      gt == 0.0 || gt == 1.0 || gt == 2.0

    // g2mu : M x N
    // the if eliminates contribution from bad mu and bad genotype entries
    val g2mua = new Array[Double](m*n)
    var i = 0
    var j = 0
    while (i < n) {
      while (j < m) {
        val gt = g(i,j)
        val mu = mu_si(j,i)

        if (goodMu(mu) && goodGT(gt))
          g2mua(i*m + j) = gt - (2.0 * mu)

        j += 1
      }
      i += 1
    }

    val g2mu = new BDM[Double](m, n, g2mua)

    val numer = g2mu.t * g2mu

    val variancea = new Array[Double](m*n)
    i = 0
    j = 0
    while (i < n) {
      while (j < m) {
        val mu = mu_si(j,i)

        if (goodMu(mu_si(j,i)))
          variancea(i*m + j) = math.sqrt(mu * (1 - mu))

        j += 1
      }
      i += 1
    }
    val variance = new BDM[Double](m, n, variancea)

    val denom = 4.0 :* (variance.t * variance)
    val phi = numer :/ denom

    def toDom(gt: Double, mu: Double): Double = gt match {
      case 0.0 => mu
      case 1.0 => 0.0
      case 2.0 => 1.0 - mu
      case _   => Double.NaN
    }

    val k2a = new Array[Double](n*n)
    i = 0
    j = 0
    while (i < n) {
      while (j < n) {
        var k = 0
        var numer = 0.0
        var denom = 0.0
        while (k < m) {
          val gt_ki = g(i,k)
          val gt_kj = g(j,k)
          val mu_ki = mu_si(k,i)
          val mu_kj = mu_si(k,j)
          val gD_ki = toDom(gt_ki, mu_ki)
          val gD_kj = toDom(gt_kj, mu_kj)

          if (goodMu(mu_ki) && goodMu(mu_kj) && goodGT(gt_ki) && goodGT(gt_kj)) {
            numer += (gD_ki - mu_ki * (1.0 - mu_ki) * 2.0 * phi(i,i)) *
            (gD_kj - mu_kj * (1.0 - mu_kj) * 2.0 * phi(j,j))

            denom += mu_ki * (1.0 - mu_ki) * mu_kj * (1.0 - mu_kj)
          }

          k += 1
        }

        k2a(j*n + i) = numer / denom

        j += 1
      }
      i += 1
    }
    val k2 = new BDM[Double](n, n, k2a)

    val k0a = new Array[Double](n*n)
    i = 0
    j = 0
    while (i < n) {
      while (j < n) {
        if (phi(i,j) >= 0.177) {
          var k = 0
          var numer = 0.0
          var denom = 0.0
          while (k < m) {
            val g_ki = g(i,k)
            val g_kj = g(j,k)
            val mu_ki = mu_si(k,i)
            val mu_kj = mu_si(k,j)

            if (g_ki == g_kj)
              numer += 1.0

            denom += mu_ki*mu_ki*(1-mu_ki)*(1-mu_ki) + mu_kj*mu_kj*(1-mu_kj)*(1-mu_kj)

            k += 1
          }

          k0a(j*n + i) = numer / denom
        } else {
          k0a(j*n + i) = 1 - 4 * phi(i,j) + k2(i,j)
        }

        j += 1
      }
      i += 1
    }

    val k0 = new BDM[Double](n, n, k0a)

    val k1 = 1.0 - (k0 :+ k2)

    PCRelate.Result(phi, k0, k1, k2)
  }
}

class PCRelateSuite extends SparkSuite {
  private val blockSize: Int = 8192

  private def toI(a: Any): Int =
    a.asInstanceOf[Int]

  private def toD(a: Any): Double =
    a.asInstanceOf[Double]

  private def toS(a: Any): String =
    a.asInstanceOf[String]

  def runPcRelateHail(vds: VariantDataset, pcs: DenseMatrix, maf: Double): Map[(String, String), (Double, Double, Double, Double)] =
    PCRelate.toPairRdd(vds, pcs, maf, blockSize).collect().toMap.asInstanceOf[Map[(String, String), (Double, Double, Double, Double)]]

  def runPcRelateR(
    vds: VariantDataset,
    maf: Double,
    rFile: String = "src/test/resources/is/hail/methods/runPcRelate.R"): Map[(String, String), (Double, Double, Double, Double)] = {

    val tmpfile = tmpDir.createTempFile(prefix = "pcrelate")
    val localTmpfile = tmpDir.createLocalTempFile(prefix = "pcrelate")
    val pcRelateScript = tmpDir.createLocalTempFile(prefix = "pcrelateScript")

    vds.exportPlink(tmpfile)

    for (suffix <- Seq(".bed", ".bim", ".fam")) {
      hadoopConf.copy(tmpfile + suffix, localTmpfile + suffix)
    }

    s"Rscript $rFile ${uriPath(localTmpfile)} ${maf}" !

    val columns = Map(
      ("ID1", TString),
      ("ID2", TString),
      ("nsnp", TDouble),
      ("kin", TDouble),
      ("k0", TDouble),
      ("k1", TDouble),
      ("k2", TDouble))
    val separator = " +"

    hadoopConf.copy(localTmpfile + ".out", tmpfile + ".out")

    val (_, rdd) = TextTableReader.read(sc)(Array(tmpfile + ".out"), columns, separator=separator)
    rdd.collect()
      .map(_.value)
      .map { ann =>
      val row = ann.asInstanceOf[Row]
      val id1 = toS(row(0))
      val id2 = toS(row(1))
      val nsnp = toD(row(2)).toInt
      val kin = toD(row(3))
      val k0 = toD(row(4))
      val k1 = toD(row(5))
      val k2 = toD(row(6))
      ((id1, id2), (kin, k0, k1, k2))
    }
      .toMap
  }

  def compareDoubleQuartuplets(cmp: (Double, Double) => Boolean)(x: (Double, Double, Double, Double), y: (Double, Double, Double, Double)): Boolean =
    cmp(x._1, y._1) && cmp(x._2, y._2) && cmp(x._3, y._3) && cmp(x._4, y._4)

  @Test
  def trivialReference() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromGtMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new DenseMatrix(4,2,pcsArray)
    val us = runPcRelateHail(vds, pcs, maf=0.0)
    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.0)
    assert(mapSameElements(us, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
  }

  @Test
  def trivialReferenceMatchesR() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromGtMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new DenseMatrix(4,2,pcsArray)
    val usRef = PCRelateReferenceImplementation(vds, pcs, maf=0.0)
    val truth = runPcRelateR(vds, maf=0.0, "src/test/resources/is/hail/methods/runPcRelateOnTrivialExample.R")
    assert(mapSameElements(usRef, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
  }

  @Test def baldingNicholsMatchesReference() {
    for {
      n <- Seq(50//, 100, 500
      )
      seed <- Seq(0//, 1, 2
      )
      nVariants <- Seq(1000// , 10000, 50000
      )
    } {
      val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
      val pcs = SamplePCA.justScores(vds, 2)
      val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)
      val actual = runPcRelateHail(vds, pcs, maf=0.01)

      printToFile(new java.io.File(s"/tmp/baldingNicholsMatchesReference-$n-$seed-$nVariants.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- actual) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      println(s"$n $seed $nVariants")
      assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => D_==(x, y, tolerance=1e-2))))
    }
  }

  @Test def baldingNicholsReferenceMatchesR() {
    for {
      n <- Seq(50//, 100, 500
      )
      seed <- Seq(0//, 1, 2
      )
      nVariants <- Seq(1000// , 10000, 50000
      )
    } {
      val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
      val pcs = SamplePCA.justScores(vds, 2)
      val truth = runPcRelateR(vds, maf=0.01)
      val actual = PCRelateReferenceImplementation(vds, pcs, maf=0.01)

      printToFile(new java.io.File(s"/tmp/baldingNicholsMatchesReference-$n-$seed-$nVariants.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- actual) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      println(s"$n $seed $nVariants")
      assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => D_==(x, y, tolerance=1e-2))))
    }
  }

}
