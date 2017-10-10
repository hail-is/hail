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
import is.hail.expr.{TDouble, TString}
import is.hail.variant.VariantDataset
import is.hail.variant.VSMSubgen
import is.hail.stats._
import is.hail.utils.{TextTableReader, _}

import scala.sys.process._
import is.hail.distributedmatrix.BlockMatrix
import is.hail.distributedmatrix.BlockMatrix.ops._

class PCRelateSuite extends SparkSuite {
  private val blockSize: Int = 8192

  private def toI(a: Any): Int =
    a.asInstanceOf[Int]

  private def toD(a: Any): Double =
    a.asInstanceOf[Double]

  private def toS(a: Any): String =
    a.asInstanceOf[String]

  def runPcRelateHail(vds: VariantDataset, pcs: DenseMatrix, maf: Double, minKinship: Double = PCRelate.defaultMinKinship): Map[(String, String), (Double, Double, Double, Double)] =
    PCRelate.toPairRdd(vds, pcs, maf, blockSize, minKinship).collect().toMap.asInstanceOf[Map[(String, String), (Double, Double, Double, Double)]]

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
    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.0)._1
    assert(mapSameElements(us, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test(enabled = false)
  def trivialReferenceMatchesR() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromGtMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new DenseMatrix(4,2,pcsArray)
    val usRef = PCRelateReferenceImplementation(vds, pcs, maf=0.0)._1
    val truth = runPcRelateR(vds, maf=0.0, "src/test/resources/is/hail/methods/runPcRelateOnTrivialExample.R")
    assert(mapSameElements(usRef, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test
  def baldingNicholsMatchesReference() {
    for {
      n <- Seq(50, 100)
      seed <- Seq(0, 1)
      nVariants <- Seq(1000, 10000)
    } {
      val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
      val pcs = SamplePCA.justScores(vds, 2)
      val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1
      val actual = runPcRelateHail(vds, pcs, maf=0.01)

      assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-14)))
    }
  }

  @Test(enabled = false)
  def baldingNicholsReferenceMatchesR() {
    for {
      n <- Seq(50, 100)
      seed <- Seq(0, 1)
      nVariants <- Seq(1000, 10000)
    } {
      val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
      val pcs = SamplePCA.justScores(vds, 2)
      val truth = runPcRelateR(vds, maf=0.01)
      val actual = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1

      assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => D_==(x, y, tolerance=1e-2))))
    }
  }

  private def compareBDMs(l: BDM[Double], r: BDM[Double], tolerance: Double) {
    val fails = l.data.zip(r.data).zipWithIndex.flatMap { case ((actual, truth), idx) =>
      val row = idx % l.rows
      val col = idx / l.rows
      if (math.abs(actual - truth) >= tolerance)
        Some(((row, col), actual - truth, actual, truth))
      else
        None
    }
    if (!fails.isEmpty)
      fails.foreach(println _)
    assert(fails.isEmpty)
  }

  @Test
  def sampleVcfMatchesReference() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val pcs = SamplePCA.justScores(vds.coalesce(10), 2)

    val (truth, truth_g, truth_ibs0, truth_mu) = PCRelateReferenceImplementation(vds, pcs, maf=0.01)

    val pcr = new PCRelate(0.01, blockSize)
    val g = PCRelate.vdsToMeanImputedMatrix(vds)
    val dmu = pcr.mu(g, pcs)
    // blockedG : variant x sample
    val blockedG = BlockMatrix.from(g, blockSize)
    val actual = runPcRelateHail(vds, pcs, 0.01)
    val actual_g = blockedG.t.toLocalMatrix()
    val actual_ibs0 = pcr.ibs0(blockedG, dmu, blockSize).toLocalMatrix()
    val actual_mean = dmu.toLocalMatrix()

    compareBDMs(actual_mean, truth_mu, tolerance=1e-14)
    compareBDMs(actual_ibs0, truth_ibs0, tolerance=1e-14)
    compareBDMs(actual_g, truth_g, tolerance=1e-14)

    assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test(enabled = false)
  def sampleVcfReferenceMatchesR() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val pcs = SamplePCA.justScores(vds.coalesce(10), 2)

    val truth = runPcRelateR(vds, maf=0.01)
    val actual = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1

    assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-2)))
  }

  @Test
  def kinshipFiltering() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val pcs = SamplePCA.justScores(vds.coalesce(10), 2)

    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1
      .filter { case (_, (kin, _, _, _)) => kin >= 0.125 }
    val actual = runPcRelateHail(vds, pcs, 0.01, 0.125)

    assert(truth.size > 0)
    assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-2)))
  }

}
