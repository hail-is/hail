package is.hail.methods

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.SparkSuite
import is.hail.distributedmatrix.BlockMatrix
import is.hail.expr.{TFloat64, TString}
import is.hail.stats._
import is.hail.utils.{TextTableReader, _}
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.sys.process._

class PCRelateSuite extends SparkSuite {
  private val blockSize: Int = 8192

  private def toD(a: java.lang.Double): Double =
    a.asInstanceOf[Double]

  private def toD(a: Any): Double =
    a.asInstanceOf[Double]

  private def toBoxedD(a: Any): java.lang.Double =
    a.asInstanceOf[java.lang.Double]

  private def toS(a: Any): String =
    a.asInstanceOf[String]

  private def quadMap[T,U](f: T => U): (T, T, T, T) => (U, U, U, U) =
    { case (x, y, z, w) => (f(x), f(y), f(z), f(w)) }

  def runPcRelateHail(vds: VariantDataset, pcs: DenseMatrix, maf: Double): Map[(String, String), (Double, Double, Double, Double)] =
    runPcRelateHail(vds, pcs, maf, PCRelate.defaultMinKinship, PCRelate.defaultStatisticSubset)
      .mapValues(quadMap(toD).tupled)

  def runPcRelateHail(vds: VariantDataset, pcs: DenseMatrix, maf: Double, minKinship: Double): Map[(String, String), (Double, Double, Double, Double)] =
    runPcRelateHail(vds, pcs, maf, minKinship, PCRelate.defaultStatisticSubset)
      .mapValues(quadMap(toD).tupled)

  def runPcRelateHail(vds: VariantDataset, pcs: DenseMatrix, maf: Double, minKinship: Double, statistics: PCRelate.StatisticSubset): Map[(String, String), (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)] =
    PCRelate.toKeyTable(vds, pcs, maf, blockSize, minKinship, statistics)
      .collect()
      .map(x => x.asInstanceOf[Row])
      .map(r => ((r(0), r(1)), (r(2), r(3), r(4), r(5))))
      .toMap
      .mapValues(quadMap(toBoxedD).tupled)
      .asInstanceOf[Map[(String, String), (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)]]

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
      ("nsnp", TFloat64),
      ("kin", TFloat64),
      ("k0", TFloat64),
      ("k1", TFloat64),
      ("k2", TFloat64))
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

  def compareDoubleQuadruplet(cmp: (Double, Double) => Boolean)(x: (Double, Double, Double, Double), y: (Double, Double, Double, Double)): Boolean =
    cmp(x._1, y._1) && cmp(x._2, y._2) && cmp(x._3, y._3) && cmp(x._4, y._4)

  @Test
  def trivialReference() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromGtMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new DenseMatrix(4,2,pcsArray)
    val us = runPcRelateHail(vds, pcs, maf=0.0)
    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.0)._1
    assert(mapSameElements(us, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test(enabled = false)
  def trivialReferenceMatchesR() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromGtMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new DenseMatrix(4,2,pcsArray)
    val usRef = PCRelateReferenceImplementation(vds, pcs, maf=0.0)._1
    val truth = runPcRelateR(vds, maf=0.0, "src/test/resources/is/hail/methods/runPcRelateOnTrivialExample.R")
    assert(mapSameElements(usRef, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test
  def baldingNicholsMatchesReference() {
    val seed = 0
    val n = 100
    val nVariants = 10000
    val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
    val pcs = SamplePCA.justScores(vds, 2)
    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1
    val actual = runPcRelateHail(vds, pcs, maf=0.01)

    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test(enabled = false)
  def baldingNicholsReferenceMatchesR() {
    val seed = 0
    val n = 100
    val nVariants = 10000
    val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
    val pcs = SamplePCA.justScores(vds, 2)
    val truth = runPcRelateR(vds, maf=0.01)
    val actual = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1

    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => D_==(x, y, tolerance=1e-2))))
  }

  private def compareBDMs(l: BDM[Double], r: BDM[Double], tolerance: Double) {
    val fails = l.toArray.zip(r.toArray).zipWithIndex.flatMap { case ((actual, truth), idx) =>
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
      .verifyBiallelic()

    val pcs = SamplePCA.justScores(vds.coalesce(10), 2)

    val (truth, truth_g, truth_ibs0, truth_mu) = PCRelateReferenceImplementation(vds, pcs, maf=0.01)

    val pcr = new PCRelate(0.01, blockSize)
    val g = PCRelate.vdsToMeanImputedMatrix(vds)
    val dmu = pcr.mu(g, pcs)
    // blockedG : variant x sample
    val blockedG = BlockMatrix.from(g, blockSize)
    val actual = runPcRelateHail(vds, pcs, 0.01)
    val actual_g = blockedG.toLocalMatrix().t
    val actual_ibs0 = pcr.ibs0(blockedG, dmu, blockSize).toLocalMatrix()
    val actual_mean = dmu.toLocalMatrix()

    compareBDMs(actual_mean, truth_mu, tolerance=1e-14)
    compareBDMs(actual_ibs0, truth_ibs0, tolerance=1e-14)
    compareBDMs(actual_g, truth_g, tolerance=1e-14)

    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test(enabled = false)
  def sampleVcfReferenceMatchesR() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz").splitMulti()

    val pcs = SamplePCA.justScores(vds.coalesce(10), 2)

    val truth = runPcRelateR(vds, maf=0.01)
    val actual = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1

    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-2)))
  }

  @Test
  def kinshipFiltering() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz").splitMulti()

    val pcs = SamplePCA.justScores(vds.coalesce(10), 2)

    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1
      .filter { case (_, (kin, _, _, _)) => kin >= 0.125 }
    val actual = runPcRelateHail(vds, pcs, 0.01, 0.125)

    assert(truth.size > 0)
    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-2)))
  }

  @Test
  def desire() {
    val seed = 0
    val n = 100
    val nVariants = 1000
    val vds = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
    val pcs = SamplePCA.justScores(vds, 10)
    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1
      .mapValues(quadMap(toBoxedD).tupled)
    val phionly = runPcRelateHail(vds, pcs, 0.01, PCRelate.defaultMinKinship, statistics=PCRelate.PhiOnly)

    assert(mapSameElements(phionly, truth,
      { (l: (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double), r: (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)) =>
        math.abs(l._1 - r._1) < 1e-14 &&
        l._2 == null &&
        l._3 == null &&
        l._4 == null }))

    val phik2 = runPcRelateHail(vds, pcs, 0.01, PCRelate.defaultMinKinship, statistics=PCRelate.PhiK2)

    assert(mapSameElements(phik2, truth,
      { (l: (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double), r: (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)) =>
        math.abs(l._1 - r._1) < 1e-14 &&
        l._2 == null &&
        l._3 == null &&
        math.abs(l._4 - r._4) < 1e-14}))

    val phik2k0 = runPcRelateHail(vds, pcs, 0.01, PCRelate.defaultMinKinship, statistics=PCRelate.PhiK2K0)

    assert(mapSameElements(phik2k0, truth,
      { (l: (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double), r: (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)) =>
        math.abs(l._1 - r._1) < 1e-14 &&
        math.abs(l._2 - r._2) < 1e-14 &&
        l._3 == null &&
        math.abs(l._4 - r._4) < 1e-14}))

    val phik2k0k1 = runPcRelateHail(vds, pcs, 0.01, PCRelate.defaultMinKinship, PCRelate.PhiK2K0K1)

    assert(mapSameElements(
      phik2k0k1.mapValues(quadMap(toD).tupled),
      truth.mapValues(quadMap(toD).tupled),
      compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

}
