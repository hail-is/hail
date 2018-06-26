package is.hail.methods

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.{SparkSuite, TestUtils}
import is.hail.linalg.BlockMatrix
import is.hail.expr.types._
import is.hail.stats._
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.MatrixTable
import is.hail.table.Table
import org.testng.annotations.Test
import scala.language.postfixOps

import scala.sys.process._

object PCRelateSuite {

  def samplePCA(
    vsm: MatrixTable,
    k: Int = 10,
    computeLoadings: Boolean = false): (IndexedSeq[Double], BDM[Double], Option[Table]) = {

    var prePCA = vsm.annotateRowsExpr(
      "AC" -> "AGG.map(g => g.GT.nNonRefAlleles().toInt64()).sum().toInt32()", "nCalled" -> "AGG.filter(g => isDefined(g.GT)).count()")
      .filterRowsExpr("va.AC > 0 && va.AC.toInt64 < 2L * va.nCalled").persist()
    val nVariants = prePCA.countRows()
    prePCA = prePCA.selectEntries(s"{__gt: let mean = va.AC.toFloat64 / va.nCalled.toFloat64 in if (isDefined(g.GT)) (g.GT.nNonRefAlleles().toFloat64 - mean) / sqrt(mean * (2d - mean) * ${ nVariants }d / 2d) else 0d}")

    PCA(prePCA, "__gt", k, computeLoadings)
  }
}

class PCRelateSuite extends SparkSuite {
  private val blockSize: Int = BlockMatrix.defaultBlockSize

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

  def runPcRelateHail(
    vds: MatrixTable,
    pcs: BDM[Double],
    maf: Double,
    minKinship: Double = PCRelate.defaultMinKinship,
    statistics: PCRelate.StatisticSubset = PCRelate.defaultStatisticSubset): Map[(String, String), (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)] =
    PCRelate(hc, vds, pcs, maf, blockSize, minKinship, statistics)
      .collect()
      .map(r => ((r(0), r(1)), (r(2), r(3), r(4), r(5))))
      .toMap
      .mapValues(quadMap(toBoxedD).tupled)
      .asInstanceOf[Map[(String, String), (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)]]

  def runPcRelateR(
    vds: MatrixTable,
    maf: Double,
    rFile: String = "src/test/resources/is/hail/methods/runPcRelate.R"): Map[(String, String), (Double, Double, Double, Double)] = {

    val tmpfile = tmpDir.createTempFile(prefix = "pcrelate")
    val localTmpfile = tmpDir.createLocalTempFile(prefix = "pcrelate")

    TestUtils.exportPlink(vds, tmpfile)

    for (suffix <- Seq(".bed", ".bim", ".fam")) {
      hadoopConf.copy(tmpfile + suffix, localTmpfile + suffix)
    }

    s"Rscript $rFile ${uriPath(localTmpfile)} $maf" !

    val columns = Map(
      ("ID1", TString()),
      ("ID2", TString()),
      ("nsnp", TFloat64()),
      ("kin", TFloat64()),
      ("ibd0", TFloat64()),
      ("ibd1", TFloat64()),
      ("ibd2", TFloat64()))
    val separator = " +"

    hadoopConf.copy(localTmpfile + ".out", tmpfile + ".out")

    val rdd = TextTableReader.read(hc)(Array(tmpfile + ".out"), columns, separator = separator).rdd
    rdd.collect()
      .map { row =>
        val id1 = toS(row(0))
        val id2 = toS(row(1))
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
    val vds = vdsFromCallMatrix(hc)(TestUtils.unphasedDiploidGtIndicesToBoxedCall(genotypeMatrix))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new BDM(4, 2, pcsArray)
    val us = runPcRelateHail(vds, pcs, maf=0.0).mapValues(quadMap(toD).tupled)
    val truth = PCRelateReferenceImplementation(vds, pcs)._1
    assert(mapSameElements(us, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test(enabled = false)
  def trivialReferenceMatchesR() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromCallMatrix(hc)(TestUtils.unphasedDiploidGtIndicesToBoxedCall(genotypeMatrix), Some(Array("s1","s2","s3","s4")))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new BDM(4, 2, pcsArray)
    val usRef = PCRelateReferenceImplementation(vds, pcs)._1
    val truth = runPcRelateR(vds, maf=0.0, "src/test/resources/is/hail/methods/runPcRelateOnTrivialExample.R")
    assert(mapSameElements(usRef, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test
  def baldingNicholsMatchesReference() {
    val seed = 0
    val n = 100
    val nVariants = 10000
    val vds: MatrixTable = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9))
      .annotateColsExpr("s" -> "str(sa.sample_idx)").keyColsBy("s")
    val pcs = PCRelateSuite.samplePCA(vds, 2)._2
    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1
    val actual = runPcRelateHail(vds, pcs, maf=0.01).mapValues(quadMap(toD).tupled)

    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test(enabled = false)
  def baldingNicholsReferenceMatchesR() {
    val seed = 0
    val n = 100
    val nVariants = 10000
    val vds: MatrixTable = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9))
    val pcs = PCRelateSuite.samplePCA(vds, 2)._2
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
      fails.foreach(println)
    assert(fails.isEmpty)
  }

  @Test
  def sampleVcfMatchesReference() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val pcs = PCRelateSuite.samplePCA(vds.coalesce(10), 2)._2

    val (truth, truth_g, truth_ibs0, truth_mu) = PCRelateReferenceImplementation(vds, pcs, maf=0.01)

    val pcr = new PCRelate(0.01, blockSize, PCRelate.PhiK2K0K1, PCRelate.defaultStorageLevel)
    val g = PCRelate.vdsToMeanImputedMatrix(vds)
    
    // blockedG : variant x sample
    val blockedG = BlockMatrix.fromIRM(g, blockSize)

    val predmu = pcr.mu(BlockMatrix.fromIRM(g, blockSize), pcs)
    
    val dmu = BlockMatrix.map2 { (g, mu) =>
      def badmu(mu: Double, maf: Double): Boolean =
        mu <= maf || mu >= (1.0 - maf) || mu <= 0.0 || mu >= 1.0
  
      def badgt(gt: Double): Boolean =
        gt != 0.0 && gt != 1.0 && gt != 2.0
      
      if (badgt(g) || badmu(mu, 0.01))
        Double.NaN
      else
        mu
    } (blockedG, predmu).cache()
    
    val actual = runPcRelateHail(vds, pcs, 0.01).mapValues(quadMap(toD).tupled)
    val actual_g = blockedG.toBreezeMatrix().t
    val actual_ibs0 = pcr.ibs0(blockedG, dmu, blockSize).toBreezeMatrix()
    val actual_mean = dmu.toBreezeMatrix()

    compareBDMs(actual_mean, truth_mu, tolerance=1e-14)
    compareBDMs(actual_ibs0, truth_ibs0, tolerance=1e-14)
    compareBDMs(actual_g, truth_g, tolerance=1e-14)

    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test(enabled = false)
  def sampleVcfReferenceMatchesR() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val pcs = PCRelateSuite.samplePCA(vds.coalesce(10), 2)._2

    val truth = runPcRelateR(vds, maf=0.01)
    val actual = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1

    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-2)))
  }

  @Test
  def kinshipFiltering() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val pcs = PCRelateSuite.samplePCA(vds.coalesce(10), 2)._2

    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)._1
      .filter { case (_, (kin, _, _, _)) => kin >= 0.125 }
    val actual = runPcRelateHail(vds, pcs, 0.01, 0.125).mapValues(quadMap(toD).tupled)

    assert(truth.nonEmpty)
    assert(mapSameElements(actual, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-2)))
  }

  @Test
  def desire() {
    val seed = 0
    val n = 100
    val nVariants = 1000
    val vds = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9))
      .annotateColsExpr("s" -> "str(sa.sample_idx)").keyColsBy("s")
    val pcs = PCRelateSuite.samplePCA(vds)._2
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
