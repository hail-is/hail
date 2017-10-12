package is.hail.io

import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.io.plink.PlinkLoader
import is.hail.utils._
import is.hail.variant._
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class ImportPlinkSuite extends SparkSuite {

  object Spec extends Properties("ImportPlink") {
    val compGen = for {
      vds <- VariantSampleMatrix.gen(hc, VSMSubgen.random).map(_.cache().splitMulti())
      nPartitions <- choose(1, PlinkLoader.expectedBedSize(vds.nSamples, vds.countVariants()).toInt.min(10))
    } yield (vds, nPartitions)

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantDataset, nPartitions: Int) =>

        val truthRoot = tmpDir.createTempFile("truth")
        val testRoot = tmpDir.createTempFile("test")

        vds.exportPlink(truthRoot)

        if (vds.nSamples == 0) {
          TestUtils.interceptFatal("Empty .fam file") {
            hc.importPlinkBFile(truthRoot, nPartitions = Some(nPartitions))
          }
          true
        } else if (vds.countVariants() == 0) {
          TestUtils.interceptFatal(".bim file does not contain any variants") {
            hc.importPlinkBFile(truthRoot, nPartitions = Some(nPartitions))
          }
          true
        } else {
          hc.importPlinkBFile(truthRoot, nPartitions = Some(nPartitions))
            .annotateGenotypesExpr("g = Genotype(g.GT)")
            .toVDS
            .exportPlink(testRoot)

          val localTruthRoot = tmpDir.createLocalTempFile("truth")
          val localTestRoot = tmpDir.createLocalTempFile("test")

          hadoopConf.copy(truthRoot + ".fam", localTruthRoot + ".fam")
          hadoopConf.copy(truthRoot + ".bim", localTruthRoot + ".bim")
          hadoopConf.copy(truthRoot + ".bed", localTruthRoot + ".bed")

          hadoopConf.copy(testRoot + ".fam", localTestRoot + ".fam")
          hadoopConf.copy(testRoot + ".bim", localTestRoot + ".bim")
          hadoopConf.copy(testRoot + ".bed", localTestRoot + ".bed")

          val exitCodeFam = s"diff ${ uriPath(localTruthRoot) }.fam ${ uriPath(localTestRoot) }.fam" !
          val exitCodeBim = s"diff ${ uriPath(localTruthRoot) }.bim ${ uriPath(localTestRoot) }.bim" !
          val exitCodeBed = s"diff ${ uriPath(localTruthRoot) }.bed ${ uriPath(localTestRoot) }.bed" !

          exitCodeFam == 0 && exitCodeBim == 0 && exitCodeBed == 0
        }
      }

  }

  @Test def testPlinkImportRandom() {
    Spec.check()
  }

  @Test def testA1Major() {
    val plinkFileRoot = tmpDir.createTempFile("plink_reftest")
    hc.importVCF("src/test/resources/sample.vcf")
      .verifyBiallelic()
      .exportPlink(plinkFileRoot)

    val a1ref = hc.importPlinkBFile(plinkFileRoot, a2Reference = false)
      .annotateGenotypesExpr("g = Genotype(g.GT)")
      .toVDS

    val a2ref = hc.importPlinkBFile(plinkFileRoot, a2Reference = true)
      .annotateGenotypesExpr("g = Genotype(g.GT)")
      .toVDS

    val a1kt = a1ref
      .variantQC()
      .variantsKT()
      .select("va.rsid", "v", "va.qc.nNotCalled", "va.qc.nHomRef", "va.qc.nHet", "va.qc.nHomVar")
      .rename(Map("v" -> "vA1", "nNotCalled" -> "nNotCalledA1",
        "nHomRef" -> "nHomRefA1", "nHet" -> "nHetA1", "nHomVar" -> "nHomVarA1"))
      .keyBy("rsid")

    val a2kt = a2ref
      .variantQC()
      .variantsKT()
      .select("va.rsid", "v", "va.qc.nNotCalled", "va.qc.nHomRef", "va.qc.nHet", "va.qc.nHomVar")
      .rename(Map("v" -> "vA2", "nNotCalled" -> "nNotCalledA2",
        "nHomRef" -> "nHomRefA2", "nHet" -> "nHetA2", "nHomVar" -> "nHomVarA2"))
      .keyBy("rsid")

    val joined = a1kt.join(a2kt, "outer")

    assert(joined.forall("vA1.ref == vA2.alt && vA1.alt == vA2.ref && nNotCalledA1 == nNotCalledA2 && " +
      "nHetA1 == nHetA2 && nHomRefA1 == nHomVarA2 && nHomVarA1 == nHomRefA2"))
  }
}
