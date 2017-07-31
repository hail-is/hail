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
}
