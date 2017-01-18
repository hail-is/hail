package is.hail.io

import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.driver._
import is.hail.io.plink.PlinkLoader
import is.hail.variant._
import is.hail.utils._
import is.hail.SparkSuite
import is.hail.utils.FatalException
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class ImportPlinkSuite extends SparkSuite {

  object Spec extends Properties("ImportPlink") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random);
      nPartitions: Int <- choose(1, PlinkLoader.expectedBedSize(vds.nSamples, vds.nVariants).toInt.min(10))) yield (vds, nPartitions)

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>

        val truthRoot = tmpDir.createTempFile("truth")
        val testRoot = tmpDir.createTempFile("test")

        var s = State(sc, sqlContext, vds)

        s = SplitMulti.run(s, Array[String]())
        s = ExportPlink.run(s, Array("-o", truthRoot))
        if (s.vds.nSamples == 0 || s.vds.nVariants == 0) {
          try {
            s = ImportPlink.run(s, Array("--bfile", truthRoot, "-n", nPartitions.toString))
            false
          } catch {
            case e: FatalException => true
            case _: Throwable => false
          }
        } else {
          s = ImportPlink.run(s, Array("--bfile", truthRoot, "-n", nPartitions.toString))
          s = ExportPlink.run(s, Array("-o", testRoot))

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
