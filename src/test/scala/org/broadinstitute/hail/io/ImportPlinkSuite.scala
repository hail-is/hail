package org.broadinstitute.hail.io

import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.io.plink.PlinkLoader
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils.FatalException
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
