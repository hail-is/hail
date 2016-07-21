package org.broadinstitute.hail.io

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.io.plink.PlinkLoader
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class ImportPlinkSuite extends SparkSuite {

  object Spec extends Properties("ImportPlink") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random);
      nPartitions: Int <- choose(1, PlinkLoader.expectedBedSize(vds.nSamples, vds.nVariants).toInt.min(10))) yield (vds, nPartitions)

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>

        val tmpTruthRoot = tmpDir.createTempFile(prefix = "truth")
        val tmpTestRoot = tmpDir.createTempFile(prefix = "test")

        var s = State(sc, sqlContext, vds)

        s = SplitMulti.run(s, Array[String]())
        val save = s.vds
        s = ExportPlink.run(s, Array("-o", tmpTruthRoot))
        if (s.vds.nSamples == 0 || s.vds.nVariants == 0)
          try {
            s = ImportPlink.run(s, Array("--bfile", tmpTruthRoot, "-n", nPartitions.toString))
            false
          } catch {
            case e: FatalException => true
            case _: Throwable => false
          }
        else {
          s = ImportPlink.run(s, Array("--bfile", tmpTruthRoot, "-n", nPartitions.toString))
          s = ExportPlink.run(s, Array("-o", tmpTestRoot))

          val exitCodeFam = s"diff $tmpTruthRoot.fam $tmpTestRoot.fam" !
          val exitCodeBim = s"diff $tmpTruthRoot.bim $tmpTestRoot.bim" !
          val exitCodeBed = s"diff $tmpTruthRoot.bed $tmpTestRoot.bed" !

          if (exitCodeFam == 0 && exitCodeBim == 0 && exitCodeBed == 0)
            true
          else {
            false
          }
        }
      }
  }

  @Test def testPlinkImportRandom() {
    Spec.check()
  }
}
