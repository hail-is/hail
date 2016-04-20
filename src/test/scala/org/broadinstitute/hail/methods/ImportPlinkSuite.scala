package org.broadinstitute.hail.methods

import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.io.plink.PlinkLoader
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class ImportPlinkSuite extends SparkSuite {

  object Spec extends Properties("ImportPlink") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _);
    nPartitions: Int <- choose(1,PlinkLoader.expectedBedSize(vds.nSamples,vds.nVariants).toInt.min(10))) yield (vds, nPartitions)

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>

        var s = State(sc, sqlContext, vds)

        s = SplitMulti.run(s, Array[String]())
        s = ExportPlink.run(s, Array("-o","/tmp/truth"))
        if (vds.nSamples == 0 || vds.nVariants == 0)
          try {
            s = ImportPlink.run(s, Array("--bfile","/tmp/truth", "-n", nPartitions.toString))
            false
          } catch {
            case e:FatalException => true
            case _: Throwable => false
          }
        else {
          s = ImportPlink.run(s, Array("--bfile","/tmp/truth", "-n", nPartitions.toString))
          s = SplitMulti.run(s, Array[String]())
          s = ExportPlink.run(s, Array("-o", "/tmp/test"))

          val exitCodeFam = "diff /tmp/truth.fam /tmp/test.fam" !
          val exitCodeBim = "diff /tmp/truth.bim /tmp/test.bim" !
          val exitCodeBed = "diff /tmp/truth.bed /tmp/test.bed" !

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
