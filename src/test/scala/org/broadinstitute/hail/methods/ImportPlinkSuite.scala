package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{ImportPlinkBfile, ExportPlink, State, SplitMulti}
import org.broadinstitute.hail.io.LoadVCF
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Gen._
import sys.process._
import scala.language.postfixOps

class ImportPlinkSuite extends SparkSuite {

  object Spec extends Properties("ImportPlink") {
    property("import generates same output as export") =
      forAll(VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _)
        .filter(vds => vds.nVariants > 0 && vds.nSamples > 0), choose(1,50)) { (vds: VariantDataset, nPartitions: Int) =>

        println(s"nPartitions:$nPartitions nSamples:${vds.nSamples} nVariants:${vds.nVariants}")
        var s = State(sc, sqlContext, vds)

        s = SplitMulti.run(s, Array[String]())
        s = ExportPlink.run(s, Array("-o","/tmp/truth"))
        s = ImportPlinkBfile.run(s, Array("--bfile","/tmp/truth", "-n", nPartitions.toString))
        s = SplitMulti.run(s, Array[String]())
        s = ExportPlink.run(s, Array("-o","/tmp/test"))

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

  @Test def testPlinkImportRandom() {
    Spec.check()
  }

  @Test def testImportIdenticalToExport() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val exportState1 = ExportPlink.run(state, Array("-o", "/tmp/hailOut"))

    val importState1 = ImportPlinkBfile.run(state, Array("--bfile","/tmp/hailOut"))
    val splitState1 = SplitMulti.run(importState1, Array.empty[String])
    val exportState2 = ExportPlink.run(splitState1, Array("-o", "/tmp/hailOut2"))
    val importState2 = ImportPlinkBfile.run(state, Array("--bfile","/tmp/hailOut2"))

    assert(importState1.vds.same(importState2.vds))

    val exitCodeFam = "diff /tmp/hailOut.fam /tmp/hailOut2.fam" !
    val exitCodeBim = "diff /tmp/hailOut.bim /tmp/hailOut2.bim" !
    val exitCodeBed = "diff /tmp/hailOut.bed /tmp/hailOut2.bed" !

    assert(exitCodeFam == 0 && exitCodeBim == 0 && exitCodeBed == 0)

  }


}
