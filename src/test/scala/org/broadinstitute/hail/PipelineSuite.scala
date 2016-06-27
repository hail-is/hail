package org.broadinstitute.hail

import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

class PipelineSuite extends SparkSuite {
  @Test def defaultPipeline() {
    val vdsFile = tmpDir.createTempFile("sample", extension = ".vds")
    val sampleQCFile = tmpDir.createTempFile("sampleqc", extension = ".tsv")
    val variantQCFile = tmpDir.createTempFile("variantqc", extension = ".tsv")
    val gqByDPFile = tmpDir.createTempFile("gqbydp", extension = ".tsv")
    val mendelBase = tmpDir.createTempFile("mendel")
    val pcaFile = tmpDir.createTempFile("pca", extension = ".tsv")

    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("-i", "src/test/resources/sample.vcf"))
    s = SplitMulti.run(s, Array.empty[String])
    s = Write.run(s, Array("-o", vdsFile))
    s = Read.run(s, Array("-i", vdsFile))
    s = SampleQC.run(s, Array("-o", sampleQCFile))
    s = VariantQC.run(s, Array.empty[String])
    s = GQByDP.run(s, Array("-o", gqByDPFile))
    s = MendelErrorsCommand.run(s, Array("-f", "src/test/resources/sample.fam",
      "-o", mendelBase))
    s = Count.run(s, Array.empty[String])

    s = FilterVariantsExpr.run(s, Array("--keep", "-c", "va.qc.AF > 0.01 && va.qc.AF < 0.99"))

    s = PCA.run(s, Array("-o", pcaFile))

    /*
    val linregFile = tmpDir.createTempFile("linreg", extension = ".tsv")

    s = LinearRegressionCommand.run(s, Array("-c", pcaFile,
      "-f", "src/test/resources/sample.fam",
      "-o", linregFile))
    */
  }
}
