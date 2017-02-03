package is.hail

import is.hail.driver._
import org.testng.annotations.Test

class PipelineSuite extends SparkSuite {
  @Test def defaultPipeline() {
    val vdsFile = tmpDir.createTempFile("sample", extension = ".vds")
    val sampleQCFile = tmpDir.createTempFile("sampleqc", extension = ".tsv")
    val variantQCFile = tmpDir.createTempFile("variantqc", extension = ".tsv")
    val gqByDPFile = tmpDir.createTempFile("gqbydp", extension = ".tsv")
    val mendelBase = tmpDir.createTempFile("mendel")
    val pcaFile = tmpDir.createTempFile("pca", extension = ".tsv")

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    vds.splitMulti()
      .write(vdsFile)

    val qc = hc.read(vdsFile)
      .sampleQC()
      .variantQC()

    qc.gqByDP(gqByDPFile)
    qc.mendelErrors(mendelBase, "src/test/resources/sample.fam")
    qc.count()
    qc.filterVariantsExpr("va.qc.AF > 0.01 && va.qc.AF < 0.99")
      .pca("sa.scores")

    /*
    val linregFile = tmpDir.createTempFile("linreg", extension = ".tsv")

    s = LinearRegressionCommand.run(s, Array("-c", pcaFile,
      "-f", "src/test/resources/sample.fam",
      "-o", linregFile))
    */
  }
}
