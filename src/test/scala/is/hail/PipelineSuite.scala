package is.hail

import is.hail.methods.Pedigree
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
    qc.mendelErrors(Pedigree.fromFam("src/test/resources/sample.fam", hadoopConf))._1.export(mendelBase)
    qc.count()
    qc.filterVariantsExpr("va.qc.AF > 0.01 && va.qc.AF < 0.99")
      .pca("sa.scores")
  }
}
