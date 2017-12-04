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

    val qc = hc.readVDS(vdsFile)
      .sampleQC()
      .variantQC()

    val pedigree = Pedigree.read("src/test/resources/sample.fam", hadoopConf)
    qc.mendelErrors(pedigree)._1.export(mendelBase)
    qc.count()
    val prePCA = qc.filterVariantsExpr("va.qc.AF > 0.01 && va.qc.AF < 0.99")
      .annotateVariantsExpr("va.mean = gs.map(g => g.GT.gt).sum()/gs.filter(g => g.GT.isDefined).count()")
      .filterVariantsExpr(s"isDefined(va.mean) && va.mean != 0 && va.mean != 2").persist()
    prePCA.pca(s"if (g.GT.isDefined) (g.GT.gt-va.mean)/sqrt(va.mean * (2- va.mean) * ${ prePCA.countVariants() }/ 2) else 0")
    prePCA.unpersist()
  }
}
