package org.broadinstitute.hail

import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

class PipelineSuite extends SparkSuite {
  @Test def defaultPipeline() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))
    s = SplitMulti.run(s, Array.empty[String])
    s = Write.run(s, Array("-o", "/tmp/sample.vds"))
    s = Read.run(s, Array("-i", "/tmp/sample.vds"))
    s = SampleQC.run(s, Array("-o", "/tmp/sampleqc.tsv"))
    s = VariantQC.run(s, Array.empty[String])
    s = GQByDP.run(s, Array("-o", "/tmp/gqbydp.tsv"))
    s = MendelErrorsCommand.run(s, Array("-f", "src/test/resources/sample.fam",
      "-o", "/tmp/mendel"))
    s = Count.run(s, Array.empty[String])

    s = FilterVariants.run(s, Array("--keep", "-c", "va.qc.MAF > 0.01 && va.qc.MAF < 0.99"))

    s = PCA.run(s, Array("-o", "/tmp/pca.tsv"))

    /*
    s = LinearRegressionCommand.run(s, Array("-c", "/tmp/pca.tsv",
      "-f", "src/test/resources/sample.fam",
      "-o", "/tmp/linreg.tsv"))
    */
  }
}
