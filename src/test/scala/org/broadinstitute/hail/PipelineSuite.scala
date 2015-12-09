package org.broadinstitute.hail

import java.io.File
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

class PipelineSuite extends SparkSuite {
  @Test def defaultPipeline() {
    // FIXME move to State object
    val jar = getClass.getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    sc.addJar(jar)

    val installDir = new File(jar).getParent + "/.."

    var s = State(installDir, sc, sqlContext, null)

    s = Import.run(s, Array("-i", "src/test/resources/sample.vcf"))
    s = Write.run(s, Array("-o", "/tmp/sample.vds"))
    s = Read.run(s, Array("-i", "/tmp/sample.vds"))
    s = SampleQC.run(s, Array("-o", "/tmp/sampleqc.tsv"))
    s = VariantQC.run(s, Array("-o", "/tmp/variantqc.tsv"))
    s = GQByDP.run(s, Array("-o", "/tmp/gqbydp.tsv"))
    s = MendelErrorsCommand.run(s, Array("-f", "src/test/resources/sample.fam",
      "-o", "/tmp/mendel"))

    s = PCA.run(s, Array("-o", "/tmp/pca.tsv"))

    s = LinearRegressionCommand.run(s, Array("-c", "/tmp/pca.tsv",
      "-f", "src/test/resources/sample.fam",
      "-o", "/tmp/linreg.tsv"))
  }
}
