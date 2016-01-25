package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{FilterSamples, AnnotateSamples, State}
import org.testng.annotations.Test

class AnnotateFromFilesSuite extends SparkSuite {

  @Test def testSampleTSV {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    println(vds.nSamples)

    val state = State(sc, sqlContext, vds)

    val anno1 = AnnotateSamples.run(state, Array("-c", "src/test/resources/sampleAnnotations.tsv", "-s", "Sample", "-t", "qPhen=Int"))
    val filt1 = FilterSamples.run(anno1, Array("--keep", "-c", "sa.Status == \"CASE\""))
    println(filt1.vds.nLocalSamples)

    val anno2 = AnnotateSamples.run(state, Array("-c", "src/test/resources/sampleAnnotations.tsv", "-s", "Sample", "-n", "phenotype"))
    val filt2 = FilterSamples.run(anno2, Array("--keep", "-c", "sa.phenotype.Status == \"CASE\""))
    println(filt2.vds.nLocalSamples)

  }

}
