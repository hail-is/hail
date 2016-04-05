package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.State
import org.testng.annotations.Test
import org.broadinstitute.hail.driver._

class SampleQCSuite extends SparkSuite {
  @Test def testStoreAfterFilter() {
    // test to show that code in SampleQC --store will fail if samples have been filtered when adding annotations
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val tmpDir = "/tmp/"
    val outFile = tmpDir + "testExportVcf.vcf"

    val vdsOrig = LoadVCF(sc, vcfFile)
    var s = State(sc, sqlContext, vdsOrig)
    s = SplitMulti.run(s, Array.empty[String])

    // Get QC metrics
    s = SampleQC.run(s, Array.empty[String])
    s = FilterSamples.run(s, Array("--remove", "-c","""s.id ~ "C1046::HG02024""""))
    s = SampleQC.run(s, Array.empty[String])
  }
}
