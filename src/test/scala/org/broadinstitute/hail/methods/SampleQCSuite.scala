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
    val stateOrig = State("", sc, sqlContext, vdsOrig)

    //Get QC metrics
    val stateQC1 = SampleQC.run(stateOrig,Array("-s"))
    val stateFilt1 = FilterSamples.run(stateQC1,Array("--remove","-c","src/test/resources/remove1sample.sample_list"))
    val stateQC2 = SampleQC.run(stateFilt1,Array("-s"))
  }
}