package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{SexCheck, State}
import org.testng.annotations.Test
import org.broadinstitute.hail.driver.SexCheck._

class SexCheckSuite extends SparkSuite {
  @Test def testInbreedingCoefficient() = {
    val vcfFile = "src/test/resources/chrX_1kg_thinned2.nobadalt.vcf"
    val vds = LoadVCF(sc, vcfFile, nPartitions = Some(10))
    val state = State(sc, sqlContext, vds)

    SexCheck.run(state,Array("-o","/tmp/sexcheck.test"))
    assert(1==1)
  }
}
