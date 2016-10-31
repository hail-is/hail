package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.testng.annotations.Test

class AnnotateSamplesSuite extends SparkSuite {

  @Test def testVDS() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = State(sc, sqlContext, vds)

    s = Put.run(s, "-n", "sample")

    s = AnnotateSamplesVDS.run(s, Array("-n", "sample", "-r", "sa.other"))

    val (_, q) = s.vds.querySA("sa.other")
    assert(vds.sampleIdsAndAnnotations == s.vds.sampleIdsAndAnnotations.map { case (id, anno) => (id, q(anno).orNull) })
  }
}
