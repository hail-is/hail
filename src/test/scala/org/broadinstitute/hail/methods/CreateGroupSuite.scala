package org.broadinstitute.hail.methods

import org.broadinstitute.hail.driver.State
import org.broadinstitute.hail.{SparkSuite, TempDir}
import org.testng.annotations.Test
import org.broadinstitute.hail.driver._

class CreateGroupSuite extends SparkSuite {
  @Test def foo1() {
    var s = State(sc, sqlContext)
    val vcf = "src/test/resources/sample.vcf.bgz"
    s = ImportVCF.run(s, Array(vcf))
    s = SplitMulti.run(s, Array.empty[String])
    s = VariantQC.run(s,Array.empty[String])
    s = CreateGroup.run(s, Array("-k","va.qc.MAC","-a","carrier", "-v","g.nNonRefAlleles"))

    println(s.group.take(1).mkString(","))
  }
}
