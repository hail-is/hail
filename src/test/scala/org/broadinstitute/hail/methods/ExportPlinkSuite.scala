package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{State, ExportPlink}
import org.broadinstitute.hail.utils.TestRDDBuilder
import org.testng.annotations.Test


class ExportPlinkSuite extends SparkSuite {

  @Test def test() {

    val vds = TestRDDBuilder.buildRDD(10, 10, sc)
    val state = State("", sc, sqlContext, vds)

    println(vds.localSamples.mkString(","))
    println(vds.sampleIds.mkString(","))
    ExportPlink.run(state, Array("-o", "/tmp/test", "-t" ,"/tmp/todelete", "--cutoff", "20"))
  }
}
