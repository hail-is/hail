package org.broadinstitute.hail.io

import org.broadinstitute.hail.SparkSuite
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class LoadPlinkSuite extends SparkSuite {
  @Test def test() {
    val vds = PlinkLoader("src/test/resources/PlinkSample", sc)
    val time = System.nanoTime()
    vds.variants.count()
    println("took %.3f seconds to parse".format((System.nanoTime() - time)/1e9 ))
  }
}
