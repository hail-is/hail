package org.broadinstitute.hail.io

import org.broadinstitute.hail.SparkSuite
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class LoadPlinkSuite extends SparkSuite {
  @Test def test() {
    val bfile = "src/test/resources/PlinkSample"
    val vds = PlinkLoader(bfile + ".bed", bfile + ".bim", bfile + ".fam", sc)
    val time = System.nanoTime()
    vds.variants.count()
    println("took %.3f seconds to parse".format((System.nanoTime() - time)/1e9 ))
  }
}
