package is.hail

import is.hail.backend.spark.SparkBackend

import org.scalatest
import org.testng.annotations.Test

class HailContextSuite extends HailSuite {
  @Test def testGetOrCreate(): scalatest.Assertion = {
    val backend = SparkBackend.getOrCreate()
    val hc2 = HailContext.getOrCreate(backend)
    assert(hc == hc2)
  }
}
