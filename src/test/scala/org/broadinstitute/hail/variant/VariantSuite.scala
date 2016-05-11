package org.broadinstitute.hail.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class VariantSuite extends TestNGSuite {
  @Test def test() {
    import Variant._

    assert(compareContig("3", "26") < 0)
    assert(compareContig("26", "3") > 0)
    assert(compareContig("7", "7") == 0)

    assert(compareContig("3", "X") < 0)
    assert(compareContig("X", "3") > 0)
    assert(compareContig("X", "X") == 0)

    assert(compareContig("X", "Y") < 0)
    assert(compareContig("Y", "X") > 0)
    assert(compareContig("MT", "Y") < 0)
  }
}
