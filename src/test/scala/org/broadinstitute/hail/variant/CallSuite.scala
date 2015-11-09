package org.broadinstitute.hail.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class CallSuite extends TestNGSuite {
  @Test def testCall() {
    val c = Call(1, 50, (100, 0, 1000))
    assert(c.gt == 1)
    assert(c.gq == 50)
    assert(c.pl == (100, 0, 1000))

    intercept[IllegalArgumentException] {
      Call(-1, 50, (0, 0, 0))
    }
    intercept[IllegalArgumentException] {
      Call(1, 200, (100, 0, 1000))
    }
    intercept[IllegalArgumentException] {
      Call(1, 50, (100, 100, 1000))
    }
  }
}
