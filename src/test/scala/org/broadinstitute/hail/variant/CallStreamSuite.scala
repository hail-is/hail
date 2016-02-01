package org.broadinstitute.hail.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class HardCallSetSuite extends TestNGSuite {
  @Test def test() {
    val d = DenseCallStream(Array[Int](0,1,2,3,3,2,1,0,1,2))

    println(d)

    d.showBinary()
  }
}
