package org.broadinstitute.hail.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class CallStreamSuite extends TestNGSuite {
  @Test def test() {
    val d = DenseCalls(Array[Int](0,1,2,3,3,2,1,0,1,2))

    println(d)

    d.showBinary()

    println(d.iterator.toList)
  }
}
