package org.broadinstitute.k3.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class IntervalListSuite extends TestNGSuite {
  @Test def test() {
    val ilist = IntervalList()

    ilist += ("1" -> (10, 20))
    ilist += ("1" -> (30, 40))
    ilist += ("2" -> (40, 50))

    assert(!ilist.contains("1", 5))
    assert(ilist.contains("1", 10))
    assert(ilist.contains("1", 15))
    assert(ilist.contains("1", 20))
    assert(!ilist.contains("1", 25))
    assert(ilist.contains("1", 35))

    assert(!ilist.contains("2", 30))
    assert(ilist.contains("2", 45))

    assert(!ilist.contains("3", 0))
  }
}
