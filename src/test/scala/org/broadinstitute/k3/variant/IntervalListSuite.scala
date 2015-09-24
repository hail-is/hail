package org.broadinstitute.k3.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class IntervalListSuite extends TestNGSuite {
  @Test def test() {
    val ilist = IntervalList(List(
      Interval("1", 10, 20),
      Interval("1", 30, 40),
      Interval("2", 40, 50)))

    assert(!ilist.contains("1", 5))
    assert(ilist.contains("1", 10))
    assert(ilist.contains("1", 15))
    assert(ilist.contains("1", 20))
    assert(!ilist.contains("1", 25))
    assert(ilist.contains("1", 35))

    assert(!ilist.contains("2", 30))
    assert(ilist.contains("2", 45))

    assert(!ilist.contains("3", 0))

    val ex1 = IntervalList.read("src/test/resources/example1.interval_list")
    // FIXME This is not the right way to handle tmp files.  See issue #60.
    ex1.write("/tmp/example1.interval_list")
    val ex1wr = IntervalList.read("/tmp/example1.interval_list")
    assert(ex1wr == ex1)

    val ex2 = IntervalList.read("src/test/resources/example2.interval_list")
    assert(ex1 == ex2)
  }
}
