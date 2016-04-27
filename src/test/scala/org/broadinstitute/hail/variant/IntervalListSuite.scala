package org.broadinstitute.hail.variant

import org.testng.annotations.Test
import org.broadinstitute.hail.SparkSuite

class IntervalListSuite extends SparkSuite {

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

    val ex1 = IntervalList.read("src/test/resources/example1.interval_list", hadoopConf)

    val f = tmpDir.createTempFile("example", extension = ".interval_list")
    ex1.write(f, hadoopConf)
    val ex1wr = IntervalList.read(f, hadoopConf)
    assert(ex1wr == ex1)

    val ex2 = IntervalList.read("src/test/resources/example2.interval_list", hadoopConf)
    assert(ex1 == ex2)
  }
}
