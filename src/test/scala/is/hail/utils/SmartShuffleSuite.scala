package is.hail.utils

import is.hail.SparkSuite
import is.hail.driver.{Count, Read, SplitMulti, State}
import org.testng.annotations.Test

class SmartShuffleSuite extends SparkSuite {

  @Test def test() {

    val s = Read.run(State(sc, sqlContext, null), "/Users/tpoterba/data/profile225.vds")
    printTime(
      Count.run(SplitMulti.run(s))
    )
  }
}
