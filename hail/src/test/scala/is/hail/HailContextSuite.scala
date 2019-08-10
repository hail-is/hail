package is.hail

import org.testng.annotations.Test

class HailContextSuite extends HailSuite {
  @Test def testGetOrCreate(): Unit = {
    val hc2 = HailContext.getOrCreate()
    assert(hc == hc2)
  }
}
