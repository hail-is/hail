package is.hail.stats

import is.hail.HailSuite

import org.testng.annotations.Test

class FisherExactTestSuite extends HailSuite {

  @Test def testPvalue(): Unit = {
    val N = 200
    val K = 100
    val k = 10
    val n = 15
    val a = 5
    val b = 10
    val c = 95
    val d = 90

    val result = fisherExactTest(a, b, c, d)

    assert(math.abs(result(0) - 0.2828) < 1e-4)
    assert(math.abs(result(1) - 0.4754059) < 1e-4)
    assert(math.abs(result(2) - 0.122593) < 1e-4)
    assert(math.abs(result(3) - 1.597972) < 1e-4)
  }
}
