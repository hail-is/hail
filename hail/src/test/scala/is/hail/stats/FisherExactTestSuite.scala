package is.hail.stats

import is.hail.HailSuite
import is.hail.utils.D_==
import org.testng.annotations.Test

class FisherExactTestSuite extends HailSuite {

  @Test def testPvalue(): Unit = {
    val a = 5
    val b = 10
    val c = 95
    val d = 90

    val result = fisherExactTest(a, b, c, d)

    assert(D_==(result(0), 0.2828, 1e-4))
    assert(D_==(result(1), 0.4754059, 1e-4))
    assert(D_==(result(2), 0.122593, 1e-4))
    assert(D_==(result(3), 1.597972, 1e-4))
  }

  @Test def testPvalue2(): Unit = {
    val a = 10
    val b = 5
    val c = 90
    val d = 95

    val result = fisherExactTest(a, b, c, d)

    assert(D_==(result(0), 0.2828, 1e-4))
  }

  @Test def test_basic(): Unit = {
    // test cases taken from scipy/stats/tests/test_stats.py
    var res = fisherExactTestPValueOnly(14500, 20000, 30000, 40000)
    assert(D_==(res, 0.01106, 1e-3))
    res = fisherExactTestPValueOnly(100, 2, 1000, 5)
    assert(D_==(res, 0.1301, 1e-3))
    res = fisherExactTestPValueOnly(2, 7, 8, 2)
    assert(D_==(res, 0.0230141, 1e-5))
    res = fisherExactTestPValueOnly(5, 1, 10, 10)
    assert(D_==(res, 0.1973244, 1e-6))
    res = fisherExactTestPValueOnly(5, 15, 20, 20)
    assert(D_==(res, 0.0958044, 1e-6))
    res = fisherExactTestPValueOnly(5, 16, 20, 25)
    assert(D_==(res, 0.1725862, 1e-5))
    res = fisherExactTestPValueOnly(10, 5, 10, 1)
    assert(D_==(res, 0.1973244, 1e-6))
    res = fisherExactTestPValueOnly(5, 0, 1, 4)
    assert(D_==(res, 0.04761904, 1e-6))
    res = fisherExactTestPValueOnly(0, 1, 3, 2)
    assert(res == 1.0)
    res = fisherExactTestPValueOnly(0, 2, 6, 4)
    assert(D_==(res, 0.4545454545))
    res = fisherExactTestPValueOnly(2, 7, 8, 2)
    assert(D_==(res, 0.0230141, 1e-5))

    res = fisherExactTestPValueOnly(6, 37, 108, 200)
    assert(D_==(res, 0.005092697748126))
    res = fisherExactTestPValueOnly(22, 0, 0, 102)
    assert(D_==(res, 7.175066786244549e-25))
    res = fisherExactTestPValueOnly(94, 48, 3577, 16988)
    assert(D_==(res, 2.069356340993818e-37))
    res = fisherExactTestPValueOnly(5829225, 5692693, 5760959, 5760959)
    assert(res <= 1e-170)
    for ((a, b, c, d) <- Array((0, 0, 5, 10), (5, 10, 0, 0), (0, 5, 0, 10), (5, 0, 10, 0))) {
      assert(fisherExactTestPValueOnly(a, b, c, d) == 1.0)
    }
  }
}
