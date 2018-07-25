package is.hail.stats

import is.hail.SparkSuite
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.{MatrixTable, _}
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class FisherExactTestSuite extends SparkSuite {

  @Test def testPvalue() {
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
