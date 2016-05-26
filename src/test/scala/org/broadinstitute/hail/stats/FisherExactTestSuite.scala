package org.broadinstitute.hail.stats

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import scala.language.postfixOps
import scala.sys.process._
import org.broadinstitute.hail.Utils.D_==
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

    val fet = new FisherExactTest(a, b, c, d)
    val result = fet.calcPvalue()
    assert(math.abs(result - 0.2828) < 1e-4)
  }

  object Spec extends Properties("FisherExactTest") {
    val twoBytwoMatrix = for (n: Int <- choose(10, 500); k: Int <- choose(1, n - 1); x: Int <- choose(1, n - 1)) yield (k, n-k, x, n-x)

    property("import generates same output as export") =
      forAll(twoBytwoMatrix) { case (t) =>
        val fet = FisherExactTest(t)

        val rResultTwoSided = s"Rscript src/test/resources/fisherExactTest.r two.sided ${t._1} ${t._2} ${t._3} ${t._4}" !!

        val rResultLess = s"Rscript src/test/resources/fisherExactTest.r less ${t._1} ${t._2} ${t._3} ${t._4}" !!

        val rResultGreater = s"Rscript src/test/resources/fisherExactTest.r greater ${t._1} ${t._2} ${t._3} ${t._4}" !!

        val hailResultTwoSided = fet.calcPvalue(alternative = "two.sided")
        val hailResultLess = fet.calcPvalue(alternative = "less")
        val hailResultGreater = fet.calcPvalue(alternative = "greater")

        D_==(rResultTwoSided.toDouble, hailResultTwoSided) &&
          D_==(rResultLess.toDouble, hailResultLess) &&
          D_==(rResultGreater.toDouble, hailResultGreater)

      }
  }

  @Test def testFisherExactTest() {
    Spec.check()
  }

}
