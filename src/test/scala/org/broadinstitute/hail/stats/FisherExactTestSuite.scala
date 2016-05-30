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
    val result = fet.result().map{_.getOrElse(Double.NaN)}

    assert(math.abs(result(0) - 0.2828) < 1e-4)
    assert(math.abs(result(1) - 0.4754059) < 1e-4)
    assert(math.abs(result(2) - 0.122593) < 1e-4)
    assert(math.abs(result(3) - 1.597972) < 1e-4)
  }

  object Spec extends Properties("FisherExactTest") {
    val twoBytwoMatrix = for (n: Int <- choose(10, 500); k: Int <- choose(1, n - 1); x: Int <- choose(1, n - 1)) yield (k, n-k, x, n-x)

    property("import generates same output as export") =
      forAll(twoBytwoMatrix) { case (t) =>
        val fet = FisherExactTest(t)

        val rResultTwoSided = s"Rscript src/test/resources/fisherExactTest.r two.sided ${t._1} ${t._2} ${t._3} ${t._4}" !!

        val rResultLess = s"Rscript src/test/resources/fisherExactTest.r less ${t._1} ${t._2} ${t._3} ${t._4}" !!

        val rResultGreater = s"Rscript src/test/resources/fisherExactTest.r greater ${t._1} ${t._2} ${t._3} ${t._4}" !!


        val rTwoSided = rResultTwoSided.split(" ").take(4)
          .map{ case s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble}
        val rLess = rResultLess.split(" ").take(4)
          .map{ case s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble}
        val rGreater = rResultGreater.split(" ").take(4)
          .map{ case s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble}

        val hailTwoSided = fet.result(alternative = "two.sided")
        val hailLess = fet.result(alternative = "less")
        val hailGreater = fet.result(alternative = "greater")

        val hailResults = Array(hailTwoSided, hailLess, hailGreater).map{_.map{_.getOrElse(Double.NaN)}}
        val rResults = Array(rTwoSided, rLess, rGreater)

        hailResults.zip(rResults).forall{case (h, r) =>
          val res = D_==(h(0), r(0)) &&
            D_==(h(1), h(1), 1e-6) &&
            D_==(h(2), r(2), 1e-6) &&
            (h(3) == Double.PositiveInfinity && r(3) == Double.PositiveInfinity) || D_==(h(3), r(3), 1e-6)
          if (!res) {
            println(h(0), r(0), D_==(h(0), r(0)))
            println(h(1), r(1), D_==(h(1), h(1), 1e-6))
            println(h(2), r(2), D_==(h(2), r(2), 1e-6))
            println(h(3), r(3), (h(3) == Double.PositiveInfinity && r(3) == Double.PositiveInfinity) || D_==(h(3), r(3), 1e-6))
          }
          res
        }
      }
  }

//  @Test def testFisherExactTest() {
//    Spec.check()
//  }
}
