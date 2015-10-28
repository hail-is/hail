package org.broadinstitute.k3.methods

import org.broadinstitute.k3.SparkSuite
import org.broadinstitute.k3.variant.Variant
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class LinearRegressionSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "sparky", "src/test/resources/linearRegression.vcf")
    val ped = Pedigree.read("src/test/resources/linearRegression.fam", vds.sampleIds)
    val cov = CovariateData.read("src/test/resources/linearRegression.cov", vds.sampleIds)

    val v1 = Variant("1", 1, "C", "T") // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T") // x = (2, ., 2, ., 0, 0)

    val linReg = LinearRegression(vds, ped, cov)
    val statsOfVariant: Map[Variant, LinRegStats] = linReg.lr.collect().toMap
    val eps = .001 //FIXME: upgrade to compare Double

    /* comparing to output of R code:
    y = c(1, 1, 2, 2, 2, 2)
    x = c(0, 1, 0, 0, 0, 1)
    c1 = c(0, 2, 1, -2, -2, 4)
    c2 = c(-1, 3, 5, 0, -4, 3)
    df = data.frame(y, x, c0, c1, c2)
    fit <- lm(y ~ x + c1 + c2, data=df)
    summary(fit)
    */

    assert(math.abs(statsOfVariant(v1).beta - -0.28589) < eps)
    assert(math.abs(statsOfVariant(v1).se   -  1.27392) < eps)
    assert(math.abs(statsOfVariant(v1).t    - -0.224  ) < eps)
    assert(math.abs(statsOfVariant(v1).p    -  0.8433 ) < eps)

    /* comparing to output of R code as above with:
    x = c(2, 1, 2, 1, 0, 0)
    */

    assert(math.abs(statsOfVariant(v2).beta - -0.5418) < eps)
    assert(math.abs(statsOfVariant(v2).se   -  0.3351) < eps)
    assert(math.abs(statsOfVariant(v2).t    - -1.617 ) < eps)
    assert(math.abs(statsOfVariant(v2).p    -  0.2473) < eps)

    //linReg.lr.collect().foreach{ case (v, lrs) => println(v + " " + lrs) }
    //val result = "rm -rf /tmp/linearRegression" !;
    //linReg.write("/tmp/linearRegression") //FIXME: How to test?
  }
}
