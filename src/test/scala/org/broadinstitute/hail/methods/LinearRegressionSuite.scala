package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.variant.Variant
import org.testng.annotations.Test

//import scala.language.postfixOps

class LinearRegressionSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/linearRegression.vcf")
    val ped = Pedigree.read("src/test/resources/linearRegression.fam", sc.hadoopConfiguration, vds.sampleIds)
    val cov = CovariateData.read("src/test/resources/linearRegression.cov", sc.hadoopConfiguration, vds.sampleIds)
    
    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (2, ., 2, ., 0, 0)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val linReg = LinearRegression(vds, ped, cov)
    val statsOfVariant: Map[Variant, Option[LinRegStats]] = linReg.lr.collect().toMap
    val eps = .001 //FIXME: use D_== when it is ready


    //linReg.lr.collect().foreach{ case (v, lrs) => println(v + " " + lrs) }

    /* comparing to output of R code:
    y = c(1, 1, 2, 2, 2, 2)
    x = c(0, 1, 0, 0, 0, 1)
    c1 = c(0, 2, 1, -2, -2, 4)
    c2 = c(-1, 3, 5, 0, -4, 3)
    df = data.frame(y, x, c0, c1, c2)
    fit <- lm(y ~ x + c1 + c2, data=df)
    summary(fit)
    */

    assert(math.abs(statsOfVariant(v1).get.beta - -0.28589) < eps)
    assert(math.abs(statsOfVariant(v1).get.se   -  1.27392) < eps)
    assert(math.abs(statsOfVariant(v1).get.t    - -0.224  ) < eps)
    assert(math.abs(statsOfVariant(v1).get.p    -  0.8433 ) < eps)

    /* comparing to output of R code as above with:
    x = c(2, 1, 2, 1, 0, 0)
    */

    assert(math.abs(statsOfVariant(v2).get.beta - -0.5418) < eps)
    assert(math.abs(statsOfVariant(v2).get.se   -  0.3351) < eps)
    assert(math.abs(statsOfVariant(v2).get.t    - -1.617 ) < eps)
    assert(math.abs(statsOfVariant(v2).get.p    -  0.2473) < eps)

    assert(statsOfVariant(v6).isEmpty)
    assert(statsOfVariant(v7).isEmpty)
    assert(statsOfVariant(v8).isEmpty)
    assert(statsOfVariant(v9).isEmpty)
    assert(statsOfVariant(v10).isEmpty)

    //val result = "rm -rf /tmp/linearRegression" !;
    linReg.write("/tmp/linearRegression") //FIXME: How to test?
  }
}
