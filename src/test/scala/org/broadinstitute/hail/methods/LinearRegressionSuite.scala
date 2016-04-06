package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{SplitMulti, LinearRegressionCommand, State}
import org.broadinstitute.hail.variant.Variant
import org.broadinstitute.hail.Utils._
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

    val linReg = LinearRegression(vds, ped, cov.filterSamples(ped.phenotypedSamples))

    val statsOfVariant: Map[Variant, Option[LinRegStats]] = linReg.rdd.collect().toMap

    //linReg.lr.collect().foreach{ case (v, lrs) => println(v + " " + lrs) }

    /* comparing to output of R code:
    y = c(1, 1, 2, 2, 2, 2)
    x = c(0, 1, 0, 0, 0, 1)
    c1 = c(0, 2, 1, -2, -2, 4)
    c2 = c(-1, 3, 5, 0, -4, 3)
    df = data.frame(y, x, c1, c2)
    fit <- lm(y ~ x + c1 + c2, data=df)
    summary(fit)["coefficients"]

    */

    assert(D_==(statsOfVariant(v1).get.beta, -0.28589, .001))
    assert(D_==(statsOfVariant(v1).get.se  ,  1.27392, .001))
    assert(D_==(statsOfVariant(v1).get.t   , -0.224  , .01)) // FIXME: add precision
    assert(D_==(statsOfVariant(v1).get.p   ,  0.8433 , .001))

    /* comparing to output of R code as above with:
    x = c(2, 1, 2, 1, 0, 0)
    */

    assert(D_==(statsOfVariant(v2).get.beta, -0.5418, .001))
    assert(D_==(statsOfVariant(v2).get.se  ,  0.3351, .001))
    assert(D_==(statsOfVariant(v2).get.t   , -1.617 , .001))
    assert(D_==(statsOfVariant(v2).get.p   ,  0.2473, .001))

    assert(statsOfVariant(v6).isEmpty)
    assert(statsOfVariant(v7).isEmpty)
    assert(statsOfVariant(v8).isEmpty)
    assert(statsOfVariant(v9).isEmpty)
    assert(statsOfVariant(v10).isEmpty)

    var s = State(sc, sqlContext, vds)
    s = SplitMulti.run(s)
    s = LinearRegressionCommand.run(s,
      Array("-f", "src/test/resources/linearRegression.fam",
      "-c", "src/test/resources/linearRegression.cov"))

    val query1 = s.vds.queryVA("linreg", "beta")
    val query2 = s.vds.queryVA("linreg", "stderr")
    val query3 = s.vds.queryVA("linreg", "tstat")
    val query4 = s.vds.queryVA("linreg", "pval")

    val annotationMap = s.vds.variantsAndAnnotations
    .collect()
    .toMap


    assert(D_==(query1(annotationMap(v1)).get.asInstanceOf[Double], -0.28589, .001))
    assert(D_==(query2(annotationMap(v1)).get.asInstanceOf[Double],  1.27392, .001))
    assert(D_==(query3(annotationMap(v1)).get.asInstanceOf[Double], -0.224  , .01)) // FIXME: add precision
    assert(D_==(query4(annotationMap(v1)).get.asInstanceOf[Double],  0.8433 , .001))

    assert(query1(annotationMap(v6)).isEmpty)


    //val result = "rm -rf /tmp/linearRegression" !;
    linReg.write("/tmp/linearRegression") //FIXME: How to test?
  }
}
