package org.broadinstitute.hail.driver.example

import breeze.linalg.{DenseMatrix, DenseVector}
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.methods.{LinRegStats, LinearRegression}
import org.broadinstitute.hail.variant.Variant
import org.testng.annotations.Test

class LinearRegressionWithAnnotSuite extends SparkSuite {
  @Test def test() {

    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/linearRegression.vcf"))

    s = AnnotateSamples.run(s, Array("tsv",
      "-i", "src/test/resources/linearRegression.cov",
      "--root", "sa.cov",
      "--types", "Cov1: Double, Cov2: Double"))

    s = AnnotateSamples.run(s, Array("tsv",
      "-i", "src/test/resources/linearRegression.pheno",
      "--root", "sa.pheno",
      "--types", "Pheno: Double",
      "--missing", "0"))

    //s.vds.metadata.sampleAnnotations.foreach(println)

    val qCov1 = s.vds.querySA("cov", "Cov1")
    val qCov2 = s.vds.querySA("cov", "Cov2")
    val qPheno = s.vds.querySA("pheno", "Pheno")

    val cov1 = s.vds.sampleAnnotations.map(qCov1)
    val cov2 = s.vds.sampleAnnotations.map(qCov2)
    val pheno = s.vds.sampleAnnotations.map(qPheno)

    // FIXME: could be function instead
    val sampleMask = Range(0, s.vds.nSamples).map(s => cov1(s).isDefined && cov2(s).isDefined && pheno(s).isDefined)

    val cov1Data = cov1.zipWithIndex.filter(x => sampleMask(x._2)).map(_._1.get.asInstanceOf[Double])
    val cov2Data = cov2.zipWithIndex.filter(x => sampleMask(x._2)).map(_._1.get.asInstanceOf[Double])
    val covArray = (cov1Data ++ cov2Data).toArray

    val yArray = pheno.zipWithIndex.filter(x => sampleMask(x._2)).map(_._1.get.asInstanceOf[Double]).toArray

    val n = yArray.size

    val y: DenseVector[Double] = DenseVector(yArray)
    val cov: Option[DenseMatrix[Double]] = Some(new DenseMatrix(n, 2, covArray))

    s = s.copy(vds = s.vds.filterSamples((s, sa) => sampleMask(s)))

//    println(y)
//    println(cov)
//    s.vds.rdd.foreach(println)

    val linreg = LinearRegression(s.vds, y, cov)

    linreg.rdd.collect().foreach{ case (v, lrs) => println(v + " " + lrs) }

    val statsOfVariant: Map[Variant, Option[LinRegStats]] = linreg.rdd.collect().toMap

    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (., 2, ., 2, 0, 0)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    /* comparing to output of R code:
    y = c(1, 1, 2, 2, 2, 2)
    x = c(0, 1, 0, 0, 0, 1)
    c1 = c(0, 2, 1, -2, -2, 4)
    c2 = c(-1, 3, 5, 0, -4, 3)
    df = data.frame(y, x, c1, c2)
    fit <- lm(y ~ x + c1 + c2, data=df)
    summary(fit)
    */

    assert(D_==(statsOfVariant(v1).get.beta, -0.28589421))
    assert(D_==(statsOfVariant(v1).get.se,    1.2739153))
    assert(D_==(statsOfVariant(v1).get.t,    -0.22442167))
    assert(D_==(statsOfVariant(v1).get.p,     0.84327106))

    /* v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assert(D_==(statsOfVariant(v2).get.beta, -0.5417647))
    assert(D_==(statsOfVariant(v2).get.se,    0.3350599))
    assert(D_==(statsOfVariant(v2).get.t,    -1.616919))
    assert(D_==(statsOfVariant(v2).get.p,     0.24728705))

    assert(statsOfVariant(v6).isEmpty)
    assert(statsOfVariant(v7).isEmpty)
    assert(statsOfVariant(v8).isEmpty)
    assert(statsOfVariant(v9).isEmpty)
    assert(statsOfVariant(v10).isEmpty)


    val (newVAS, inserter) = s.vds.insertVA(LinRegStats.`type`, "linreg")

    s.copy(
      vds = s.vds.copy(
        rdd = s.vds.rdd.zipPartitions(linreg.rdd) { case (it, jt) =>
          it.zip(jt).map { case ((v, va, gs), (v2, comb)) =>
            assert(v == v2)
            (v, inserter(va, comb.map(_.toAnnotation)), gs)
          }

        }, vaSignature = newVAS))

    val qBeta = s.vds.queryVA("linreg", "beta")
    val qSe = s.vds.queryVA("linreg", "se")
    val qTstat = s.vds.queryVA("linreg", "tstat")
    val qPval = s.vds.queryVA("linreg", "pval")

    val annotationMap = s.vds.variantsAndAnnotations
      .collect()
      .toMap

    assert(D_==(qBeta(annotationMap(v1)).get.asInstanceOf[Double], -0.28589421))
    assert(D_==(qSe(annotationMap(v1)).get.asInstanceOf[Double],  1.2739153))
    assert(D_==(qTstat(annotationMap(v1)).get.asInstanceOf[Double], -0.22442167))
    assert(D_==(qPval(annotationMap(v1)).get.asInstanceOf[Double],  0.84327106))

    assert(D_==(qBeta(annotationMap(v2)).get.asInstanceOf[Double], -0.5417647))
    assert(D_==(qSe(annotationMap(v2)).get.asInstanceOf[Double],  0.3350599))
    assert(D_==(qTstat(annotationMap(v2)).get.asInstanceOf[Double], -1.616919))
    assert(D_==(qPval(annotationMap(v2)).get.asInstanceOf[Double],  0.24728705))

    assert(qBeta(annotationMap(v6)).isEmpty)
    assert(qBeta(annotationMap(v7)).isEmpty)
    assert(qBeta(annotationMap(v8)).isEmpty)
    assert(qBeta(annotationMap(v9)).isEmpty)
    assert(qBeta(annotationMap(v10)).isEmpty)

    //val result = "rm -rf /tmp/linearRegression" !;
    linreg.write("/tmp/linearRegression.linreg") //FIXME: How to test?


  }
}
