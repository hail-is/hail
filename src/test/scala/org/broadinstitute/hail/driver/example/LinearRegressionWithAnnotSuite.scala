package org.broadinstitute.hail.driver.example

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{AnnotateSamples, ImportVCF, State}
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

    s.vds.metadata.sampleAnnotations.foreach(println)

    val qCov = s.vds.querySA("cov", "Cov1")
    val qPheno = s.vds.querySA("pheno", "Pheno")

    /*
    s = CaseControlCount.run(s, Array[String]())

    val qCase = s.vds.queryVA("nCase")
    val qControl = s.vds.queryVA("nControl")

    val r = s.vds.mapWithAll { case (v, va, s, g) =>
      (v.start, (qCase(va).get.asInstanceOf[Int],
        qControl(va).get.asInstanceOf[Int]))
    }.collectAsMap()

    assert(r == Map(1 ->(1, 0),
      2 ->(0, 2)))
    */
  }
}
