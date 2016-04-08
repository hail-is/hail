package org.broadinstitute.hail.driver.example

import breeze.linalg.{DenseMatrix, DenseVector}
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{AnnotateSamples, ImportVCF, State}
import org.testng.annotations.Test

import scala.collection.mutable

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

    val qCov1 = s.vds.querySA("cov", "Cov1")
    val qCov2 = s.vds.querySA("cov", "Cov2")
    val qPheno = s.vds.querySA("pheno", "Pheno")

    val cov1 = s.vds.sampleAnnotations.map(qCov1)
    val cov2 = s.vds.sampleAnnotations.map(qCov1)
    val pheno = s.vds.sampleAnnotations.map(qPheno)

    val sampleMask = Range(0, s.vds.nSamples).map(s => cov1(s).isDefined && cov2(s).isDefined && pheno(s).isDefined)

    val filtVds = s.vds.filterSamples((s, sa) => sampleMask(s))

    val cov1Data = cov1.zipWithIndex.filter(x => sampleMask(x._2)).map(_._1.get.asInstanceOf[Double])
    val cov2Data = cov1.zipWithIndex.filter(x => sampleMask(x._2)).map(_._1.get.asInstanceOf[Double])
    val covArray = (cov1Data ++ cov2Data).toArray

    val yArray = pheno.zipWithIndex.filter(x => sampleMask(x._2)).map(_._1.get.asInstanceOf[Double]).toArray

    val n = yArray.size

    val y: DenseVector[Double] = DenseVector(yArray)
    val cov: DenseMatrix[Double] = new DenseMatrix(n, 2, covArray)

    println(y)
    println(cov)

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
