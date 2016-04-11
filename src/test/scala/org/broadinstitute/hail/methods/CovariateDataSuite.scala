package org.broadinstitute.hail.methods

import breeze.linalg.DenseMatrix
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class CovariateDataSuite extends SparkSuite {
  @Test def test() {
    def assertCov(cov1: CovariateData, cov2: CovariateData) {
      assert(cov1.covRowSample sameElements cov2.covRowSample)
      assert(cov1.covName sameElements cov2.covName)
      assert(cov1.data == cov2.data)
    }

    val vds = LoadVCF(sc, "src/test/resources/covariateData.vcf")
    val cov = CovariateData.read("src/test/resources/covariateData.cov", sc.hadoopConfiguration, vds.sampleIds)

    val covRowSample = Array("A", "B", "C", "D", "E", "F", "X")
    val covName = Array("Cov1", "Cov2")
    val data = Some(DenseMatrix.create[Double](7, 2,
      Array(0.0, 2.0, 1.0, -2.0, -2.0, 4.0, 0.0,
           -1.0, 3.0, 5.0,  0.0, -4.0, 3.0, 0.0)))
    assertCov(cov, CovariateData(covRowSample, covName, data))

    val covS = cov.filterSamples(Set("A", "D", "E"))
    val dataS = Some(DenseMatrix.create[Double](3, 2, Array(0.0, -2.0, -2.0,
                                                           -1.0,  0.0, -4.0)))
    assertCov(covS, CovariateData(Array("A", "D", "E"), covName, dataS))

    val cov1 = cov.filterCovariates(Set("Cov1"))
    val data1 = Some(DenseMatrix.create[Double](7, 1, Array(0.0, 2.0, 1.0, -2.0, -2.0, 4.0, 0.0)))
    assertCov(cov1, CovariateData(covRowSample, Array("Cov1"), data1))

    val covNoSamples = cov.filterSamples(Set[String]())
    assertCov(covNoSamples, CovariateData(Array[String](), covName, None))

    val covNoCovs = cov.filterCovariates(Set[String]())
    assertCov(covNoCovs, CovariateData(covRowSample, Array[String](), None))

    val cov2 = cov.filterCovariates(Set("Cov2"))
    assertCov(cov, cov1.appendCovariates(cov2))
    assertCov(cov, cov.appendCovariates(covNoCovs))
    assertCov(cov, covNoCovs.appendCovariates(cov))
    assertCov(covNoCovs, covNoCovs.appendCovariates(covNoCovs))
  }
}