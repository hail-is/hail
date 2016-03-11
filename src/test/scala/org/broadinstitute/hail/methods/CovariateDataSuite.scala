package org.broadinstitute.hail.methods

import breeze.linalg.DenseMatrix
import org.broadinstitute.hail.{methods, SparkSuite}
import org.testng.annotations.Test

class CovariateDataSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/covariateData.vcf")
    val cov = CovariateData.read("src/test/resources/covariateData.cov", sc.hadoopConfiguration, vds.sampleIds)

    val covRowSample = Array(0,1,2,4,5,6,7)
    val covName = Array("Cov1", "Cov2")
    val data = Some(DenseMatrix.create[Double](7, 2,
      Array(0.0, 2.0, 1.0, -2.0, -2.0, 4.0, 0.0,
           -1.0, 3.0, 5.0,  0.0, -4.0, 3.0, 0.0)))

    assert(cov.covRowSample sameElements covRowSample)
    assert(cov.covName sameElements covName)
    assert(cov.data == data)

    val covS = cov.filterSamples(Set(0,4,5))
    assert(covS.covRowSample sameElements Array(0,4,5))
    assert(covS.covName sameElements covName)
    assert(covS.data == Some(DenseMatrix.create[Double](3, 2, Array(0.0, -2.0, -2.0,
                                                                   -1.0,  0.0, -4.0))))

    val covC = cov.filterCovariates(Set("Cov1"))
    assert(covC.covRowSample sameElements covRowSample)
    assert(covC.covName sameElements Array("Cov1"))
    assert(covC.data == Some(DenseMatrix.create[Double](7, 1, Array(0.0, 2.0, 1.0, -2.0, -2.0, 4.0, 0.0))))

    val covNoSamples = cov.filterSamples(Set[Int]())
    assert(covNoSamples.covRowSample.isEmpty)
    assert(covNoSamples.covName sameElements covName)
    assert(covNoSamples.data.isEmpty)

    val covNoCovs = cov.filterCovariates(Set[String]())
    assert(covNoCovs.covRowSample sameElements covRowSample)
    assert(covNoCovs.covName.isEmpty)
    assert(covNoCovs.data.isEmpty)

  }
}
