package org.broadinstitute.hail.methods

import breeze.linalg.DenseMatrix
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class CovariateDataSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/covariateData.vcf")
    val cov = CovariateData.read("src/test/resources/covariateData.cov", sc.hadoopConfiguration, vds.sampleIds)

    val covRowName = Array(0,1,2,4,5,6,7)
    val covName = Array("Cov1", "Cov2")
    val data = DenseMatrix.create[Double](7, 2,
      Array(0.0, 2.0, 1.0, -2.0, -2.0, 4.0, 0.0,
           -1.0, 3.0, 5.0,  0.0, -4.0, 3.0, 0.0))

    assert(cov.covRowSample sameElements covRowName)
    assert(cov.covName sameElements covName)
    assert(cov.data == data)
  }
}
