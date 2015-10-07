package org.broadinstitute.k3.methods

import org.broadinstitute.k3.SparkSuite
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class LinearRegressionSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "sparky", "src/test/resources/linearRegression.vcf")
    val ped = Pedigree.read("src/test/resources/linearRegression.fam", vds.sampleIds)
    val cov = CovariateData.read("src/test/resources/linearRegression.cov", vds.sampleIds)

    println(cov.covIds.mkString(" "))
    println(cov.rowIds.mkString(" "))
    println(cov.data)

    val linReg = LinearRegression(vds, ped, cov)
    linReg.betas.collect().foreach{ case (v, b) => println(v + " " + b) }

    //val result = "rm -rf /tmp/linearRegression" !;
    //linReg.write("/tmp/linearRegression")
  }
}
