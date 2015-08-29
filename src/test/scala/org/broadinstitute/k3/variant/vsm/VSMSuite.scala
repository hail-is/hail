package org.broadinstitute.k3.variant.vsm

import org.broadinstitute.k3.SparkSuite
import org.broadinstitute.k3.variant.VariantSampleMatrix
import sys.process._
import scala.language.postfixOps
import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.k3.methods.{nSingletonPerSample, LoadVCF}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class VSMSuite extends SparkSuite {
  val vsmTypes = List("managed", "sparky", "tuple")

  @Test def testsSingletonVariants() {
    val singletons: List[Map[Int, Int]] =
      vsmTypes
      .map(vsmtype => {
        val vdsdir = "/tmp/sample." + vsmtype + ".vds"

        val result = "rm -rf " + vdsdir !;
        assert(result == 0)

        LoadVCF(sc, vsmtype, "src/test/resources/sample.vcf.gz")
        .write(sqlContext, vdsdir)

        val vds = VariantSampleMatrix.read(sqlContext, vsmtype, vdsdir)
        nSingletonPerSample(vds)
      })

    assert(singletons.tail.forall(s => s == singletons.head))
  }
}
