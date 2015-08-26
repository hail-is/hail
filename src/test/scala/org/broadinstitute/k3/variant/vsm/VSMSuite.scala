package org.broadinstitute.k3.variant.vsm

import org.broadinstitute.k3.variant.VariantSampleMatrix

import sys.process._
import scala.language.postfixOps

import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.k3.methods.{nSingletonPerSample, LoadVCF}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class VSMSuite extends TestNGSuite {
  val vsmTypes = List("managed", "sparky", "tuple")

  @Test def testsSingletonVariants() {
    val conf = new SparkConf().setMaster("local").setAppName("test")
    conf.set("spark.sql.parquet.compression.codec", "uncompressed")
    // FIXME KryoSerializer causes jacoco to throw IllegalClassFormatException exception
    // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

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

    sc.stop()
  }
}
