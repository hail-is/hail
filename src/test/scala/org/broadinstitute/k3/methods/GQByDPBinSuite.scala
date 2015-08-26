package org.broadinstitute.k3.methods

import org.apache.spark.{SparkContext, SparkConf}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class GQByDPBinSuite extends TestNGSuite {
  val conf = new SparkConf().setMaster("local").setAppName("test")
  conf.set("spark.sql.parquet.compression.codec", "uncompressed")
  // FIXME KryoSerializer causes jacoco to throw IllegalClassFormatException exception
  // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  val sc = new SparkContext(conf)

  @Test def test() {
    val vds = LoadVCF(sc, "sparky", "src/test/resources/gqbydp_test.vcf")
    val gqbydp = GQByDPBins(vds)
    assert(gqbydp == Map((0, 5) -> 0.5, (1, 2) -> 0.0))

    sc.stop()
  }
}
