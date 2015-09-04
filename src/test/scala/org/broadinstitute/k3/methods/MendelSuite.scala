package org.broadinstitute.k3.methods

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class MendelSuite extends TestNGSuite {
  val conf = new SparkConf().setMaster("local").setAppName("test")
  conf.set("spark.sql.parquet.compression.codec", "uncompressed")
  // FIXME KryoSerializer causes jacoco to throw IllegalClassFormatException exception
  // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  val sc = new SparkContext(conf)

  @Test def test() {
    val ped = Pedigree.read("src/test/resources/sample_mendel.fam")
    val vds = LoadVCF(sc, "sparky", "src/test/resources/sample_mendel.vcf")

    println(ped)

    sc.stop()
  }
}
