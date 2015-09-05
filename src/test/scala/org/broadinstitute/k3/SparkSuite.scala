package org.broadinstitute.k3

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.k3.driver.Main._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{BeforeClass, AfterClass}

class SparkSuite extends TestNGSuite {
  var sc: SparkContext = null
  var sqlContext: SQLContext = null

  @BeforeClass def startSpark() {
    val conf = new SparkConf().setMaster("local").setAppName("test")
    conf.set("spark.sql.parquet.compression.codec", "uncompressed")

    // FIXME KryoSerializer causes jacoco to throw IllegalClassFormatException exception
    // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    sc = new SparkContext(conf)
    val jar = getClass.getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    sc.addJar(jar)

    sqlContext = new org.apache.spark.sql.SQLContext(sc)
  }

  @AfterClass(alwaysRun=true) def stopSparkContext() {
    sc.stop()

    sc = null
    sqlContext = null
  }
}
