package org.broadinstitute.hail

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.hail.driver.Main._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{BeforeClass, AfterClass}

class SparkSuite(master: String = "local", verbose: Boolean = false) extends TestNGSuite {
  var sc: SparkContext = null
  var sqlContext: SQLContext = null

  @BeforeClass def startSpark() {
    val conf = new SparkConf().setMaster(master).setAppName("test")
    conf.set("spark.sql.parquet.compression.codec", "uncompressed")

    // FIXME KryoSerializer causes jacoco to throw IllegalClassFormatException exception
    // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    sc = new SparkContext(conf)
    val jar = getClass.getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    sc.addJar(jar)

    sqlContext = new org.apache.spark.sql.SQLContext(sc)

    if (!verbose) {
      Logger.getLogger("org").setLevel(Level.OFF)
      Logger.getLogger("akka").setLevel(Level.OFF)
    }
  }

  @AfterClass(alwaysRun=true) def stopSparkContext() {
    sc.stop()

    sc = null
    sqlContext = null
  }
}
