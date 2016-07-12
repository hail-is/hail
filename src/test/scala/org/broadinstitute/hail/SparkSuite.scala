package org.broadinstitute.hail

import java.io.File
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.hail.driver.HailConfiguration
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{BeforeClass, AfterClass}
import org.apache.hadoop

class SparkSuite extends TestNGSuite {
  var sc: SparkContext = null
  var sqlContext: SQLContext = null
  val noArgs: Array[String] = Array.empty[String]

  def hadoopConf: hadoop.conf.Configuration =
    sc.hadoopConfiguration

  var _tmpDir: TempDir = null
  def tmpDir: TempDir = {
    if (_tmpDir == null)
      _tmpDir = TempDir("/tmp", hadoopConf)
    _tmpDir
  }

  @BeforeClass
  def startSpark() {
    val conf = new SparkConf().setAppName("Hail.TestNG")

    val master = System.getProperty("hail.master")
    if (master != null)
      conf.setMaster(master)
    else if (!conf.contains("spark.master"))
      // conf.setMaster("local[*]")
      conf.setMaster("local")

    conf.set("spark.sql.parquet.compression.codec", "uncompressed")

    // FIXME KryoSerializer causes jacoco to throw IllegalClassFormatException exception
    // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    sc = new SparkContext(conf)
    sqlContext = new org.apache.spark.sql.SQLContext(sc)

    sc.hadoopConfiguration.set("io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec,org.seqdoop.hadoop_bam.util.BGZFCodec,org.apache.hadoop.io.compress.GzipCodec")

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val jar = getClass.getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    HailConfiguration.installDir = new File(jar).getParent + "/.."
    HailConfiguration.tmpDir = "/tmp"
  }

  @AfterClass(alwaysRun = true)
  def stopSparkContext() {
    sc.stop()

    sc = null
    sqlContext = null
  }
}
