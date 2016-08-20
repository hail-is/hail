package org.broadinstitute.hail

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.broadinstitute.hail.driver.{HailConfiguration, SparkManager}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{AfterClass, BeforeClass}
import org.apache.hadoop

class SparkSuite extends TestNGSuite {
  var sc: SparkContext = _
  var sqlContext: SQLContext = _
  lazy val hadoopConf: hadoop.conf.Configuration =
    sc.hadoopConfiguration

  val noArgs: Array[String] = Array.empty[String]

  var _tmpDir: TempDir = _
  def tmpDir: TempDir = {
    if (_tmpDir == null)
      _tmpDir = TempDir("/tmp", hadoopConf)
    _tmpDir
  }

  @BeforeClass
  def startSpark() {
    val master = System.getProperty("hail.master", "local")
    sc = SparkManager.createSparkContext("Hail.TestNG", master)

    sqlContext = SparkManager.createSQLContext()

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val jar = getClass.getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    HailConfiguration.installDir = new File(jar).getParent + "/.."
    HailConfiguration.tmpDir = "/tmp"
  }

  @AfterClass(alwaysRun = true)
  def stopSparkContext() {
    sc = null
    sqlContext = null
  }
}
