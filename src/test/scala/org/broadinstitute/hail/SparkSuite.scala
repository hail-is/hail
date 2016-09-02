package org.broadinstitute.hail

import java.io.File

import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.broadinstitute.hail.driver.{HailConfiguration, SparkManager}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{AfterClass, BeforeClass, Test}
import org.apache.hadoop
import org.broadinstitute.hail.check.{Parameters, Prop}

import scala.collection.mutable.ArrayBuffer

trait SparkSuite extends TestNGSuite {
  lazy val sc: SparkContext = {
    val jar = getClass.getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    HailConfiguration.installDir = new File(jar).getParent + "/.."
    HailConfiguration.tmpDir = "/tmp"

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val master = System.getProperty("hail.master")
    SparkManager.createSparkContext("Hail.TestNG", Option(master), "local")
  }

  lazy val sqlContext: SQLContext = {
    // force
    sc

    SparkManager.createSQLContext()
  }

  lazy val hadoopConf: hadoop.conf.Configuration =
    sc.hadoopConfiguration

  val noArgs: Array[String] = Array.empty[String]

  var _tmpDir: TempDir = _

  def tmpDir: TempDir = {
    if (_tmpDir == null)
      _tmpDir = TempDir("/tmp", hadoopConf)
    _tmpDir
  }
}

trait PropertySuite extends SparkSuite {
  val properties = ArrayBuffer.empty[(String, Prop)]

  class PropertySpecifier {
    def update(propName: String, prop: Prop) {
      properties += propName -> prop
    }
  }

  lazy val property = new PropertySpecifier

  @Test def test() {
    val size = System.getProperty("check.size", "1000").toInt
    val count = System.getProperty("check.count", "10").toInt

    println(s"check: size = $size, count = $count")

    val rng = new RandomDataGenerator()
    rng.reSeed(Prop.seed)
    val p = Parameters(rng, size, count)

    properties.foreach { case (name, prop) =>
      prop(p, Some(name))
    }
  }
}
