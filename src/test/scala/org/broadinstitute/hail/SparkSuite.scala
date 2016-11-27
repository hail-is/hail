package org.broadinstitute.hail

import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.scalatest.testng.TestNGSuite
import org.apache.hadoop
import org.broadinstitute.hail.utils.TempDir
import org.broadinstitute.hail.driver.{configureAndCreateSparkContext, createSQLContext}

object SparkSuite {
  lazy val sc: SparkContext = configureAndCreateSparkContext("Hail.TestNG",
    Option(System.getProperty("hail.master")), local = "local[2]", quiet = true)

  lazy val sqlContext: SQLContext = createSQLContext(sc)
}

class SparkSuite extends TestNGSuite {
  def sc = SparkSuite.sc

  def sqlContext = SparkSuite.sqlContext

  lazy val hadoopConf: hadoop.conf.Configuration =
    sc.hadoopConfiguration

  val noArgs: Array[String] = Array.empty[String]

  lazy val tmpDir: TempDir = TempDir(hadoopConf)
}
