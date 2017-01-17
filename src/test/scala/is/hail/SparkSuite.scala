package is.hail

import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.scalatest.testng.TestNGSuite
import org.apache.hadoop
import is.hail.utils.TempDir
import is.hail.driver._

object SparkSuite {
  lazy val sc: SparkContext = {
    configureHail()
    configureLogging(quiet = true)
    configureAndCreateSparkContext("Hail.TestNG",
      Option(System.getProperty("hail.master")), local = "local[2]")
  }

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
