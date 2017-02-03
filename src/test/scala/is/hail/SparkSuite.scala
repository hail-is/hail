package is.hail

import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.scalatest.testng.TestNGSuite
import org.apache.hadoop
import is.hail.utils.TempDir
import is.hail.driver._

object SparkSuite {
  lazy val hc = HailContext(master = Option(System.getProperty("hail.master")),
    appName = "Hail.TestNG",
    local = "local[2]",
    quiet = true)
}

class SparkSuite extends TestNGSuite {

  lazy val hc: HailContext = SparkSuite.hc

  lazy val sc: SparkContext = hc.sc

  lazy val sqlContext: SQLContext = hc.sqlContext

  lazy val hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  lazy val tmpDir: TempDir = TempDir(hadoopConf)
}
