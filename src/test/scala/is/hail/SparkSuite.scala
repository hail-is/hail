package is.hail

import is.hail.utils.TempDir
import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.scalatest.testng.TestNGSuite

object SparkSuite {
  lazy val hc: HailContext = {
    var master: String = System.getProperty("hail.master")
    if (master == null)
      master = "local[2]"

    HailContext(master = Option(master),
      appName = "Hail.TestNG",
      quiet = true,
      minBlockSize = 0)
  }
}

class SparkSuite extends TestNGSuite {

  lazy val hc: HailContext = SparkSuite.hc

  lazy val sc: SparkContext = hc.sc

  lazy val sqlContext: SQLContext = hc.sqlContext

  lazy val hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  lazy val tmpDir: TempDir = TempDir(hadoopConf)
}
