package is.hail

import is.hail.utils.TempDir
import is.hail.variant.GenomeReference
import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.scalatest.testng.TestNGSuite

object SparkSuite {
  lazy val hc = HailContext(master = Option(System.getProperty("hail.master")),
    appName = "Hail.TestNG",
    local = "local[2]",
    quiet = true,
    minBlockSize = 0)
}

class SparkSuite extends TestNGSuite {

  lazy val hc: HailContext = SparkSuite.hc

  lazy val sc: SparkContext = hc.sc

  lazy val sqlContext: SQLContext = hc.sqlContext

  lazy val hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  lazy val tmpDir: TempDir = TempDir(hadoopConf)
}
