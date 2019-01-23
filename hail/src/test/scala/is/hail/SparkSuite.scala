package is.hail

import is.hail.utils.TempDir
import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.scalatest.testng.TestNGSuite

object SparkSuite {
  lazy val hc = HailContext(
    sc = new SparkContext(
      HailContext.createSparkConf(
        appName = "Hail.TestNG",
        master = Option(System.getProperty("hail.master")),
        local = "local[2]",
        blockSize = 0)
        .set("spark.unsafe.exceptionOnMemoryLeak", "true")),
    logFile = "/tmp/hail.log")
}

class SparkSuite extends TestNGSuite {
  lazy val hc: HailContext = {
    if (System.getenv("HAIL_ENABLE_CPP_CODEGEN") != null)
      SparkSuite.hc.flags.set("cpp", "1")
    SparkSuite.hc
  }

  lazy val sc: SparkContext = hc.sc

  def initializeHailContext(): Unit = { assert(hc != null) }

  lazy val sqlContext: SQLContext = hc.sqlContext

  lazy val hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  lazy val tmpDir: TempDir = TempDir(hadoopConf)
}
