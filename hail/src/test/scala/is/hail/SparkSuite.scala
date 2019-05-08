package is.hail

import is.hail.utils.TempDir
import is.hail.io.fs.FS
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.BeforeClass

object SparkSuite {
  lazy val hc: HailContext = {
    val hc = HailContext(
      sc = new SparkContext(
        HailContext.createSparkConf(
          appName = "Hail.TestNG",
          master = Option(System.getProperty("hail.master")),
          local = "local[2]",
          blockSize = 0)
          .set("spark.unsafe.exceptionOnMemoryLeak", "true")),
      logFile = "/tmp/hail.log")
    if (System.getenv("HAIL_ENABLE_CPP_CODEGEN") != null)
      hc.flags.set("cpp", "1")
    hc.checkRVDKeys = true
    hc
  }
}

class SparkSuite extends TestNGSuite {
  def hc: HailContext = SparkSuite.hc

  def sc: SparkContext = hc.sc

  @BeforeClass def ensureHailContextInitialized() {
    hc
  }

  def sqlContext: SQLContext = hc.sqlContext

  def fs: FS = hc.sFS

  lazy val tmpDir: TempDir = TempDir(fs)
}
