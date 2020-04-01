package is.hail

import is.hail.annotations.Region
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.ExecuteContext
import is.hail.utils.{ExecutionTimer, TempDir}
import is.hail.io.fs.FS
import org.apache.spark.SparkContext
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.BeforeClass

object HailSuite {
  def withSparkBackend(): HailContext = {
    val backend = SparkBackend(
      sc = new SparkContext(
        SparkBackend.createSparkConf(
          appName = "Hail.TestNG",
          master = System.getProperty("hail.master"),
          local = "local[2]",
          blockSize = 0)
          .set("spark.unsafe.exceptionOnMemoryLeak", "true")))
    HailContext(backend, logFile = "/tmp/hail.log")
  }

  lazy val hc: HailContext = {
    val hc = withSparkBackend()
    hc.flags.set("lower", "1")
    hc.checkRVDKeys = true
    hc
  }
}

class HailSuite extends TestNGSuite {
  def hc: HailContext = HailSuite.hc

  def sc: SparkContext = hc.sc

  @BeforeClass def ensureHailContextInitialized() { hc }

  val ctx = new ExecuteContext(hc.backend, hc.fs, Region(), new ExecutionTimer) // will get cleaned up on suite GC

  def fs: FS = hc.fs

  lazy val tmpDir: TempDir = TempDir(fs)
}
