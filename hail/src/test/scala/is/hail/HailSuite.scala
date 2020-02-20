package is.hail

import is.hail.annotations.Region
import is.hail.expr.ir.ExecuteContext
import is.hail.utils.{ExecutionTimer, TempDir}
import is.hail.io.fs.FS
import org.apache.spark.SparkContext
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.BeforeClass

object HailSuite {
  def withSparkBackend(): HailContext =
    HailContext(
      sc = new SparkContext(
        HailContext.createSparkConf(
          appName = "Hail.TestNG",
          master = Option(System.getProperty("hail.master")),
          local = "local[2]",
          blockSize = 0)
          .set("spark.unsafe.exceptionOnMemoryLeak", "true")),
      logFile = "/tmp/hail.log")


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

  val ctx = ExecuteContext(Region(), new ExecutionTimer) // will get cleaned up on suite GC

  def sFS: FS = hc.sFS

  lazy val tmpDir: TempDir = TempDir(sFS)
}
