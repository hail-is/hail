package is.hail

import is.hail.annotations.Region
import is.hail.expr.ir.ExecuteContext
import is.hail.utils.TempDir
import is.hail.io.fs.FS
import org.apache.spark.SparkContext
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.BeforeClass

object HailSuite {
  def withDistributedBackend(host: String): HailContext =
    HailContext.createDistributed(host, logFile = "/tmp/hail.log")

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
    val schedulerHost = System.getenv("HAIL_TEST_SCHEDULER_HOST")
    val hc = if (schedulerHost == null)
      withSparkBackend()
    else
      withDistributedBackend(schedulerHost)
    if (System.getenv("HAIL_ENABLE_CPP_CODEGEN") != null)
      hc.flags.set("cpp", "1")
    hc.flags.set("lower", "1")
    hc.checkRVDKeys = true
    hc
  }
}

class HailSuite extends TestNGSuite {
  def hc: HailContext = HailSuite.hc

  def sc: SparkContext = hc.sc

  @BeforeClass def ensureHailContextInitialized() { hc }

  val ctx = ExecuteContext(Region()) // will get cleaned up on suite GC

  def sFS: FS = hc.sFS

  lazy val tmpDir: TempDir = TempDir(sFS)
}
