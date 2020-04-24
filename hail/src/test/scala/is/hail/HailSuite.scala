package is.hail

import is.hail.annotations.Region
import is.hail.backend.BroadcastValue
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.ExecuteContext
import is.hail.utils.ExecutionTimer
import is.hail.io.fs.FS
import org.apache.spark.SparkContext
import org.scalatest.testng.TestNGSuite
import org.testng.ITestContext
import org.testng.annotations.{AfterMethod, BeforeClass, BeforeMethod}

object HailSuite {
  def withSparkBackend(): HailContext = {
    val backend = SparkBackend(
      sc = new SparkContext(
        SparkBackend.createSparkConf(
          appName = "Hail.TestNG",
          master = System.getProperty("hail.master"),
          local = "local[2]",
          blockSize = 0)
          .set("spark.unsafe.exceptionOnMemoryLeak", "true")),
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp")
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

  @BeforeClass def ensureHailContextInitialized() { hc }

  def backend: SparkBackend = hc.sparkBackend("HailSuite.backend")

  def sc: SparkContext = backend.sc

  def fs: FS = backend.fs

  def fsBc: BroadcastValue[FS] = fs.broadcast

  var ctx: ExecuteContext = _

  @BeforeMethod
  def setupContext(context: ITestContext): Unit = {
    assert(ctx == null)
    ctx = new ExecuteContext(backend.tmpdir, backend.localTmpdir, backend, fs, Region(), new ExecutionTimer)
  }

  @AfterMethod
  def tearDownContext(context: ITestContext): Unit = {
    ctx.close()
    ctx = null
  }

  def withExecuteContext[T]()(f: ExecuteContext => T): T = {
    hc.sparkBackend("HailSuite.withExecuteContext").withExecuteContext()(f)
  }
}
