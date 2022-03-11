package is.hail

import is.hail.asm4s.HailClassLoader
import is.hail.annotations.{Region, RegionPool}
import is.hail.backend.{BroadcastValue, ExecuteContext}
import is.hail.backend.spark.SparkBackend
import is.hail.utils.{ExecutionTimer, using}
import is.hail.io.fs.FS
import org.apache.spark.SparkContext
import org.scalatest.testng.TestNGSuite
import org.testng.ITestContext
import org.testng.annotations.{AfterMethod, BeforeClass, BeforeMethod}

object HailSuite {
  val theHailClassLoader = TestUtils.theHailClassLoader

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
  val theHailClassLoader = HailSuite.theHailClassLoader

  def hc: HailContext = HailSuite.hc

  @BeforeClass def ensureHailContextInitialized() { hc }

  def backend: SparkBackend = hc.sparkBackend("HailSuite.backend")

  def sc: SparkContext = backend.sc

  def fs: FS = backend.fs

  def fsBc: BroadcastValue[FS] = fs.broadcast

  var timer: ExecutionTimer = _

  var ctx: ExecuteContext = _

  var pool: RegionPool = _

  @BeforeMethod
  def setupContext(context: ITestContext): Unit = {
    assert(timer == null)
    timer = new ExecutionTimer("HailSuite")
    assert(ctx == null)
    pool = RegionPool()
    ctx = new ExecuteContext(backend.tmpdir, backend.localTmpdir, backend, fs, Region(pool=pool), timer, null, HailSuite.theHailClassLoader)
  }

  @AfterMethod
  def tearDownContext(context: ITestContext): Unit = {
    ctx.close()
    ctx = null
    timer.finish()
    timer = null
    pool.close()

    if (backend.sc.isStopped)
      throw new RuntimeException(s"method stopped spark context!")
  }

  def withExecuteContext[T]()(f: ExecuteContext => T): T = {
    ExecutionTimer.logTime("HailSuite.withExecuteContext") { timer =>
      hc.sparkBackend("HailSuite.withExecuteContext").withExecuteContext(timer)(f)
    }
  }
}
