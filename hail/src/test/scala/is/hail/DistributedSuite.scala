package is.hail

import is.hail.backend.distributed.DistributedBackend
import is.hail.scheduler.Executor
import is.hail.utils.TempDir
import org.apache.hadoop
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.BeforeClass

object DistributedSuite {
  def getSchedulerHost: String = {
    val schedulerHost = System.getenv("HAIL_TEST_SCHEDULER_HOST")
    if (schedulerHost == null) "localhost" else schedulerHost
  }

  lazy val hc: HailContext = {
    val hc = HailContext.createDistributed(getSchedulerHost, logFile = "/tmp/hail.log")
    if (System.getenv("HAIL_ENABLE_CPP_CODEGEN") != null)
      hc.flags.set("cpp", "1")
    hc.checkRVDKeys = true
    hc
  }
}

class DistributedSuite extends TestNGSuite {

  def hc: HailContext = DistributedSuite.hc

  def backend: DistributedBackend = hc.backend.asInstanceOf[DistributedBackend]

  @BeforeClass def ensureHailContextInitialized() { hc }

  def hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  lazy val tmpDir: TempDir = TempDir(hadoopConf)
}