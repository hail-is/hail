package is.hail

import is.hail.backend.distributed.DistributedBackend
import is.hail.scheduler.Executor
import is.hail.utils.TempDir
import org.apache.hadoop
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.BeforeClass

class Executors(hostname: String, cores: Int) {
  private val t = new Thread (new Runnable {
    def run() { Executor.main(Array(hostname, cores.toString)) }
  })
  t.start()
}

object DistributedSuite {

  lazy val executors: Array[Executors] = Array.fill(8)(new Executors("localhost", 1))

  lazy val hc: HailContext = {
    executors
    val hc = HailContext.createDistributed("localhost", logFile = "/tmp/hail.log")
    if (System.getenv("HAIL_ENABLE_CPP_CODEGEN") != null)
      hc.flags.set("cpp", "1")
    hc.checkRVDKeys = true
    hc
  }
}

class DistributedSuite extends TestNGSuite {

  def hc: HailContext = DistributedSuite.hc

  def backend: DistributedBackend = hc.backend.asInstanceOf[DistributedBackend]

  @BeforeClass def ensureHailContextInitialized() {
    hc
  }

  def hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  lazy val tmpDir: TempDir = TempDir(hadoopConf)
}