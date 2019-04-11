package is.hail

import org.apache.spark.SparkContext
import org.testng.annotations.Test

class HailContextSuite {
  @Test def testGetOrCreate(): Unit = {
    val hc = HailContext.getOrCreate(
      sc = new SparkContext(
        HailContext.createSparkConf(
          appName = "Hail.TestNG",
          master = Option(System.getProperty("hail.master")),
          local = "local[2]",
          blockSize = 0)
          .set("spark.unsafe.exceptionOnMemoryLeak", "true")),
      logFile = "/tmp/hail.log")
    val hc2 = HailContext.getOrCreate()
    assert(hc == hc2)
  }
}
