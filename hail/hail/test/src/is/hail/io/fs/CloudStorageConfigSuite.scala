package is.hail.io.fs

import is.hail.TestUtils._

import org.junit.jupiter.api.Test

class CloudStorageConfigSuite {
  @Test def testAWSUsesHadoopFallbackOnly(): Unit = {
    val config = CloudStorageConfig.readEnv(None, Map("HAIL_CLOUD" -> "aws"))
    assertEq(config, CloudStorageConfig())
    assert(config.google.isEmpty)
    assert(config.azure.isEmpty)
  }
}
