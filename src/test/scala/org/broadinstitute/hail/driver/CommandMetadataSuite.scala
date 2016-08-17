package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class CommandMetadataSuite extends SparkSuite {
  @Test def buildIndex() {
    val outputFile = tmpDir.createTempFile(prefix = "commandOptions", extension = ".json")
    CommandMetadata.run(State(sc, sqlContext, null), Array("-o", outputFile))
  }
}
