package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class BuildDocsSuite extends SparkSuite {
  @Test def buildIndex() {
    val outputFile = tmpDir.createTempFile(prefix = "commandOptions", extension = ".json")
    BuildDocs.run(State(sc, sqlContext, null), Array("-o", outputFile))
  }
}
