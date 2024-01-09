package is.hail.utils

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class SemanticVersionSuite extends TestNGSuite {
  @Test def testOrdering() {
    val versions = Array(
      SemanticVersion(1, 1, 0),
      SemanticVersion(1, 1, 1),
      SemanticVersion(1, 2, 2),
      SemanticVersion(1, 2, 3),
      SemanticVersion(1, 3, 0),
      SemanticVersion(1, 3, 1),
      SemanticVersion(2, 0, 0),
      SemanticVersion(2, 0, 1),
    )

    versions.zipWithIndex.foreach { case (v, i) =>
      (0 until i).foreach(j => assert(v > versions(j)))

      (i + 1 until versions.length).foreach(j => assert(v < versions(j)))
    }
  }
}
