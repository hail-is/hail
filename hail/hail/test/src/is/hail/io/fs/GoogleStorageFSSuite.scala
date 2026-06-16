package is.hail.io.fs

import is.hail.TestUtils._
import is.hail.services.oauth2.GoogleCloudCredentials

import org.junit.jupiter.api.{BeforeAll, Test}
import org.junit.jupiter.api.Assumptions.assumeTrue

class GoogleStorageFSSuite extends FSSuite {
  @BeforeAll
  def beforeclass(): Unit =
    assumeTrue(
      System.getenv("HAIL_CLOUD") == "gcp" &&
        root != null &&
        fsResourcesRoot != null,
      "not in GCP",
    )

  override lazy val fs: FS =
    new GoogleStorageFS(
      GoogleCloudCredentials(None)
        .scoped(GoogleStorageFS.RequiredOAuthScopes),
      None,
    )

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "gs://bucket/path"
    assertEq(fs.makeQualified(qualifiedFileName), qualifiedFileName)

    val unqualifiedFileName = "not-gs://bucket/path"
    intercept[IllegalArgumentException] {
      fs.makeQualified(unqualifiedFileName)
    }: Unit
  }
}
