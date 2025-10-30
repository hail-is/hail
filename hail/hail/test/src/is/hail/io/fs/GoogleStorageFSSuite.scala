package is.hail.io.fs

import is.hail.services.oauth2.GoogleCloudCredentials

import org.scalatestplus.testng.TestNGSuite
import org.testng.SkipException
import org.testng.annotations.{BeforeClass, Test}

class GoogleStorageFSSuite extends TestNGSuite with FSSuite {
  @BeforeClass
  def beforeclass(): Unit =
    if (
      System.getenv("HAIL_CLOUD") != "gcp" ||
      root == null ||
      fsResourcesRoot == null
    )
      throw new SkipException("skip")

  override lazy val fs: FS =
    new GoogleStorageFS(
      GoogleCloudCredentials(None)
        .scoped(GoogleStorageFS.RequiredOAuthScopes),
      None,
    )

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "gs://bucket/path"
    assert(fs.makeQualified(qualifiedFileName) == qualifiedFileName)

    val unqualifiedFileName = "not-gs://bucket/path"
    assertThrows[IllegalArgumentException] {
      fs.makeQualified(unqualifiedFileName)
    }
  }
}
