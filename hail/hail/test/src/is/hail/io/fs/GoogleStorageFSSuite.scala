package is.hail.io.fs

import is.hail.services.oauth2.GoogleCloudCredentials

import org.scalatest
import org.scalatestplus.testng.TestNGSuite
import org.testng.SkipException
import org.testng.annotations.{BeforeClass, Test}

class GoogleStorageFSSuite extends TestNGSuite with FSSuite {
  @BeforeClass
  def beforeclass(): scalatest.Assertion = {
    if (System.getenv("HAIL_CLOUD") != "gcp") {
      throw new SkipException("This test suite is only run in GCP.");
    } else {
      assert(root != null)
      assert(fsResourcesRoot != null)
    }
  }

  override lazy val fs: FS =
    new GoogleStorageFS(GoogleCloudCredentials(None, GoogleStorageFS.RequiredOAuthScopes), None)

  @Test def testMakeQualified(): scalatest.Assertion = {
    val qualifiedFileName = "gs://bucket/path"
    assert(fs.makeQualified(qualifiedFileName) == qualifiedFileName)

    val unqualifiedFileName = "not-gs://bucket/path"
    assertThrows[IllegalArgumentException] {
      fs.makeQualified(unqualifiedFileName)
    }
  }
}
