package is.hail.io.fs

import is.hail.services.oauth2.GoogleCloudCredentials

import org.scalatestplus.testng.TestNGSuite
import org.testng.SkipException
import org.testng.annotations.{BeforeClass, Test}

class GoogleStorageFSSuite extends TestNGSuite with FSSuite {
  @BeforeClass
  def beforeclass(): Unit = {
    if (System.getenv("HAIL_CLOUD") != "gcp") {
      throw new SkipException("This test suite is only run in GCP.");
    } else {
      assert(root != null)
      assert(fsResourcesRoot != null)
    }
  }

  lazy val fs = new GoogleStorageFS(GoogleCloudCredentials(None), None)

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "gs://bucket/path"
    assert(fs.makeQualified(qualifiedFileName) == qualifiedFileName)

    val unqualifiedFileName = "not-gs://bucket/path"
    try
      fs.makeQualified(unqualifiedFileName)
    catch {
      case _: IllegalArgumentException =>
        return
    }
    assert(false)
  }
}
