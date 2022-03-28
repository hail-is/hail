package is.hail.fs.gs

import java.io.FileInputStream
import is.hail.fs.FSSuite
import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.io.fs.GoogleStorageFS
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{BeforeClass, Test}
import org.testng.SkipException

class GoogleStorageFSSuite extends TestNGSuite with FSSuite {
  @BeforeClass
  def beforeclass(): Unit = {
    if (System.getenv("HAIL_CLOUD") != "gcp") {
      throw new SkipException("This test suite is only run in GCP.");
    } else {
      assert(hail_test_storage_uri != null)
      assert(fsResourcesRoot != null)
    }
  }

  val hail_test_storage_uri: String = System.getenv("HAIL_TEST_STORAGE_URI")

  val root: String = hail_test_storage_uri

  val fsResourcesRoot: String = System.getenv("HAIL_FS_TEST_CLOUD_RESOURCES_URI")

  lazy val fs = {
    val gac = System.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if (gac == null) {
      new GoogleStorageFS()
    } else {
      new GoogleStorageFS(
        Some(new String(IOUtils.toByteArray(new FileInputStream(gac)))))
    }
  }

  lazy val tmpdir: String = hail_test_storage_uri

  @Test def testDropTailingSlash(): Unit = {
    assert(dropTrailingSlash("") == "")
    assert(dropTrailingSlash("/foo/bar") == "/foo/bar")
    assert(dropTrailingSlash("foo/bar/") == "foo/bar")
    assert(dropTrailingSlash("/foo///") == "/foo")
    assert(dropTrailingSlash("///") == "")
  }
}
