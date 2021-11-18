package is.hail.fs.gs

import java.io.FileInputStream

import is.hail.fs.FSSuite
import is.hail.io.fs.GoogleStorageFS
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{Test, BeforeClass}
import org.testng.SkipException

class GoogleStorageFSSuite extends TestNGSuite with FSSuite {
  @BeforeClass
  def beforeclass(): Unit = {
    if (System.getenv("CLOUD") != "gcp") {
      throw new SkipException("This test suite is only run in GCP.");
    }
  }

  val bucket: String = System.getenv("HAIL_TEST_GCS_BUCKET")

  assert(bucket != null)

  val root: String = s"gs://$bucket"

  val fsResourcesRoot: String = System.getenv("HAIL_FS_TEST_CLOUD_RESOURCES_URI")

  assert(fsResourcesRoot != null)

  lazy val fs = new GoogleStorageFS()

  lazy val tmpdir: String = s"gs://$bucket/tmp"

  @Test def testDropTailingSlash(): Unit = {
    import GoogleStorageFS._

    assert(dropTrailingSlash("") == "")
    assert(dropTrailingSlash("/foo/bar") == "/foo/bar")
    assert(dropTrailingSlash("foo/bar/") == "foo/bar")
    assert(dropTrailingSlash("/foo///") == "/foo")
    assert(dropTrailingSlash("///") == "")
  }
}
