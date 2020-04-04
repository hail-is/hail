package is.hail.fs.gs

import java.io.FileInputStream

import is.hail.fs.FSSuite
import is.hail.io.fs.GoogleStorageFS
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class GoogleStorageFSSuite extends TestNGSuite with FSSuite {
  val bucket: String = "hail-test-dmk9z"

  val root: String = s"gs://$bucket"

  val fsResourcesRoot: String = System.getenv("HAIL_GS_FS_TEST_RESOURCES")

  private val keyFile = "/test-gsa-key/key.json"

  lazy val fs = new GoogleStorageFS(
    new String(IOUtils.toByteArray(new FileInputStream(keyFile))))

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
