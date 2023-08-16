package is.hail.fs.gs

import java.io.FileInputStream
import is.hail.fs.BlobStorageFSSuite
import is.hail.io.fs.GoogleStorageFS
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{BeforeClass, Test}
import org.testng.SkipException

class GoogleStorageFSSuite extends BlobStorageFSSuite {
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

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "gs://bucket/path"
    assert(fs.makeQualified(qualifiedFileName) == qualifiedFileName)

    val unqualifiedFileName = "not-gs://bucket/path"
    try {
      fs.makeQualified(unqualifiedFileName)
    }
    catch {
      case _: IllegalArgumentException =>
        return
    }
    assert(false)
  }
}
