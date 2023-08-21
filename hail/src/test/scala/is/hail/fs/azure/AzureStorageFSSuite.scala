package is.hail.fs.azure

import is.hail.fs.BlobStorageFSSuite
import is.hail.io.fs.AzureStorageFS
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.SkipException
import org.testng.annotations.{BeforeClass, Test}

import java.io.FileInputStream


class AzureStorageFSSuite extends BlobStorageFSSuite {
  @BeforeClass
  def beforeclass(): Unit = {
    if (System.getenv("HAIL_CLOUD") != "azure") {
      throw new SkipException("This test suite is only run in Azure.");
    } else {
      assert(hail_test_storage_uri != null)
      assert(fsResourcesRoot != null)
    }
  }

  val hail_test_storage_uri: String = System.getenv("HAIL_TEST_STORAGE_URI")

  val root: String = hail_test_storage_uri

  val fsResourcesRoot: String = System.getenv("HAIL_FS_TEST_CLOUD_RESOURCES_URI")

  lazy val tmpdir: String = hail_test_storage_uri

  lazy val fs = {
    val aac = System.getenv("AZURE_APPLICATION_CREDENTIALS")
    if (aac == null) {
    new AzureStorageFS()
    }
    else {
      new AzureStorageFS(
        Some(new String(IOUtils.toByteArray(new FileInputStream(aac)))))
    }
  }

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "https://account.blob.core.windows.net/container/path"
    assert(fs.makeQualified(qualifiedFileName) == qualifiedFileName)

    val unqualifiedFileName = "https://account/container/path"
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
