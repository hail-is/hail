package is.hail.io.fs

import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.SkipException
import org.testng.annotations.{BeforeClass, Test}

import java.io.FileInputStream


class AzureStorageFSSuite extends FSSuite {
  @BeforeClass
  def beforeclass(): Unit = {
    if (System.getenv("HAIL_CLOUD") != "azure") {
      throw new SkipException("This test suite is only run in Azure.");
    } else {
      assert(root != null)
      assert(fsResourcesRoot != null)
    }
  }

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
