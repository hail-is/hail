package is.hail.io.fs

import is.hail.services.oauth2.AzureCloudCredentials

import org.testng.SkipException
import org.testng.annotations.{BeforeClass, Test}

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

  override lazy val fs: FS =
    new AzureStorageFS(AzureCloudCredentials(None, AzureStorageFS.RequiredOAuthScopes))

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "https://account.blob.core.windows.net/container/path"
    assert(fs.makeQualified(qualifiedFileName) == qualifiedFileName)

    val unqualifiedFileName = "https://account/container/path"
    try
      fs.makeQualified(unqualifiedFileName)
    catch {
      case _: IllegalArgumentException =>
        return
    }
    assert(false)
  }
}
