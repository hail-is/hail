package is.hail.io.fs

import is.hail.services.oauth2.AzureCloudCredentials

import org.scalatestplus.testng.TestNGSuite
import org.testng.SkipException
import org.testng.annotations.{BeforeClass, Test}

class AzureStorageFSSuite extends TestNGSuite with FSSuite {
  @BeforeClass
  def beforeclass(): Unit =
    if (
      System.getenv("HAIL_CLOUD") != "azure" ||
      root == null ||
      fsResourcesRoot == null
    )
      throw new SkipException("skip")

  override lazy val fs: FS =
    new AzureStorageFS(
      AzureCloudCredentials(None)
        .scoped(AzureStorageFS.RequiredOAuthScopes)
    )

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "https://account.blob.core.windows.net/container/path"
    assert(fs.makeQualified(qualifiedFileName) == qualifiedFileName)

    val unqualifiedFileName = "https://account/container/path"
    assertThrows[IllegalArgumentException] {
      fs.makeQualified(unqualifiedFileName)
    }
  }
}
